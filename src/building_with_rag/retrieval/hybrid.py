"""Hybrid retrieval: Atlas Search keyword route + vector route, fused by Reciprocal Rank Fusion."""

from pymongo.errors import PyMongoError
from voyageai.error import VoyageError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion.mongodb_schema import (
    CHUNKS_COLLECTION,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDINGS_COLLECTION,
    KEYWORD_INDEX_NAME,
    SECTIONS_COLLECTION,
    VECTOR_INDEX_NAME,
)
from building_with_rag.retrieval import semantic as sem
from building_with_rag.retrieval.semantic import RetrievalError
from building_with_rag.settings import get_settings

RRF_K = 60
WEIGHTS = {"semantic": 1.0, "keyword": 1.0}
KEYWORD_COMMAND = "uv run python -m building_with_rag.ingestion.keyword_index"
FUSION_NOTE = "The fused score ranks only; it does not prove a passage is correct or answers the question."
_FIELDS = ("semantic_score", "semantic_rank", "keyword_score", "keyword_rank", "fused_score", "fused_rank")

_ready = False


def route_depth(limit: int) -> int:
    return max(limit, min(50, max(20, 4 * limit)))


def fuse(semantic_hits: list[dict], keyword_hits: list[dict], limit: int) -> list[dict]:
    """RRF over ranks. Hits are best-first dicts with chunk_id and score; no I/O."""
    merged: dict[str, dict] = {}
    for route, hits in (("semantic", semantic_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            row = merged.setdefault(hit["chunk_id"], {f: None for f in _FIELDS} | {"chunk_id": hit["chunk_id"], "fused_score": 0.0})
            if row[f"{route}_rank"] is not None:
                continue  # duplicate within a route: keep the best rank
            row[f"{route}_score"] = float(hit["score"])
            row[f"{route}_rank"] = rank
            row["fused_score"] += WEIGHTS[route] / (RRF_K + rank)
    ordered = sorted(
        merged.values(),
        key=lambda r: (
            -r["fused_score"],
            r["semantic_rank"] if r["semantic_rank"] is not None else float("inf"),
            r["chunk_id"],
        ),
    )
    for position, row in enumerate(ordered, start=1):
        row["fused_rank"] = position
    return ordered[:limit]


def _check_ready(db) -> None:
    global _ready
    if _ready:
        return
    sem._check_ready(db)  # vector index queryable, embeddings non-empty and matching
    indexes = list(db[CHUNKS_COLLECTION].list_search_indexes(KEYWORD_INDEX_NAME))
    if not indexes:
        raise sem._not_ready(f"Keyword index '{KEYWORD_INDEX_NAME}' not found; run: {KEYWORD_COMMAND}")
    if not indexes[0].get("queryable"):
        raise sem._not_ready(f"Keyword index '{KEYWORD_INDEX_NAME}' is not queryable yet; retry shortly.")
    _ready = True


def _keyword_pipeline(question: str, filters: dict[str, list[str]], depth: int) -> list[dict]:
    return [
        {
            "$search": {
                "index": KEYWORD_INDEX_NAME,
                "compound": {
                    "must": [{"text": {"query": question, "path": "text"}}],
                    "filter": [{"in": {"path": f, "value": v}} for f, v in filters.items()],
                },
            }
        },
        {"$limit": depth},
        {"$project": {"_id": 0, "chunk_id": 1, "section_id": 1, "score": {"$meta": "searchScore"}}},
    ]


def _reset() -> None:
    global _ready
    _ready = False
    sem._ready = False


def run_hybrid(request: QueryRequest) -> QueryResult:
    sem.validate_request(request)
    filters = sem.effective_filters(request)
    limit = request.limit
    depth = route_depth(limit)
    candidates = sem.num_candidates(depth)
    question = request.question

    trace = {
        "mode": "hybrid",
        "query": question,
        "embedding": {"model": EMBEDDING_MODEL, "input_type": "query", "dimensions": EMBEDDING_DIMENSIONS},
        "filters": filters,
        "caller_id": get_settings().webui_demo_caller_id,
        "result_count": 0,
        "unresolved_hits": 0,
        "semantic": {"index": VECTOR_INDEX_NAME, "limit": depth, "num_candidates": candidates, "hit_count": 0},
        "keyword": {"index": KEYWORD_INDEX_NAME, "path": "text", "operator": "text", "limit": depth, "hit_count": 0},
        "fusion": {"method": "rrf", "k": RRF_K, "weights": WEIGHTS, "route_depth": depth},
        "contribution": {"both": 0, "semantic_only": 0, "keyword_only": 0},
    }

    mc, vc = sem._clients()
    try:
        db = mc[get_settings().mongodb_db_name]
        _check_ready(db)
        vector = sem._embed_query(vc, question)
        semantic_hits = list(
            db[EMBEDDINGS_COLLECTION].aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": VECTOR_INDEX_NAME,
                            "path": "vector",
                            "queryVector": vector,
                            "numCandidates": candidates,
                            "limit": depth,
                            "filter": sem._mongo_filter(filters),
                        }
                    },
                    {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
                ]
            )
        )
        keyword_hits = list(db[CHUNKS_COLLECTION].aggregate(_keyword_pipeline(question, filters, depth)))
        trace["semantic"]["hit_count"] = len(semantic_hits)
        trace["keyword"]["hit_count"] = len(keyword_hits)

        chunk_ids = list({h["chunk_id"] for h in semantic_hits + keyword_hits})
        chunks = {c["_id"]: c for c in db[CHUNKS_COLLECTION].find({"_id": {"$in": chunk_ids}})}
        section_ids = list({c["section_id"] for c in chunks.values() if c.get("section_id")})
        sections = {s["_id"]: s for s in db[SECTIONS_COLLECTION].find({"_id": {"$in": section_ids}})}
    except RetrievalError:
        _reset()
        raise
    except (PyMongoError, VoyageError) as exc:
        _reset()
        raise sem._upstream(f"Retrieval backend error ({type(exc).__name__}); try again.") from None

    # Fuse only resolvable chunks so fused_rank stays contiguous; count the rest.
    resolvable = {
        i for i, c in chunks.items() if c.get("section_id") in sections
    }
    trace["unresolved_hits"] = len(set(chunk_ids) - resolvable)
    fused = fuse(
        [h for h in semantic_hits if h["chunk_id"] in resolvable],
        [h for h in keyword_hits if h["chunk_id"] in resolvable],
        limit,
    )

    results: list[RetrievedChunk] = []
    contribution = trace["contribution"]
    for row in fused:
        chunk = chunks[row["chunk_id"]]
        item = sem._to_chunk({"score": row["fused_score"]}, chunk, sections[chunk["section_id"]])
        for field in _FIELDS:
            setattr(item, field, row[field])
        results.append(item)
        if row["semantic_rank"] is not None and row["keyword_rank"] is not None:
            contribution["both"] += 1
        elif row["semantic_rank"] is not None:
            contribution["semantic_only"] += 1
        else:
            contribution["keyword_only"] += 1
    trace["result_count"] = len(results)

    if not results:
        return QueryResult(
            pattern="hybrid",
            status="no_results",
            message="No passages matched in either route. " + FUSION_NOTE,
            trace=trace,
        )
    return QueryResult(
        pattern="hybrid",
        status="ok",
        message=f"Returned {len(results)} passage(s) fused from keyword and semantic routes. " + FUSION_NOTE,
        trace=trace,
        results=results,
    )
