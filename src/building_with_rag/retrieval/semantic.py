"""Semantic retrieval: embed the question, run MongoDB $vectorSearch, return source passages."""

import threading

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from voyageai import Client as VoyageClient
from voyageai.error import VoyageError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion.mongodb_schema import (
    CHUNKS_COLLECTION,
    DEFAULT_ACCESS_LEVEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDINGS_COLLECTION,
    SECTIONS_COLLECTION,
    VECTOR_INDEX_NAME,
)
from building_with_rag.settings import get_settings

MIN_NUM_CANDIDATES = 50
MAX_NUM_CANDIDATES = 200
NUM_CANDIDATES_MULTIPLIER = 10  # numCandidates = clamp(limit * 10, 50, 200), never below limit

MONGO_TIMEOUT_MS = 5000
VOYAGE_TIMEOUT_SECONDS = 15.0

SCORE_NOTE = "Scores rank vector similarity only; they do not prove a passage is correct or answers the question."


class RetrievalError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


_lock = threading.Lock()
_mongo: MongoClient | None = None
_voyage: VoyageClient | None = None
_ready = False


def _not_ready(message: str) -> RetrievalError:
    return RetrievalError(503, "retrieval_not_ready", message)


def _upstream(message: str) -> RetrievalError:
    return RetrievalError(502, "retrieval_upstream_error", message)


def _clients() -> tuple[MongoClient, VoyageClient]:
    global _mongo, _voyage
    settings = get_settings()
    if not settings.mongodb_uri:
        raise _not_ready("MONGODB_URI is not set.")
    if not settings.voyage_api_key:
        raise _not_ready("VOYAGE_API_KEY is not set.")
    with _lock:
        if _mongo is None:
            _mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
                socketTimeoutMS=MONGO_TIMEOUT_MS * 3,
            )
        if _voyage is None:
            _voyage = VoyageClient(
                api_key=settings.voyage_api_key, max_retries=1, timeout=VOYAGE_TIMEOUT_SECONDS
            )
        return _mongo, _voyage


def _check_ready(db) -> None:
    """Cheap readiness check; a positive result is cached per process."""
    global _ready
    if _ready:
        return
    indexes = list(db[EMBEDDINGS_COLLECTION].list_search_indexes(VECTOR_INDEX_NAME))
    if not indexes:
        raise _not_ready(
            f"Vector index '{VECTOR_INDEX_NAME}' not found; run Story 2.2 ingestion "
            "(uv run python -m building_with_rag.ingestion.ingest)."
        )
    if not indexes[0].get("queryable"):
        raise _not_ready(f"Vector index '{VECTOR_INDEX_NAME}' is not queryable yet; retry shortly.")
    sample = db[EMBEDDINGS_COLLECTION].find_one({}, {"model": 1, "dimensions": 1})
    if sample is None:
        raise _not_ready("The embeddings collection is empty; run Story 2.2 ingestion.")
    if sample.get("model") != EMBEDDING_MODEL or sample.get("dimensions") != EMBEDDING_DIMENSIONS:
        raise _not_ready(
            f"Stored embeddings do not match {EMBEDDING_MODEL}/{EMBEDDING_DIMENSIONS}; re-run ingestion."
        )
    _ready = True


def validate_request(request: QueryRequest) -> None:
    """Reject scope the semantic mode cannot honour (HTTP 422)."""
    problems = []
    effective_caller = get_settings().webui_demo_caller_id
    if request.caller_id is not None and request.caller_id != effective_caller:
        problems.append("caller_id is not accepted for this server's caller")
    if request.required_acts:
        problems.append("required_acts is not supported by semantic mode")
    if request.chapter:
        problems.append("chapter is not supported by semantic mode")
    if problems:
        raise RetrievalError(422, "invalid_request", "; ".join(problems) + ".")


def effective_filters(request: QueryRequest) -> dict[str, list[str]]:
    """Server fixes access_level; caller lists only narrow (empty list = no narrowing)."""
    caller = request.filters
    filters: dict[str, list[str]] = {"access_level": [DEFAULT_ACCESS_LEVEL]}
    if caller and caller.act:
        filters["act"] = list(dict.fromkeys(caller.act))
    if caller and caller.status:
        filters["status"] = list(dict.fromkeys(caller.status))
    return filters


def _mongo_filter(filters: dict[str, list[str]]) -> dict:
    return {"$and": [{field: {"$in": values}} for field, values in filters.items()]}


def num_candidates(limit: int) -> int:
    return max(limit, min(MAX_NUM_CANDIDATES, max(MIN_NUM_CANDIDATES, limit * NUM_CANDIDATES_MULTIPLIER)))


def _embed_query(vc: VoyageClient, question: str) -> list[float]:
    vector = vc.embed(texts=[question], model=EMBEDDING_MODEL, input_type="query").embeddings[0]
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise _not_ready(
            f"Voyage returned {len(vector)} dimensions, expected {EMBEDDING_DIMENSIONS}."
        )
    return vector


def _to_chunk(hit: dict, chunk: dict, section: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk["chunk_id"],
        section_id=section["section_id"],
        act=section.get("act") or chunk.get("act") or "",
        text=chunk.get("text") or "",
        heading=section.get("heading") or "",
        score=float(hit["score"]),
        chunk_index=chunk.get("chunk_index"),
        act_label=section.get("act_label"),
        status=section.get("status"),
        chapter=section.get("chapter"),
        chapter_title=section.get("chapter_title"),
        section_number=section.get("section_number"),
        source_pdf=section.get("source_pdf"),
        source_sha256=section.get("source_sha256"),
        needs_review=section.get("needs_review"),
    )


def run_semantic(request: QueryRequest) -> QueryResult:
    global _ready
    validate_request(request)
    filters = effective_filters(request)
    limit = request.limit
    candidates = num_candidates(limit)
    question = request.question  # already trimmed by QueryRequest
    caller_id = get_settings().webui_demo_caller_id
    mongo_filter = _mongo_filter(filters)

    trace = {
        "mode": "semantic",
        "query": question,
        "embedding": {"model": EMBEDDING_MODEL, "input_type": "query", "dimensions": EMBEDDING_DIMENSIONS},
        "index": VECTOR_INDEX_NAME,
        "limit": limit,
        "num_candidates": candidates,
        "filters": filters,
        "caller_id": caller_id,
        "result_count": 0,
        "ignored": (
            ["generate_answer: no answer is generated in semantic retrieval; generation is null"]
            if request.generate_answer
            else []
        ),
        "unresolved_hits": 0,
    }

    mc, vc = _clients()
    try:
        db = mc[get_settings().mongodb_db_name]
        _check_ready(db)
        if db[EMBEDDINGS_COLLECTION].find_one(mongo_filter, {"_id": 1}) is None:
            return QueryResult(
                pattern="semantic",
                status="no_results",
                message="No indexed passages match the requested filters. " + SCORE_NOTE,
                trace=trace,
            )
        vector = _embed_query(vc, question)
        hits = list(
            db[EMBEDDINGS_COLLECTION].aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": VECTOR_INDEX_NAME,
                            "path": "vector",
                            "queryVector": vector,
                            "numCandidates": candidates,
                            "limit": limit,
                            "filter": mongo_filter,
                        }
                    },
                    {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
                ]
            )
        )
        chunk_ids = [h["chunk_id"] for h in hits]
        chunks = {
            c["_id"]: c for c in db[CHUNKS_COLLECTION].find({"_id": {"$in": chunk_ids}})
        }
        section_ids = list({c["section_id"] for c in chunks.values() if c.get("section_id")})
        sections = {
            s["_id"]: s for s in db[SECTIONS_COLLECTION].find({"_id": {"$in": section_ids}})
        }
    except RetrievalError:
        _ready = False
        raise
    except (PyMongoError, VoyageError) as exc:
        _ready = False
        raise _upstream(f"Retrieval backend error ({type(exc).__name__}); try again.") from None

    results: list[RetrievedChunk] = []
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk.get("section_id")) if chunk else None
        if chunk is None or section is None:
            trace["unresolved_hits"] += 1
            continue
        results.append(_to_chunk(hit, chunk, section))
    trace["result_count"] = len(results)

    if not results:
        return QueryResult(
            pattern="semantic",
            status="no_results",
            message="No passages could be returned for this query. " + SCORE_NOTE,
            trace=trace,
        )
    return QueryResult(
        pattern="semantic",
        status="ok",
        message=f"Returned {len(results)} passage(s) ranked by similarity. " + SCORE_NOTE,
        trace=trace,
        results=results,
    )
