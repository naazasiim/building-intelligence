"""Hybrid re-ranking: bounded hybrid candidates re-scored by one provider call, then selected."""

import logging
import math
import time

import httpx

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.registry import Pattern
from building_with_rag.retrieval import semantic as sem
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.settings import get_settings

log = logging.getLogger(__name__)

MODE = Pattern.HYBRID_RERANKED.value
MAX_CANDIDATES = 20
RERANK_NOTE = "Re-ranking scores rank only; they do not prove a passage is correct or answers the question."
UPSTREAM_MESSAGE = "Re-ranking request failed; no re-ranked result returned."


def _check_scope(request: QueryRequest) -> None:
    problems = []
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        problems.append("caller_id is not accepted for this server's caller")
    if request.required_acts:
        problems.append(f"required_acts is not supported by {MODE}")
    if request.chapter:
        problems.append(f"chapter is not supported by {MODE}")
    if problems:
        raise sem.RetrievalError(422, "invalid_request", "; ".join(problems) + ".")


def _config():
    s = get_settings()
    for name, value in (
        ("RERANK_API_KEY", s.rerank_api_key),
        ("RERANK_API_BASE_URL", s.rerank_api_base_url),
        ("RERANK_MODEL_NAME", s.rerank_model_name),
    ):
        if not str(value).strip():
            raise sem._not_ready(f"{name} is not set; {MODE} requires it.")
    if s.rerank_request_timeout_seconds < 1:
        raise sem._not_ready("RERANK_REQUEST_TIMEOUT_SECONDS must be at least 1.")
    for name, value, upper in (
        ("RERANK_CANDIDATE_LIMIT", s.rerank_candidate_limit, MAX_CANDIDATES),
        ("RERANK_SEND_LIMIT", s.rerank_send_limit, s.rerank_candidate_limit),
        ("RERANK_RETURN_LIMIT", s.rerank_return_limit, s.rerank_send_limit),
    ):
        if not 1 <= value <= upper:
            raise sem._not_ready(f"{name} is invalid; need 1 <= RETURN <= SEND <= CANDIDATE <= {MAX_CANDIDATES}.")
    return s


def _document(chunk: RetrievedChunk) -> str:
    return f"{chunk.heading}\n{chunk.text}"


def _validate_reply(body, sent: int) -> tuple[list[dict], int | None]:
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        raise sem._upstream(UPSTREAM_MESSAGE)
    seen: set[int] = set()
    scored = []
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        score = item.get("relevance_score") if isinstance(item, dict) else None
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < sent
            or index in seen
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise sem._upstream(UPSTREAM_MESSAGE)
        seen.add(index)
        scored.append({"index": index, "relevance_score": float(score)})
    usage = body.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return scored, tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None


def call_reranker(query: str, documents: list[str]) -> tuple[list[dict], int | None]:
    """One provider call. Returns validated [{index, relevance_score}] and usage tokens."""
    s = get_settings()
    try:
        response = httpx.post(
            f"{s.rerank_api_base_url.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {s.rerank_api_key}"},
            json={"model": s.rerank_model_name, "query": query, "documents": documents},
            timeout=float(s.rerank_request_timeout_seconds),
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("rerank provider error: %s", type(exc).__name__)
        raise sem._upstream(UPSTREAM_MESSAGE) from None
    return _validate_reply(body, len(documents))


def select_evidence(
    candidates: list[RetrievedChunk],
    scores: list[dict],
    send_limit: int,
    return_limit: int,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Pure selection. Returns (final results by rerank_rank, omitted in fused_rank order)."""
    inf = math.inf
    sent = candidates[:send_limit]
    omitted = [c.model_copy(update={"omitted_reason": "not_sent_to_reranker"}) for c in candidates[send_limit:]]
    by_index = {s["index"]: s["relevance_score"] for s in scores}
    scored = [(c, by_index[i]) for i, c in enumerate(sent) if i in by_index]
    scored.sort(key=lambda p: (-p[1], p[0].fused_rank if p[0].fused_rank is not None else inf))
    ranked = [
        c.model_copy(update={"rerank_score": score, "rerank_rank": rank, "score": score})
        for rank, (c, score) in enumerate(scored, start=1)
    ]
    final = ranked[:return_limit]
    omitted += [c.model_copy(update={"omitted_reason": "below_return_limit"}) for c in ranked[return_limit:]]
    omitted.sort(key=lambda c: c.fused_rank if c.fused_rank is not None else inf)
    return final, omitted


def run_hybrid_reranked(request: QueryRequest) -> QueryResult:
    _check_scope(request)
    s = _config()
    hybrid_request = request.model_copy(update={"pattern": Pattern.HYBRID, "limit": s.rerank_candidate_limit})
    hybrid_result = run_hybrid(hybrid_request)
    ht = hybrid_result.trace
    candidates = hybrid_result.results
    trace = {
        "mode": MODE,
        "query": request.question,
        "filters": ht.get("filters"),
        "caller_id": ht.get("caller_id"),
        "result_count": 0,
        "hybrid": {
            k: ht.get(k) for k in ("embedding", "semantic", "keyword", "fusion", "contribution", "unresolved_hits")
        },
        "rerank": {
            "model": s.rerank_model_name,
            "candidate_limit": s.rerank_candidate_limit,
            "send_limit": s.rerank_send_limit,
            "return_limit": s.rerank_return_limit,
            "candidates": len(candidates),
            "sent": 0,
            "returned": 0,
            "omitted_before": 0,
            "omitted_after": 0,
            "latency_ms": 0.0,
        },
    }
    if not candidates:
        return QueryResult(
            pattern=MODE,
            status="no_results",
            message="No passages matched in either route; nothing to re-rank. " + RERANK_NOTE,
            trace=trace,
        )

    sent = candidates[: s.rerank_send_limit]
    started = time.perf_counter()
    scores, tokens = call_reranker(request.question, [_document(c) for c in sent])
    latency = round((time.perf_counter() - started) * 1000, 1)

    final, omitted = select_evidence(
        candidates, scores, s.rerank_send_limit, min(request.limit, s.rerank_return_limit)
    )
    before = len(candidates) - len(sent)
    info = trace["rerank"]
    info.update(
        sent=len(sent),
        returned=len(final),
        omitted_before=before,
        omitted_after=len(omitted) - before,
        latency_ms=latency,
    )
    if tokens is not None:
        info["usage_tokens"] = tokens
    trace["result_count"] = len(final)
    return QueryResult(
        pattern=MODE,
        status="ok",
        message=f"Returned {len(final)} passage(s) re-ranked from {len(candidates)} hybrid candidate(s). "
        + RERANK_NOTE,
        trace=trace,
        results=final,
        omitted_candidates=omitted,
    )
