"""Offline Story 4.2 tests: re-ranking selection and run_hybrid_reranked with faked provider/hybrid."""

import httpx
import pytest

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.generation.context import build_context
from building_with_rag.registry import Pattern
from building_with_rag.retrieval import rerank
from building_with_rag.retrieval import semantic as sem


def _chunks(n):
    return [
        RetrievedChunk(
            chunk_id=f"c{i}", section_id=f"s{i}", act="BNS_2023", text=f"t{i}", heading=f"H{i}",
            score=1.0 / i, fused_score=1.0 / i, fused_rank=i,
        )
        for i in range(1, n + 1)
    ]


def _settings(monkeypatch, key="k", cand=4, send=3, ret=2, timeout=5):
    s = type("S", (), {
        "rerank_api_key": key, "rerank_api_base_url": "http://x", "rerank_model_name": "m",
        "rerank_request_timeout_seconds": timeout, "rerank_candidate_limit": cand,
        "rerank_send_limit": send, "rerank_return_limit": ret, "webui_demo_caller_id": "c",
    })()
    monkeypatch.setattr(rerank, "get_settings", lambda: s)


def _fake_hybrid(monkeypatch, n=4):
    calls = []

    def fake(request):
        calls.append(request)
        return QueryResult(
            pattern="hybrid", status="ok" if n else "no_results", message="m",
            trace={"filters": {}, "caller_id": "c"}, results=_chunks(n),
        )

    monkeypatch.setattr(rerank, "run_hybrid", fake)
    return calls


def _provider(monkeypatch, scores):
    calls = []

    def fake(query, documents):
        calls.append((query, documents))
        return [{"index": i, "relevance_score": s} for i, s in enumerate(scores)], 7

    monkeypatch.setattr(rerank, "call_reranker", fake)
    return calls


def _req(limit=5):
    return QueryRequest(question="q", pattern=Pattern.HYBRID_RERANKED, limit=limit)


def test_reorders_and_keeps_fused_fields(monkeypatch):
    _settings(monkeypatch, cand=3, send=3, ret=3)
    _fake_hybrid(monkeypatch, 3)
    _provider(monkeypatch, [0.1, 0.9, 0.5])
    r = rerank.run_hybrid_reranked(_req())
    assert r.status == "ok"
    assert [c.chunk_id for c in r.results] == ["c2", "c3", "c1"]
    assert [c.rerank_rank for c in r.results] == [1, 2, 3]
    assert [c.fused_rank for c in r.results] == [2, 3, 1]
    assert all(c.score == c.rerank_score for c in r.results)
    assert r.trace["rerank"]["usage_tokens"] == 7 and r.omitted_candidates == []


def test_omitted_before_and_after(monkeypatch):
    _settings(monkeypatch)  # 4/3/2
    calls = _fake_hybrid(monkeypatch, 4)
    sent = _provider(monkeypatch, [0.2, 0.9, 0.5])
    r = rerank.run_hybrid_reranked(_req())
    assert calls[0].pattern is Pattern.HYBRID and calls[0].limit == 4
    assert len(sent[0][1]) == 3 and sent[0][1][0] == "H1\nt1"
    assert [c.chunk_id for c in r.results] == ["c2", "c3"]
    o = {c.chunk_id: c for c in r.omitted_candidates}
    assert [c.chunk_id for c in r.omitted_candidates] == ["c1", "c4"]
    assert o["c4"].omitted_reason == "not_sent_to_reranker"
    assert o["c4"].rerank_score is None and o["c4"].rerank_rank is None
    assert o["c1"].omitted_reason == "below_return_limit"
    assert o["c1"].rerank_rank == 3 and o["c1"].rerank_score == 0.2
    t = r.trace["rerank"]
    assert (t["candidates"], t["sent"], t["returned"], t["omitted_before"], t["omitted_after"]) == (4, 3, 2, 1, 1)
    assert all(c.omitted_reason is None for c in r.results)


def test_request_limit_caps_return(monkeypatch):
    _settings(monkeypatch)
    _fake_hybrid(monkeypatch, 4)
    _provider(monkeypatch, [0.2, 0.9, 0.5])
    assert len(rerank.run_hybrid_reranked(_req(limit=1)).results) == 1


def test_empty_key_503_no_calls(monkeypatch):
    _settings(monkeypatch, key="")
    hy = _fake_hybrid(monkeypatch)
    pr = _provider(monkeypatch, [])
    with pytest.raises(sem.RetrievalError) as exc:
        rerank.run_hybrid_reranked(_req())
    assert exc.value.status_code == 503 and exc.value.code == "retrieval_not_ready"
    assert "RERANK_API_KEY" in exc.value.message and not hy and not pr


def test_invalid_limits_503(monkeypatch):
    _settings(monkeypatch, cand=3, send=4, ret=2)
    hy = _fake_hybrid(monkeypatch)
    with pytest.raises(sem.RetrievalError) as exc:
        rerank.run_hybrid_reranked(_req())
    assert exc.value.status_code == 503 and "RERANK_SEND_LIMIT" in exc.value.message and not hy


def test_no_results_skips_provider(monkeypatch):
    _settings(monkeypatch)
    _fake_hybrid(monkeypatch, 0)
    pr = _provider(monkeypatch, [])
    r = rerank.run_hybrid_reranked(_req())
    assert r.status == "no_results" and r.results == [] and not pr


class _Resp:
    def __init__(self, status=200, body=None):
        self.status, self.body = status, body

    def raise_for_status(self):
        if self.status >= 400:
            raise httpx.HTTPStatusError("x", request=None, response=None)

    def json(self):
        return self.body


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.TimeoutException("t"),
        _Resp(500),
        _Resp(200, {"data": []}),
        _Resp(200, {"data": [{"index": 5, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.2}]}),
        _Resp(200, {"data": [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.2}]}),
        _Resp(200, {"data": [{"index": 0}, {"index": 1, "relevance_score": 0.2}]}),
        _Resp(200, {"data": [{"index": 0, "relevance_score": float("nan")}]}),
    ],
)
def test_provider_failures_502_no_fallback(monkeypatch, outcome):
    _settings(monkeypatch)
    _fake_hybrid(monkeypatch, 4)

    def post(*a, **k):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(rerank.httpx, "post", post)
    with pytest.raises(sem.RetrievalError) as exc:
        rerank.run_hybrid_reranked(_req())
    assert exc.value.status_code == 502 and exc.value.code == "retrieval_upstream_error"


def test_context_labels_only_final_and_routing(monkeypatch):
    _settings(monkeypatch)
    _fake_hybrid(monkeypatch, 4)
    _provider(monkeypatch, [0.2, 0.9, 0.5])
    r = pipeline.retrieve(_req())
    assert r.pattern == "hybrid-reranked" and "hybrid-reranked" in pipeline.REAL_PATTERNS
    ctx = build_context(r.results)
    assert [e["chunk_id"] for e in ctx.entries] == ["c2", "c3"]
