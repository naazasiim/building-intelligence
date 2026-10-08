"""Offline Story 4.1 tests: RRF fusion and run_hybrid with faked backends."""

import pytest

from building_with_rag.contracts import QueryRequest
from building_with_rag.registry import Pattern
from building_with_rag.retrieval import hybrid
from building_with_rag.retrieval import semantic as sem


def _h(cid, score=1.0):
    return {"chunk_id": cid, "score": score}


def test_fuse_both_routes_rank_first():
    rows = hybrid.fuse([_h("a"), _h("b")], [_h("c"), _h("b")], 5)
    top = rows[0]
    assert top["chunk_id"] == "b"
    assert top["semantic_rank"] == 2 and top["keyword_rank"] == 2
    assert top["fused_score"] == pytest.approx(2 / (hybrid.RRF_K + 2))
    assert [r["fused_rank"] for r in rows] == [1, 2, 3]


def test_fuse_single_route_leaves_other_none():
    rows = {r["chunk_id"]: r for r in hybrid.fuse([_h("a")], [_h("k")], 5)}
    assert rows["a"]["keyword_score"] is None and rows["a"]["keyword_rank"] is None
    assert rows["k"]["semantic_score"] is None and rows["k"]["semantic_rank"] is None


def test_fuse_tie_order_deterministic():
    # Equal fused scores: semantic_rank first (missing last), then chunk_id.
    rows = hybrid.fuse([_h("z")], [_h("a")], 5)
    assert [r["chunk_id"] for r in rows] == ["z", "a"]
    assert [r["chunk_id"] for r in hybrid.fuse([], [_h("b"), _h("a")], 1)] == ["b"]


class _Coll:
    def __init__(self, rows=None, indexes=None, hits=None):
        self.rows, self.indexes, self.hits = rows or [], indexes or [], hits or []

    def list_search_indexes(self, name=None):
        return iter(self.indexes)

    def find_one(self, *a, **k):
        return {"model": sem.EMBEDDING_MODEL, "dimensions": sem.EMBEDDING_DIMENSIONS}

    def find(self, *a, **k):
        return iter(self.rows)

    def aggregate(self, pipeline):
        return iter(self.hits)


class _DB(dict):
    pass


def _patch(monkeypatch, keyword_indexes):
    chunks = _Coll(
        rows=[{"_id": c, "chunk_id": c, "section_id": "s1", "text": "t"} for c in ("a", "k")],
        indexes=keyword_indexes,
        hits=[_h("k", 3.0)],
    )
    sections = _Coll(rows=[{"_id": "s1", "section_id": "s1", "act": "BNS_2023", "heading": "H"}])
    embeddings = _Coll(indexes=[{"queryable": True}], hits=[_h("a", 0.9)])
    db = _DB(chunks=chunks, sections=sections, embeddings=embeddings)
    monkeypatch.setattr(sem, "_clients", lambda: ({"x": db}, object()))
    monkeypatch.setattr(sem, "_embed_query", lambda vc, q: [0.0] * sem.EMBEDDING_DIMENSIONS)
    monkeypatch.setattr(hybrid, "get_settings", lambda: type("S", (), {"mongodb_db_name": "x", "webui_demo_caller_id": "c"})())
    monkeypatch.setattr(sem, "get_settings", hybrid.get_settings)
    monkeypatch.setattr(hybrid, "_ready", False)
    monkeypatch.setattr(sem, "_ready", False)


def test_run_hybrid_sets_fields(monkeypatch):
    _patch(monkeypatch, [{"queryable": True}])
    result = hybrid.run_hybrid(QueryRequest(question="theft", pattern=Pattern.HYBRID, limit=5))
    assert result.status == "ok"
    for r in result.results:
        assert r.score == r.fused_score
        assert r.semantic_rank or r.keyword_rank
    assert [r.fused_rank for r in result.results] == [1, 2]
    assert result.trace["contribution"] == {"both": 0, "semantic_only": 1, "keyword_only": 1}


def test_run_hybrid_missing_keyword_index(monkeypatch):
    _patch(monkeypatch, [])
    with pytest.raises(sem.RetrievalError) as exc:
        hybrid.run_hybrid(QueryRequest(question="theft", pattern=Pattern.HYBRID, limit=5))
    assert exc.value.status_code == 503 and exc.value.code == "retrieval_not_ready"
