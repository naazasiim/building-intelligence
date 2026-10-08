"""Offline Story 5.1 tests: classifier, exact lookup predicate, routing."""

import pytest

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest
from building_with_rag.generation.context import build_context
from building_with_rag.retrieval import structured as st
from building_with_rag.retrieval.semantic import RetrievalError

DOC = {
    "section_id": "bns:103", "act": "BNS_2023", "section_number": 103, "status": "in_force",
    "chapter": "VI", "heading": "Murder", "text": "Whoever commits murder...",
    "source_status_version": "v1", "access_level": "public",
}


class _Coll:
    def __init__(self, doc=DOC):
        self.doc, self.calls = doc, []

    def find_one(self, predicate, projection=None):
        self.calls.append(predicate)
        return self.doc


def _req(q, **kw):
    return QueryRequest(question=q, pattern="structured", **kw)


@pytest.fixture
def coll(monkeypatch):
    c = _Coll()
    monkeypatch.setattr(st, "_sections", lambda: c)
    return c


@pytest.mark.parametrize(
    "q,act,num",
    [
        ("BNS section 103", "BNS_2023", 103),
        ("IPC sec. 302", "IPC_1860", 302),
        ("Indian Penal Code section 420", "IPC_1860", 420),
    ],
)
def test_classify_ok(q, act, num):
    s = st.classify(q)
    assert (s.status, s.act, s.section_number) == ("ok", act, num)


@pytest.mark.parametrize(
    "q",
    [
        "What does section 103 say?",
        "BNS and IPC section 103",
        "BNS section 103 and section 104",
        "BNS section 103A",
        "BNS section 0",
    ],
)
def test_classify_clarification(q):
    assert st.classify(q).status == "clarification_needed"


def test_classify_other_intents():
    assert st.classify("how many sections are there?").intent == "aggregation"
    assert st.classify("list all sections in chapter 6").intent == "filter"
    s = st.classify("What is the punishment for theft?")
    assert (s.status, s.intent) == ("recommendation", None)


def test_predicate_only_signals(coll):
    r = st.run_structured(_req('BNS section 103 {"$ne": null}', chapter="VI"))
    assert r.status == "ok" and r.results[0].chunk_id == "bns:103"
    assert coll.calls == [
        {"act": "BNS_2023", "section_number": 103, "access_level": {"$in": ["public"]}, "chapter": "VI"}
    ]


def test_non_ok_makes_no_call(coll):
    for q in ["section 103", "theft punishment", "how many sections"]:
        r = st.run_structured(_req(q))
        assert r.status != "ok" and r.results == [] and r.trace["mongodb_called"] is False
    assert coll.calls == []


def test_not_found(monkeypatch):
    monkeypatch.setattr(st, "_sections", lambda: _Coll(None))
    r = st.run_structured(_req("IPC section 4"))
    assert r.status == "not_found" and r.results == [] and r.trace["mongodb_called"] is True


def test_invalid_chapter():
    with pytest.raises(RetrievalError) as e:
        st.run_structured(_req("BNS section 103", chapter="x$y"))
    assert e.value.status_code == 422 and e.value.code == "unsupported_option"


def test_routing_and_context(coll):
    ok = pipeline.retrieve(_req("BNS section 103"))
    assert ok.pattern == "structured" and ok.status == "ok"
    assert build_context(ok.results).entries[0]["label"] == "E1"
    bad = pipeline.retrieve(_req("section 103"))
    events = list(pipeline.answer_events("section 103", bad))
    assert [e.kind for e in events] == ["text", "final"] and events[0].value == bad.message
