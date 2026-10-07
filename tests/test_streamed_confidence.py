import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag.app import create_app
from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer
from building_with_rag.pipeline import chat_pieces, drain
from building_with_rag.settings import get_settings

GOOD = "BNS 303 defines theft [E1]."
BAD_CITE = "BNS 303 defines theft [E9]."


class Fake:
    """Scripted provider: each attempt is a list of chunks (an Exception item raises mid-stream)."""

    def __init__(self, attempts, verdict=True):
        self.attempts, self.verdict, self.stream_calls = list(attempts), verdict, 0

    def stream(self, messages):
        self.stream_calls += 1
        for item in self.attempts.pop(0):
            if isinstance(item, Exception):
                raise item
            yield item

    def complete(self, messages):
        return json.dumps({"verdicts": [{"claim": 1, "supported": self.verdict}]})


@pytest.fixture()
def retrieval():
    chunk = RetrievedChunk(
        chunk_id="c1", section_id="bns:303", act="BNS_2023", text="Theft text",
        heading="Theft", score=0.9, section_number=303,
    )
    return QueryResult(pattern="semantic", status="ok", message="m", trace={}, results=[chunk])


@pytest.fixture()
def use(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "generation_api_base_url", "http://fake")
    monkeypatch.setattr(s, "generation_api_key", "k")

    def install(fake):
        monkeypatch.setattr(answer, "_stream_chunks", fake.stream)
        monkeypatch.setattr(answer, "_complete", fake.complete)
        return fake

    return install


def _run(retrieval):
    pieces = list(chat_pieces("q", retrieval))
    return pieces


def test_pass_single_call_stream_matches_text(use, retrieval):
    fake = use(Fake([["BNS 303 ", "defines theft [E1]."]]))
    events = list(answer.stream_answer("q", retrieval))
    final = events[-1].value
    streamed = "".join(e.value for e in events if e.kind == "text")
    assert fake.stream_calls == 1 and streamed == final.text == GOOD
    assert final.confidence == "high" and final.outcome == "answered"
    assert final.citations[0].section_id == "bns:303"


def test_retry_then_pass(use, retrieval):
    use(Fake([[BAD_CITE], [GOOD]]))
    pieces = _run(retrieval)
    out = "".join(pieces)
    assert out.count("DRAFT \u2014 checking evidence") == 2
    assert "Retrying (attempt 2 of 2)" in out and "confidence: high" in out
    use(Fake([[BAD_CITE], [GOOD]]))
    g = drain(answer.stream_answer("q", retrieval))
    assert [a.status for a in g.attempts] == ["failed", "passed"]
    assert g.issues[0].check == "citation_labels" and g.confidence == "high"


def test_both_fail_low_confidence(use, retrieval):
    use(Fake([[BAD_CITE], [BAD_CITE]]))
    out = "".join(_run(retrieval))
    assert out.rstrip().splitlines()[-3].startswith("DRAFT \u2014 low confidence")
    use(Fake([[BAD_CITE], [BAD_CITE]]))
    g = drain(answer.stream_answer("q", retrieval))
    assert g.outcome == "malformed" and g.text == "" and g.confidence == "low"
    assert g.draft_answer == BAD_CITE and g.low_confidence_reason and len(g.issues) == 2
    assert g.claims[0].evidence_labels == ["E9"]


def test_provider_failure_mid_stream(use, retrieval, monkeypatch):
    use(Fake([["BNS 303 is", answer.ProviderError("timeout")]]))
    g = drain(answer.stream_answer("q", retrieval))
    assert g.outcome == "unavailable" and g.confidence is None and g.draft_answer
    use(Fake([["BNS 303 is", answer.ProviderError("timeout")]]))
    monkeypatch.setattr("building_with_rag.routes.chat.retrieve", lambda r: retrieval)
    client = TestClient(create_app())
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "rag-semantic", "stream": True, "messages": [{"role": "user", "content": "q"}]},
    ) as resp:
        raw = b"".join(resp.iter_bytes()).decode()
    assert resp.status_code == 200 and "unchecked draft" in raw
    assert '"finish_reason": "stop"' in raw and raw.strip().endswith("data: [DONE]")
