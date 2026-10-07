import json

from building_with_rag.contracts import RetrievedChunk
from building_with_rag.generation.answer import parse_answer
from building_with_rag.generation.context import build_context


def _parse(payload):
    chunk = RetrievedChunk(
        chunk_id="c1", section_id="bns:303", act="BNS_2023", text="Theft text", heading="Theft", score=0.9
    )
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return parse_answer(content, build_context([chunk]), "m", {})


def _answered(claims):
    return {"outcome": "answered", "answer": "a", "claims": claims, "reason": "r"}


def test_unknown_label_malformed():
    r = _parse(_answered([{"text": "t", "evidence": ["E9"]}]))
    assert r.outcome == "malformed" and r.text == ""


def test_non_json_malformed():
    assert _parse("not json").outcome == "malformed"


def test_answered_without_claims_malformed():
    assert _parse(_answered([])).outcome == "malformed"


def test_valid_answer_resolves_citations():
    r = _parse("```json\n" + json.dumps(_answered([{"text": "t", "evidence": ["E1"]}])) + "\n```")
    assert r.outcome == "answered"
    assert r.citations[0].chunk_id == "c1" and r.citations[0].section_id == "bns:303"
    assert r.supporting_passages[0].chunk_id == "c1"
