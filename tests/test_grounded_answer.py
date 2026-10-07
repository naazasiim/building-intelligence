from building_with_rag.contracts import RetrievedChunk
from building_with_rag.generation.answer import check_structure, derive_claims
from building_with_rag.generation.context import build_context


def _ctx():
    chunk = RetrievedChunk(
        chunk_id="c1", section_id="bns:303", act="BNS_2023", text="Theft text", heading="Theft", score=0.9
    )
    return build_context([chunk])


def test_claims_split_and_label_tail_merged():
    claims = derive_claims("Intro:\n- BNS 303 defines theft [E1].\n- Punishment is stated. [E1, E2]")
    assert [c.evidence_labels for c in claims] == [["E1"], ["E1", "E2"]]


def test_unknown_label_and_uncited_flagged_not_stripped():
    text = "Theft is covered [E9]. Another claim."
    claims = derive_claims(text)
    checks = [c for c, _ in check_structure(text, claims, _ctx())]
    assert checks == ["citation_labels", "claim_cited"]
    assert claims[0].evidence_labels == ["E9"]


def test_valid_structure_passes():
    text = "BNS 303 defines theft [E1]."
    assert check_structure(text, derive_claims(text), _ctx()) == []
