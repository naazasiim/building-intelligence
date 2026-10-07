"""Query endpoint: shared retrieve + answer_events path (semantic is real; other modes placeholder)."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.pipeline import QueryError, answer_events, drain, retrieve

router = APIRouter()

_OUTCOME_NOTES = {
    "answered": "A grounded answer was generated from the cited passages.",
    "insufficient_evidence": "The passages were insufficient to answer; no answer was generated.",
    "unavailable": "The generation model was unavailable; no validated answer was generated.",
    "malformed": "The generation model returned an unusable response; no answer was generated.",
}
_LOW_CONFIDENCE_NOTE = (
    "The answer failed the evidence check; no validated answer was produced "
    "(see generation.issues and draft_answer)."
)


@router.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest):
    try:
        result = retrieve(request)
    except QueryError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )
    if request.generate_answer:
        result.generation = drain(answer_events(request.question, result))
        if result.generation is not None:
            note = (
                _LOW_CONFIDENCE_NOTE
                if result.generation.confidence == "low"
                else _OUTCOME_NOTES[result.generation.outcome]
            )
            result.message += " " + note
    return result
