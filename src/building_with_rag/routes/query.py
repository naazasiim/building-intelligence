"""Query endpoint: semantic mode is real; every other mode uses the shared run_pattern placeholder."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.answer import generate_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import RetrievalError, run_semantic

router = APIRouter()

_OUTCOME_NOTES = {
    "answered": "A grounded answer was generated from the cited passages.",
    "insufficient_evidence": "The passages were insufficient to answer; no answer was generated.",
    "unavailable": "The generation model was unavailable; no answer was generated.",
    "malformed": "The generation model returned an unusable response; no answer was generated.",
}


@router.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest):
    if request.pattern is Pattern.SEMANTIC:
        try:
            result = run_semantic(request)
        except RetrievalError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": {"code": exc.code, "message": exc.message}},
            )
        if request.generate_answer:
            result.generation = generate_answer(request.question, result)
            result.message += " " + _OUTCOME_NOTES[result.generation.outcome]
        return result
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
