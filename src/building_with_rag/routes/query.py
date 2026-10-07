"""Query endpoint: semantic mode is real; every other mode uses the shared run_pattern placeholder."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import RetrievalError, run_semantic

router = APIRouter()


@router.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest):
    if request.pattern is Pattern.SEMANTIC:
        try:
            return run_semantic(request)
        except RetrievalError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": {"code": exc.code, "message": exc.message}},
            )
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
