from .models import QueryRequest, QueryResult
from .registry import MODES


def run_pattern(request: QueryRequest) -> QueryResult:
    if request.pattern not in MODES:
        raise ValueError(f"unknown pattern: {request.pattern}")
    return QueryResult(
        pattern=request.pattern,
        status="not_implemented",
        message=f"Pattern '{request.pattern}' is not implemented yet.",
    )
