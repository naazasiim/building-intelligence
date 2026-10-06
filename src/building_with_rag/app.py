import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import (
    ChatChunkChoice,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDelta,
    ErrorDetail,
    ErrorEnvelope,
    ModelCard,
    ModelList,
    QueryRequest,
    QueryResult,
    SemanticFilters,
)
from .patterns import run_pattern
from .registry import MODEL_TO_MODE, MODES
from .settings import get_settings

app = FastAPI(title="Building with RAG")


def _error(status: int, message: str, etype: str, param: str | None = None) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorDetail(message=message, type=etype, param=param))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/v1/chat/"):
        return _error(400, "Invalid request body.", "invalid_request_error")
    return JSONResponse(status_code=422, content={"detail": "Invalid request."})


@app.exception_handler(StarletteHTTPException)
async def _http_handler(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/v1/"):
        return _error(exc.status_code, str(exc.detail), "invalid_request_error")
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest) -> QueryResult:
    return run_pattern(request)


@app.get("/v1/models", response_model=ModelList)
def models() -> ModelList:
    return ModelList(data=[ModelCard(id=m) for m in MODES.values()])


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest):
    mode = MODEL_TO_MODE.get(body.model)
    if mode is None:
        return _error(404, f"Model '{body.model}' not found.", "invalid_request_error", "model")
    user_msgs = [m for m in body.messages if m.role == "user"]
    if not user_msgs or not user_msgs[-1].content.strip():
        return _error(400, "A user message is required.", "invalid_request_error", "messages")
    opts = body.rag_options
    pattern = (opts.pattern if opts and opts.pattern else mode)
    filters = None
    if opts and (opts.act or opts.status or opts.access_level):
        filters = SemanticFilters(
            act=opts.act, status=opts.status, access_level=opts.access_level
        )
    try:
        req = QueryRequest(
            question=user_msgs[-1].content,
            pattern=pattern,
            caller_id=get_settings().webui_demo_caller_id,
            filters=filters,
            limit=(opts.limit if opts and opts.limit else 5),
            generate_answer=False,
            required_acts=opts.required_acts if opts else [],
            chapter=opts.chapter if opts else None,
        )
    except ValueError:
        return _error(400, "Invalid request.", "invalid_request_error")
    result = run_pattern(req)
    text = result.message
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not body.stream:
        resp = ChatCompletionResponse(
            id=cid,
            created=created,
            model=body.model,
            choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=text))],
        )
        return JSONResponse(content=resp.model_dump())

    def sse():
        def chunk(delta: ChatDelta, finish: str | None = None) -> str:
            c = ChatCompletionChunk(
                id=cid,
                created=created,
                model=body.model,
                choices=[ChatChunkChoice(delta=delta, finish_reason=finish)],
            )
            return f"data: {c.model_dump_json()}\n\n"

        yield chunk(ChatDelta(role="assistant"))
        yield chunk(ChatDelta(content=text))
        yield chunk(ChatDelta(), "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
