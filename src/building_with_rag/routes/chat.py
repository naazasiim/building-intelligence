"""OpenAI-compatible chat adapter.

Maps the selected rag-<pattern> model to the same QueryRequest and shared
retrieve + answer_events path as /v1/query, with server-side demo caller_id and generate_answer.
Supports normal JSON responses and role/content/stop SSE frames; no custom
SSE events. The OpenAI-style error envelope applies before streaming begins.
"""

import hmac
import json
import time
import uuid

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from building_with_rag.contracts import (
    ChatCompletionRequest,
    QueryRequest,
    QueryResult,
    SemanticFilters,
)
from building_with_rag.pipeline import QueryError, chat_pieces, retrieve
from building_with_rag.registry import MODEL_ID_TO_PATTERN
from building_with_rag.settings import get_settings

router = APIRouter()

_FINISH_STOP = "stop"
_DONE = "[" + "DONE" + "]"


def _error(status: int, message: str, code: str, kind: str = "invalid_request_error") -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": kind, "code": code}},
    )


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _check_auth(authorization: str | None) -> None:
    key = get_settings().capstone_api_key
    if not key:
        return
    supplied = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
    if not hmac.compare_digest(supplied.encode(), key.encode()):
        raise _error(401, "Invalid API key.", "invalid_api_key", "authentication_error")


def _build_request(request: ChatCompletionRequest) -> QueryRequest:
    pattern = MODEL_ID_TO_PATTERN.get(request.model)
    if pattern is None:
        raise _error(400, f"Model '{request.model}' not found.", "model_not_found")
    latest_user = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if latest_user is None:
        raise _error(400, "At least one user message is required.", "missing_user_message")
    options = request.rag_options
    filters = None
    if options:
        nested = options.filters
        filters = SemanticFilters(
            act=options.act or (nested.act if nested else []),
            status=options.status or (nested.status if nested else []),
            access_level=options.access_level,
        )
    return QueryRequest(
        question=latest_user,
        pattern=pattern,
        caller_id=get_settings().webui_demo_caller_id,  # server-side demo caller
        filters=filters,
        limit=options.limit if options else 5,
        generate_answer=True,  # server-side decision; adapter never trusts client
        required_acts=options.required_acts if options else None,
        chapter=options.chapter if options else None,
    )


def _retrieve_for_chat(request: ChatCompletionRequest) -> tuple[QueryRequest, QueryResult]:
    query_request = _build_request(request)
    try:
        return query_request, retrieve(query_request)
    except QueryError as exc:
        raise _error(exc.status_code, exc.message, exc.code, "api_error") from None


def _sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    query_request, retrieval = _retrieve_for_chat(request)
    pieces = chat_pieces(query_request.question, retrieval)

    if not request.stream:
        text = "".join(pieces)
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": index,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _FINISH_STOP,
                }
                for index in range(request.n)
            ],
        }

    def chunk(base: dict, delta: dict, finish: str | None) -> str:
        return _sse_frame(
            {
                **base,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": index, "delta": delta, "finish_reason": finish}
                    for index in range(request.n)
                ],
            }
        )

    def generate():
        base = {"id": _completion_id(), "created": int(time.time()), "model": request.model}
        yield chunk(base, {"role": "assistant"}, None)
        for piece in pieces:
            if piece:
                yield chunk(base, {"content": piece}, None)
        yield chunk(base, {}, _FINISH_STOP)
        yield "data: " + _DONE + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
