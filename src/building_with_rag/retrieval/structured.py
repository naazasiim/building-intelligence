"""Structured exact retrieval: rule-based classifier + one read-only `sections` lookup."""

import re

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from building_with_rag.contracts import (
    QueryRequest,
    QueryResult,
    RetrievedChunk,
    StructuredSignals,
)
from building_with_rag.ingestion.mongodb_schema import SECTIONS_COLLECTION
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalError, effective_filters
from building_with_rag.settings import get_settings

MIN_SECTION, MAX_SECTION = 1, 999
_CHAPTER = re.compile(r"[A-Za-z0-9 .\-]{1,40}")

_AGGREGATION = re.compile(r"\bhow\s+many\b|\bcount\b|\btotal\s+number\b", re.IGNORECASE)
_FILTER = re.compile(
    r"\blist\b|\ball\s+sections\b|\bwhich\s+sections\b|\bsections\s+(?:in|under)\b", re.IGNORECASE
)
_SECTION_REF = re.compile(r"(?:\bsections?\b|\bsec\b\.?|\bs\.|§)\s*(\d+\w*)", re.IGNORECASE)
_BNS = re.compile(r"\bBNS\b|\bBharatiya\s+Nyaya\s+Sanhita\b", re.IGNORECASE)
_IPC = re.compile(r"\bIPC\b|\bIndian\s+Penal\s+Code\b", re.IGNORECASE)

_OPEN_HINT = "Use `semantic` or `hybrid` mode for open questions."


def _sig(status, reason, intent=None, act=None, number=None, chapter=None) -> StructuredSignals:
    return StructuredSignals(
        intent=intent, act=act, section_number=number, chapter=chapter, status=status, reason=reason
    )


def classify(question: str, chapter: str | None = None) -> StructuredSignals:
    """Pure rule-based classification; acts and numbers are never guessed."""
    if _AGGREGATION.search(question):
        return _sig(
            "recommendation",
            "Counting questions are recognised but not executed. " + _OPEN_HINT,
            intent="aggregation",
            chapter=chapter,
        )
    if _FILTER.search(question):
        return _sig(
            "recommendation",
            "Listing/filter questions are recognised but not executed. " + _OPEN_HINT,
            intent="filter",
            chapter=chapter,
        )
    raw_numbers = list(dict.fromkeys(_SECTION_REF.findall(question)))
    if not raw_numbers:
        return _sig("recommendation", "No section reference found. " + _OPEN_HINT, chapter=chapter)

    def ask(reason: str) -> StructuredSignals:
        return _sig("clarification_needed", reason, intent="exact_lookup", chapter=chapter)

    acts = []
    if _BNS.search(question):
        acts.append("BNS_2023")
    if _IPC.search(question):
        acts.append("IPC_1860")
    if len(raw_numbers) > 1:
        return ask("Several section numbers found; ask for one section of one act at a time.")
    raw = raw_numbers[0]
    if not raw.isdigit() or not MIN_SECTION <= int(raw) <= MAX_SECTION:
        return ask(
            f"Section '{raw}' is not a plain integer between {MIN_SECTION} and {MAX_SECTION}."
        )
    if not acts:
        return ask("The act is missing; say BNS or IPC.")
    if len(acts) > 1:
        return ask("Both BNS and IPC were named; ask about one act at a time.")
    return _sig(
        "ok",
        "Exact section reference found.",
        intent="exact_lookup",
        act=acts[0],
        number=int(raw),
        chapter=chapter,
    )


def _check_scope(request: QueryRequest) -> None:
    problems = []
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        problems.append("caller_id is not accepted for this server's caller")
    if request.required_acts:
        problems.append("required_acts is not supported by this mode")
    if problems:
        raise RetrievalError(422, "invalid_request", "; ".join(problems) + ".")
    if request.chapter is not None and not _CHAPTER.fullmatch(request.chapter):
        raise RetrievalError(
            422,
            "unsupported_option",
            "chapter must be 1-40 characters of letters, digits, spaces, '.' and '-'.",
        )


def _sections():
    settings = get_settings()
    if not settings.mongodb_uri:
        raise RetrievalError(503, "retrieval_not_ready", "MONGODB_URI is not set.")
    with semantic._lock:
        if semantic._mongo is None:
            semantic._mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=semantic.MONGO_TIMEOUT_MS,
                connectTimeoutMS=semantic.MONGO_TIMEOUT_MS,
                socketTimeoutMS=semantic.MONGO_TIMEOUT_MS * 3,
            )
        return semantic._mongo[settings.mongodb_db_name][SECTIONS_COLLECTION]


def _to_chunk(doc: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=doc["section_id"],
        section_id=doc["section_id"],
        act=doc.get("act") or "",
        text=doc.get("text") or "",
        heading=doc.get("heading") or "",
        score=1.0,  # exact match, not a similarity
        act_label=doc.get("act_label"),
        status=doc.get("status"),
        chapter=doc.get("chapter"),
        chapter_title=doc.get("chapter_title"),
        section_number=doc.get("section_number"),
        source_pdf=doc.get("source_pdf"),
        source_sha256=doc.get("source_sha256"),
        needs_review=doc.get("needs_review"),
    )


def run_structured(request: QueryRequest) -> QueryResult:
    _check_scope(request)
    signals = classify(request.question, request.chapter)
    filters = effective_filters(request)
    trace = {
        "mode": "structured",
        "signals": signals.model_dump(),
        "mongodb_called": False,
        "collection": SECTIONS_COLLECTION,
        "filters": filters,
        "caller_id": get_settings().webui_demo_caller_id,
        "result_count": 0,
    }

    if signals.status == "ok" and "act" in filters and signals.act not in filters["act"]:
        signals = _sig(
            "clarification_needed",
            f"The requested act filter excludes {signals.act}.",
            intent="exact_lookup",
            chapter=signals.chapter,
        )
        trace["signals"] = signals.model_dump()
    if signals.status != "ok":
        return QueryResult(
            pattern="structured", status=signals.status, message=signals.reason, trace=trace
        )

    predicate: dict = {
        "act": signals.act,
        "section_number": signals.section_number,
        "access_level": {"$in": filters["access_level"]},
    }
    if "status" in filters:
        predicate["status"] = {"$in": filters["status"]}
    if signals.chapter:
        predicate["chapter"] = signals.chapter

    collection = _sections()
    trace["mongodb_called"] = True
    try:
        doc = collection.find_one(predicate, {"provenance": 0})
    except PyMongoError as exc:
        raise RetrievalError(
            502, "retrieval_upstream_error", f"Retrieval backend error ({type(exc).__name__}); try again."
        ) from None

    if doc is None:
        return QueryResult(
            pattern="structured",
            status="not_found",
            message="This corpus has no matching record for that request.",
            trace=trace,
        )
    trace["result_count"] = 1
    trace["record"] = {
        "section_id": doc.get("section_id"),
        "status": doc.get("status"),
        "source_status_version": doc.get("source_status_version"),
    }
    return QueryResult(
        pattern="structured",
        status="ok",
        message=(
            "Exact record from the supplied corpus. Its status and source_status_version describe "
            "the source document and do not claim current legal applicability."
        ),
        trace=trace,
        results=[_to_chunk(doc)],
    )
