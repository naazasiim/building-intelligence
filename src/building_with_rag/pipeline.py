"""The one shared path: retrieve -> context -> generate/validate -> final outcome.

Both /v1/query and /v1/chat/completions call retrieve() + answer_events(); chat renders
the same events (plus a footer from the final result) as readable text.
"""

from collections.abc import Iterator

from building_with_rag.contracts import GenerationResult, QueryRequest, QueryResult
from building_with_rag.generation.answer import Event, stream_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.semantic import RetrievalError as QueryError
from building_with_rag.retrieval.semantic import run_semantic

__all__ = ["QueryError", "answer_events", "chat_pieces", "drain", "retrieve"]

REAL_PATTERNS = frozenset({Pattern.SEMANTIC.value, Pattern.HYBRID.value})
UNAVAILABLE_AFTER_TEXT = "Answer generation unavailable — the text above is an unchecked draft."
UNAVAILABLE = "Answer generation unavailable."
MAX_REASON_CHARS = 200


def retrieve(request: QueryRequest) -> QueryResult:
    if request.pattern is Pattern.SEMANTIC:
        return run_semantic(request)  # raises QueryError
    if request.pattern is Pattern.HYBRID:
        return run_hybrid(request)  # raises QueryError
    return QueryResult(**run_pattern(request.pattern, request.question, request.caller_id))


def answer_events(question: str, retrieval: QueryResult) -> Iterator[Event]:
    """Text/notice events then one final event (value None for placeholder modes)."""
    if retrieval.pattern not in REAL_PATTERNS:
        yield Event("text", retrieval.message)
        yield Event("final", None)
        return
    yield from stream_answer(question, retrieval)


def drain(events: Iterator[Event]) -> GenerationResult | None:
    final = None
    for event in events:
        if event.kind == "final":
            final = event.value
    return final


def _source_line(c) -> str:
    act = c.act.split("_")[0]
    section = f"{act} §{c.section_number}" if c.section_number is not None else act
    return f"{c.label} · {section} · {c.heading} · {c.section_id}"


def _footer(g: GenerationResult, emitted: bool) -> str:
    if g.outcome == "answered":
        sources = "\n".join(_source_line(c) for c in g.citations)
        return f"\n\nEvidence check passed — confidence: high\n\nSources:\n{sources}"
    if g.outcome == "insufficient_evidence":
        reason = str(g.trace.get("reason") or "").strip()[:MAX_REASON_CHARS]
        base = "The retrieved passages do not contain enough evidence to answer this question."
        return f"{base} {reason}".strip()
    if g.confidence == "low":
        last = max((i.attempt for i in g.issues), default=0)
        details = "\n".join(f"- {i.check}: {i.detail}" for i in g.issues if i.attempt == last)
        return (
            "\n\nDRAFT — low confidence, not the final answer.\n"
            f"{g.low_confidence_reason}\n{details}"
        )
    if g.outcome == "malformed" and g.draft_answer:
        return "\n\nDRAFT — the evidence check could not be completed; not the final answer."
    if g.outcome == "malformed":
        return "\n\nAnswer generation returned an unusable response."
    return ("\n\n" if emitted else "") + (UNAVAILABLE_AFTER_TEXT if emitted else UNAVAILABLE)


def chat_pieces(question: str, retrieval: QueryResult) -> Iterator[str]:
    """Readable text pieces for chat: streamed text/notices, then a footer from the final result."""
    emitted = False
    final = None
    for event in answer_events(question, retrieval):
        if event.kind == "final":
            final = event.value
        else:
            emitted = True
            yield event.value
    if final is not None:
        yield _footer(final, emitted)
