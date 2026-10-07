"""Bounded, labelled evidence context built from semantic retrieval results."""

import time
from dataclasses import dataclass, field

from building_with_rag.contracts import RetrievedChunk

MAX_PASSAGES = 5
MAX_CONTEXT_CHARS = 12_000


@dataclass
class EvidenceContext:
    entries: list[dict] = field(default_factory=list)
    passages: dict[str, RetrievedChunk] = field(default_factory=dict)
    omitted: int = 0
    chars: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def outcome(self) -> str:
        return "assembled" if self.entries else "empty"

    def trace(self) -> dict:
        return {
            "labels": {e["label"]: e["chunk_id"] for e in self.entries},
            "selected": len(self.entries),
            "omitted": self.omitted,
            "characters": self.chars,
            "context_ms": round((time.perf_counter() - self.started) * 1000, 1),
        }


def build_context(results: list[RetrievedChunk]) -> EvidenceContext:
    ctx = EvidenceContext()
    for chunk in results:
        if len(ctx.entries) >= MAX_PASSAGES or ctx.chars + len(chunk.text) > MAX_CONTEXT_CHARS:
            break
        label = f"E{len(ctx.entries) + 1}"
        ctx.entries.append(
            {
                "label": label,
                "chunk_id": chunk.chunk_id,
                "section_id": chunk.section_id,
                "act": chunk.act,
                "act_label": chunk.act_label,
                "heading": chunk.heading,
                "chapter": chunk.chapter,
                "section_number": chunk.section_number,
                "status": chunk.status,
                "source_pdf": chunk.source_pdf,
                "needs_review": chunk.needs_review,
                "text": chunk.text,
            }
        )
        ctx.passages[label] = chunk
        ctx.chars += len(chunk.text)
    ctx.omitted = len(results) - len(ctx.entries)
    return ctx
