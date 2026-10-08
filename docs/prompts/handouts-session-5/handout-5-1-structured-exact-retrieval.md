# Handout 5.1 — Structured exact retrieval

Paste this prompt into your coding assistant from the capstone repository:

---

Create `docs/stories/story-5-1-structured-exact-retrieval.md`. Create the story only; do **not** implement it. Read the architecture, all previous stories, MongoDB schema, registry, `/v1/query`, and chat adapter first. Preserve all existing modes and shared contracts.

Write a focused implementation story with purpose, prerequisites, work to do, completion checks, and handover. It adds `structured` using the same `QueryRequest`/`QueryResult` contract and existing MongoDB settings in the project’s `.env.example`, with no renamed configuration. Use the `StructuredSignals` model defined in Story 1.1: write a small rule-based (no LLM) classifier that turns `QueryRequest.question` into `StructuredSignals` with only the fixed `exact_lookup`, `filter`, or `aggregation` intent, extracting `act` (BNS or IPC) and `section_number` from phrases like "BNS section 103"; use those signals plus optional validated `chapter`; and return a recommendation/clarification without a MongoDB call for any other question. It is a read-only MongoDB path, not an LLM extraction, semantic-search replacement, or automatic router.

For an exact lookup, use only the classifier’s validated act and section signals to query the compatible `sections` model by `act`, `section_number`, status, and public `access_level`; never turn raw request text into a MongoDB predicate. Return `QueryResult.results` as compatible `RetrievedChunk` records. When the classifier cannot establish a supported exact request, return its recommendation/clarification status and reason rather than guessing. When the exact record is missing, return `not_found`. Preserve source status/version as metadata without claiming current legal applicability.

Expose the mode through the shared registry, `/v1/query`, and `rag-structured` model selection. It must pass its result through the existing context/grounded-answer path only when an explanatory answer is requested; direct inspection remains visible in diagnostics. Reuse existing citations and Open WebUI streaming behavior. Do not add arbitrary database queries, write operations, automatic routing, permissions systems, or new UI screens.

Require a few direct checks: one valid BNS or IPC exact lookup, an ambiguous section-number request, and a missing record. Update the architecture with the exact-input contract and answer boundary. After creating the story, report its path only.
