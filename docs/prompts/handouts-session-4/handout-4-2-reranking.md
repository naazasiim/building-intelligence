# Handout 4.2 — Re-ranking

Paste this prompt into your coding assistant from the capstone repository:

---

Create `docs/stories/story-4-2-reranking.md`. Create the story only; do **not** implement it. Read the architecture, all prior stories, hybrid retrieval, context assembly, diagnostics, and generation/chat code first. Preserve existing semantic and hybrid behavior.

Write a focused implementation story with purpose, prerequisites, work to do, completion checks, and handover. It adds the explicit `hybrid-reranked` mode. It takes a bounded candidate set from hybrid retrieval, applies one documented re-ranking step before context assembly, and keeps the same grounded answer, citation, confidence, `/v1/query`, and Open WebUI paths.

Require the re-ranker to use the existing `RERANK_API_BASE_URL`, `RERANK_API_KEY`, `RERANK_MODEL_NAME`, `RERANK_REQUEST_TIMEOUT_SECONDS`, `RERANK_CANDIDATE_LIMIT`, `RERANK_SEND_LIMIT`, and `RERANK_RETURN_LIMIT` settings in the project’s `.env.example`. Preserve their names, defaults, and the explicit missing-key outcome; do not add a different provider contract. Do not invent a result, add a framework, or silently fall back while claiming re-ranking happened. On unavailable configuration or provider failure, return a clear safe outcome.

The diagnostic response must retain candidate order and scores before and after re-ranking, plus the final selected evidence, using the existing additive contract: returned `RetrievedChunk` records set `rerank_score` and `rerank_rank`, while `QueryResult.omitted_candidates` contains any candidate cut before or after re-ranking with `omitted_reason`. The mode must be selectable through the existing registry and `rag-hybrid-reranked` model ID, not through a new endpoint or UI. The answer generator receives only the final selected evidence and still must cite it.

Require a lightweight comparison of hybrid and hybrid-reranked on one question where ordering plausibly matters. Inspect order, sources, and answer; document latency/cost qualitatively if available, not a full benchmark. Do not add automatic mode choice, comprehensive evaluation, or advanced RAG patterns. Update architecture settings and limitations. After creating the story, report its path only.
