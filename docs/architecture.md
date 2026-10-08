# Capstone Architecture — Building Intelligence with RAG

Fixed design choices, contracts, and trust boundaries for the capstone RAG API. Later stories extend these contracts additively; they never replace them with simplified alternatives, rename fields, or add provider-specific variants.

## Scope

One small application: the capstone RAG API. No chat frontend (Open WebUI is a separate trainer-supplied client) and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Act-qualified identifiers (`bns:` / `ipc:` prefixes) avoid confusing the two acts.
- Supplied provenance lives at `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never application instructions.

## Trust boundaries

Kept light for this course:

- Validate API input.
- Preserve source origin on retrieved passages.
- Do not put secrets in code, responses, or logs.
- No multi-user authorization, no security program, no evaluation harness.

## Fixed embedding choices

- Model: `voyage-3.5`, model version `voyage-3.5`, 1,024 dimensions.
- Used for every document and query embedding.
- Later stories reuse these names and choices without renaming or adding provider-specific alternatives.

## Course modes and shared registry

One shared registry (single source of truth) holds only:

| Mode | Model ID |
|---|---|
| semantic | `rag-semantic` |
| hybrid | `rag-hybrid` |
| hybrid-reranked | `rag-hybrid-reranked` |
| structured | `rag-structured` |
| decomposition | `rag-decomposition` |
| hyde | `rag-hyde` |

`semantic` (Story 2.3), `hybrid` (Story 4.1), `hybrid-reranked` (Story 4.2) and `structured` (Story 5.1) are real on `POST /v1/query`; `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked` and `rag-structured` on chat run the same real path (Stories 3.2, 4.1, 4.2, 5.1). Every other mode, on query and chat, still returns an honest `not_implemented` placeholder until its own story.

## API contracts

### `QueryRequest` (`POST /v1/query`)

- `question`: string, 1–4,000 characters, required.
- `pattern`: one of the six modes.
- `caller_id`: optional.
- `filters`: optional `SemanticFilters` — `act`, `status`, `access_level`, each a list.
- `limit`: default 5, range 1–20.
- `generate_answer`: default false.
- `required_acts`: optional.
- `chapter`: optional.

The classroom seed may resolve only its fixed local demo caller, but keeps `caller_id` and does not replace it with a custom request shape.

### `QueryResult`

- `pattern`, `status`, `message`, `trace`, `results`. `structured` adds `status` values `not_found`, `clarification_needed`, `recommendation`.
- Optional `generation`.
- Additive empty-by-default fields later modes use: `omitted_candidates` (used by `hybrid-reranked` for cut candidates), `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- No parallel top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` fields.

### `RetrievedChunk`

A retrieved passage is always this shape: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and optional source fields (`chunk_index`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `source_pdf`, `source_sha256`, `needs_review`; null when unavailable). Hybrid adds optional `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank` (null in semantic mode and for a route that did not return the chunk). Story 4.2 adds optional `rerank_score`, `rerank_rank`, `omitted_reason` (null outside `hybrid-reranked`; rerank fields also null for candidates not sent to the re-ranker).

### Endpoints

- `GET /healthz` — safe, no credentials required.
- `POST /v1/query` — accepts `QueryRequest`, returns `QueryResult`: real retrieval for `semantic`, `hybrid`, `hybrid-reranked` and `structured`, a `not_implemented` placeholder per other mode via the shared `run_pattern` path.
- `GET /v1/models` — lists the six `rag-<pattern>` model IDs.
- `POST /v1/chat/completions` — OpenAI-compatible, text-only `ChatCompletionRequest`: `model`, `messages` with `system`/`developer`/`user`/`assistant` roles, `stream`, `n`, and optional strict `rag_options` (`pattern`, flat or nested `filters.{act,status}`, `limit`, `required_acts`, `chapter`).

The chat adapter maps the selected model to the same `QueryRequest` and `pipeline.retrieve` + `answer_events` path as `/v1/query`; it sets server-side demo `caller_id` and `generate_answer`. Supports normal OpenAI Chat Completions JSON responses and role/content/stop frames, plus the OpenAI-style error envelope before streaming begins. Open WebUI shows readable text only: streamed answer text, DRAFT/retry lines, and a confidence/sources footer. No duplicated implementations, no custom SSE events that Open WebUI cannot render.

## Open WebUI (trainer-supplied, separate client)

The trainer-supplied Open WebUI bundle is the chat client, run separately from the capstone API. Its pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The capstone's adapter accepts that exact request and uses server-side `caller_id`/`generate_answer`.

`/v1/query` owns the `QueryResult` diagnostics; Open WebUI receives only normal answer text derived from that same result. Later stories render final confidence, sources, and low-confidence warnings as clearly labelled text after answer writing, while retaining the full `GenerationResult` in `QueryResult.generation`.

## Environment

- `.env` is untracked; secrets are never committed.
- The application must start with no database or model credentials and expose a safe `GET /healthz`.
- Canonical environment values are defined in `.env.example`; `RERANK_API_KEY` (empty by default) is required only by `hybrid-reranked`.

## Semantic retrieval (Story 2.3)

Module: `src/building_with_rag/retrieval/semantic.py`, called from `routes/query.py` for `pattern: "semantic"`.

Flow: validate → scope filters → embed query → `$vectorSearch` → resolve chunk/section → `QueryResult`.

- Validate: `SemanticFilters` forbids unknown fields and accepts only known plain-string values. `caller_id` must be omitted or equal `WEBUI_DEMO_CALLER_ID`; `required_acts` and `chapter` are rejected (HTTP 422). `generate_answer: true` triggers grounded generation (see Story 3.1 below); omitted/false leaves `generation` null.
- Scope: server fixes `access_level` to `["public"]`; caller `act`/`status` lists only narrow it (`$in` inside the `$vectorSearch` `filter`).
- Embed: raw question, `voyage-3.5`, `input_type="query"`, 1,024 dimensions checked. Clients are created lazily with bounded timeouts; `/healthz` needs no credentials.
- Search: `vector_index`, `limit` from the request, `numCandidates = clamp(limit * 10, 50, 200)`. Chunks are returned in score order (not de-duplicated by section).
- Resolve: `chunk_id` → `chunks` (text) and `sections` (heading, chapter, source fields). Unresolvable hits are omitted and counted in `trace.unresolved_hits`.

Outcomes: `ok` (passages returned); `no_results` (HTTP 200, filters match nothing). No score cutoff; scores rank similarity only. HTTP 503 `retrieval_not_ready` (missing credentials, index absent/not queryable, empty embeddings, model/dimension mismatch); HTTP 502 `retrieval_upstream_error` (Voyage/MongoDB error during a query).

Diagnostic command (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

## Hybrid retrieval (Story 4.1)

Module: `retrieval/hybrid.py`, `pattern: "hybrid"` (`rag-hybrid`); explicit selection only, no routing or fallback. Same corpus, scope rules (`validate_request`, `effective_filters`) and shared `retrieve` + `answer_events` path as semantic.

- Keyword route: Atlas Search `$search` with the `text` operator (BM25 `searchScore`) on `chunks.text`; filters `act`/`status`/`access_level` via `in` inside `compound.filter`. The question is only the `text.query` value.
- Index `chunk_text_index` on `chunks` (`search` type): `text` string/`lucene.standard`; `act`, `status`, `access_level` token; `dynamic: false`. Create/reuse: `uv run python -m building_with_rag.ingestion.keyword_index` (never replaces a differing index; `ingest` ignores it).
- Fusion: Reciprocal Rank Fusion on ranks, `fused_score = sum 1/(RRF_K + rank)`, `RRF_K = 60`, equal weights. Each route returns `ROUTE_DEPTH = max(limit, min(50, max(20, 4*limit)))`. Order: `fused_score` desc, then `semantic_rank` (missing last), then `chunk_id`; top `limit`; `fused_rank` = position. Fusion is by `chunk_id`.
- Result: `score == fused_score`; per-route score/rank fields show which route found each passage.
- Trace: `mode`, `query`, `embedding`, `filters`, `caller_id`, `result_count`, `unresolved_hits`, `semantic`, `keyword`, `fusion`, `contribution` (`both`/`semantic_only`/`keyword_only`).
- Outcomes: `ok`, `no_results` (both routes empty), 503 `retrieval_not_ready` (missing/not-ready keyword or vector index; never degrades to semantic-only), 502 `retrieval_upstream_error`.
- Limitation: rank-only fusion ignores score magnitude; `text` matches any query term (OR), so long questions pull in common words; section numbers match only inside chunk `text`; no stemming/synonyms beyond the standard analyzer. Atlas required; no `$text`, regex, or client-side keyword scoring.

Diagnostic (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question":"<q>","pattern":"hybrid","limit":5}'   | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'
```

## Re-ranking (Story 4.2)

Module: `retrieval/rerank.py`, `pattern: "hybrid-reranked"` (`rag-hybrid-reranked`); explicit selection only, no routing or fallback. Flow: `run_hybrid` (limit `RERANK_CANDIDATE_LIMIT`) → first `RERANK_SEND_LIMIT` candidates → one provider call → final evidence. Final evidence uses the shared `retrieve` + `answer_events` path.

- Settings: `RERANK_API_BASE_URL` (`https://api.voyageai.com/v1`), `RERANK_API_KEY` (empty by default, required), `RERANK_MODEL_NAME` (`rerank-2.5`), `RERANK_REQUEST_TIMEOUT_SECONDS` (30), `RERANK_CANDIDATE_LIMIT` (20), `RERANK_SEND_LIMIT` (10), `RERANK_RETURN_LIMIT` (5). Valid when `1 <= RETURN <= SEND <= CANDIDATE <= 20` and timeout >= 1; otherwise 503 `retrieval_not_ready` naming the setting. Checked before any Voyage/MongoDB call.
- Request: one `POST {RERANK_API_BASE_URL}/rerank` (`httpx`, Bearer key, `{model, query, documents}`, no `top_k`, no retries). Document = `"{heading}\n{chunk text}"`.
- Reply validation: `data` non-empty list; each `index` an int in the sent range, unique; each `relevance_score` finite. Otherwise 502. Scores are never invented or defaulted.
- Selection (`select_evidence`, pure): sent = first `SEND` by `fused_rank`; the rest are omitted `not_sent_to_reranker`. Sent are ordered by `relevance_score` desc, ties by `fused_rank`; `rerank_rank` is 1-based. Final = first `min(request.limit, RETURN)`; the rest are omitted `below_return_limit` (rerank fields kept).
- Result: `results` ordered by `rerank_rank`, `score == rerank_score`, hybrid fields kept as the "before" evidence. `omitted_candidates` = all cut candidates in `fused_rank` order. Together they show every candidate's before (`fused_rank`/`fused_score`) and after (`rerank_rank`/`rerank_score`).
- Trace: `mode`, `query`, `filters`, `caller_id`, `result_count`, `hybrid` (embedding, semantic, keyword, fusion, contribution, unresolved_hits), `rerank` (model, limits, counts `candidates`/`sent`/`returned`/`omitted_before`/`omitted_after`, `latency_ms`, `usage_tokens` when reported). No keys, URLs, or vectors.
- Outcomes: `ok`; `no_results` (no provider call); 503 `retrieval_not_ready` (missing/invalid configuration incl. empty `RERANK_API_KEY`); 502 `retrieval_upstream_error` (timeout, connection error, non-2xx, invalid reply); hybrid errors pass through. On failure no `results`; never hybrid order labelled as re-ranked.
- Limitations: candidates outside the top `RERANK_CANDIDATE_LIMIT` are never seen; the re-ranker scores each passage independently against the question; scores are model-specific, uncalibrated, not comparable with fused scores or across questions, and there is no score cutoff; one extra provider call per request (latency, cost, rate limits); no retry, no fallback; the answer context cap (5 passages / 12,000 chars) still applies.

Diagnostic (text truncated; merges returned and omitted records):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"<q>","pattern":"hybrid-reranked","limit":5}' \
  | jq '{status, t: .trace.rerank, r: [(.results[]|.+{kept:true}), (.omitted_candidates[]|.+{kept:false})] | map({section_id, kept, fr: .fused_rank, fs: .fused_score, rr: .rerank_rank, rs: .rerank_score, why: .omitted_reason, text: .text[:50]}) | sort_by(.fr)}'
```

## Structured exact retrieval (Story 5.1)

Module `retrieval/structured.py`; `StructuredSignals` lives in `contracts.py`. `pattern: "structured"` runs: scope check → rule-based `classify` → one read-only `find_one` on `sections`. No LLM, embedding or index; no fallback to or from other modes.

- **Exact-input contract:** only classifier output (validated `act`, integer `section_number` 1–999, optional validated `chapter`) plus server filters (`access_level`, optional `act`/`status` narrowing) reach MongoDB. Raw question text never does. Classified: `BNS`/`IPC` (or full names) + `section N`/`sec. N`/`s. N`/`§N`. Refused with `clarification_needed`: missing act, both acts, several numbers, non-integer or out-of-range numbers. `filter` and `aggregation` questions and anything else return `recommendation`, no MongoDB call. Invalid `chapter` → 422 `unsupported_option`.
- **Outcomes:** `ok` (one `RetrievedChunk`, `chunk_id = section_id`, `score = 1.0` = exact match, not similarity), `not_found` (this corpus has no such record, not a claim about the law), `clarification_needed`, `recommendation`; 503 `retrieval_not_ready` (no `MONGODB_URI`, only when a lookup is needed), 502 `retrieval_upstream_error`.
- **Answer boundary:** retrieval returns the exact record. Explanation only via the existing grounded-answer path (`generate_answer` or chat), and only for `ok`. `not_found`/`clarification_needed`/`recommendation` never produce an answer; chat shows `message`. `status` and `source_status_version` are source metadata, not current legal applicability.
- **Limitations:** integer sections only (no `103A`); no multi-section or cross-act comparison; filter/aggregation recognised but not executed; IPC sections 4, 5, 18, 34, 40, 75, 161–165 absent from `sections`; rule-based phrasing misses; a section over the 12,000-char context cap gives `insufficient_evidence` when answering (direct inspection still returns it).

## Context and answer boundaries (Story 3.1)

Modules: `generation/context.py`, `generation/answer.py`; called from `routes/query.py` when `pattern: "semantic"` and `generate_answer: true`.

Flow: real-mode (semantic, hybrid, hybrid-reranked, structured with `status: ok`) retrieval result → bounded labelled context (`E1`…, max 5 passages / 12,000 chars, never cut mid-passage) → one `POST {GENERATION_API_BASE_URL}/chat/completions` (30 s, no retries) → strict JSON parse → citations resolved from the supplied context only.

- Outcomes (`generation.outcome`): `answered` (text, claims, citations, supporting_passages), `insufficient_evidence` (also when retrieval is empty; no model call), `unavailable` (missing settings, timeout, connection error, non-2xx), `malformed` (non-JSON, missing `choices`, unknown label, claim without evidence, answered without text/claims). Non-answered outcomes have empty text and no claims/citations/passages.
- HTTP 200 for all four; `status` stays the retrieval status and `results` are always returned. Messages and traces never contain the proxy URL, key, prompt text, or vectors; the model `reason` is kept in `generation.trace` only.
- `GenerationResult` gains optional fields: `outcome`, `claims`, `citations`, `supporting_passages`, `provider`, `trace`, `context_outcome` (`text`/`model` unchanged).
- Evidence is untrusted source text, never instructions. No claims of current legal applicability beyond the supplied BNS/IPC documents.
- Story 3.2 replaced the strict JSON output with streamed plain text (below); outcomes and citation resolution are unchanged.

## Corpus

Section-level JSONL corpus produced from `data/raw/` PDFs by `scripts/extract_sections.py`. One JSON object per line, one record per section. No MongoDB, no embeddings, no vector indexes.

### Parser

- **Library**: `pymupdf` (fitz) 1.28.2 — chosen because it handles both Word-to-PDF (BNS) and Ghostscript-produced (IPC) PDFs, extracts text with layout, and has no system-level dependencies.
- **Extraction command**: `uv run python scripts/extract_sections.py`

### Output format

JSONL files at `data/processed/`:

| File | Records | Sections |
|---|---|---|
| `data/processed/bns_sections.jsonl` | 358 | 1–358 |
| `data/processed/ipc_sections.jsonl` | 500 | 1–511 (11 unextractable) |

Each record has 14 fields: `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`.

### Known limitations

- **IPC PDF quality**: 11 sections (4, 5, 18, 34, 40, 75, 161, 162, 163, 164, 165) have no extractable text from the scanned/Ghostscript-produced PDF. Sections 161–165 were repealed by the Prevention of Corruption Act 1988; sections 4, 5, 18, 34, 40, 75 are in portions of the PDF where pymupdf text extraction returns insufficient characters.
- **IPC section headings**: Some IPC sections (e.g. 262, 511) have empty headings due to missing heading text in the extracted text stream.
- **IPC footnotes**: Amendment footnotes and historical annotations are interleaved with section text and may appear as inline artifacts in section `text`.
- **BNS chapter markers**: Chapter boundaries are detected from `CHAPTER <roman>` lines in the body text. The BNS index (pages 2–19) provides section headings; the correspondence table (pages 20–73) is skipped.
- **Source-hash safety rule**: If a source PDF hash changes, the corpus for that act is regenerated as an atomic replacement. Records from different PDF versions are never mixed in one corpus file.

## Streamed answers and confidence (Story 3.2)

- Shared path (`pipeline.py`): `retrieve(request)` → `answer_events(question, retrieval)`. `/v1/query` drains the events into `QueryResult.generation`; chat forwards `text`/`notice` events as SSE and renders the footer from the same final `GenerationResult`. One request = one generation/validation operation. `QueryError` is raised before any streaming starts.
- Event flow: role chunk → one content chunk per piece → empty delta with `finish_reason: "stop"` → `[DONE]`. `n>1` repeats the same piece in every choice. `stream: false` returns the same text in one JSON response.
- Model output is plain text with inline labels (`[E1]`); `INSUFFICIENT_EVIDENCE: <reason>` is buffered and never streamed as an answer.
- `MAX_ATTEMPTS = 2`. Per attempt, in order: `citation_labels` (every cited label was supplied; never rewritten), `claim_cited` (every sentence/bullet has a label, text non-empty), `support` (one non-streamed validator call, strict JSON, run only when the first two pass). A retry runs only after a failed attempt and receives the plain-language issues.
- New optional `GenerationResult` fields: `confidence` (`high` = final attempt passed; `low` = final attempt failed; null = nothing judged), `issues` (all failed checks across attempts), `attempts` (`attempt`, `status` passed|failed|unjudged, `chars`, `latency_ms`), `draft_answer` (last text that did not pass), `low_confidence_reason`.
- Outcome mapping: passed → `answered` + high; failed final → `malformed`, empty `text`, `draft_answer`, low; provider failure → `unavailable`; validator unreachable → `unavailable` + draft; validator invalid reply → `malformed` + draft, no confidence; `insufficient_evidence` unchanged.
- Chat labels: each attempt starts `DRAFT — checking evidence`; `Check failed: … Retrying (attempt 2 of 2)…`; footer `Evidence check passed — confidence: high` + `Sources:`; failed final `DRAFT — low confidence, not the final answer.`; insufficient evidence is one plain sentence.
- Failure after text began: one final line `Answer generation unavailable — the text above is an unchecked draft.`, then stop and `[DONE]` (HTTP 200). Failure before any text: `Answer generation unavailable.`
- `CAPSTONE_API_KEY`: when non-empty, `/v1/chat/completions` requires `Authorization: Bearer <key>` (401 `invalid_api_key` otherwise). Empty = no check. `/v1/query`, `/v1/models`, `/healthz` are unchanged.
