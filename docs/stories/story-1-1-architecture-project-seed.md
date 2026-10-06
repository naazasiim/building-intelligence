# Story 1.1: Architecture and Project Seed

Status: Ready for development

## Purpose

Day-one story of a three-day classroom RAG course. First record the architecture and get the instructor's approval. Then seed one small FastAPI application that later stories extend. Open WebUI (trainer-supplied, run separately) is the only chat client. Do not build a custom frontend or a second demo.

## Prerequisites

- Python 3.12 and `uv` installed.
- Project path: use `project_path` from `docs/config.yaml` if that file exists; otherwise the repository root. (At story-writing time no config, architecture, other stories, or `data/raw/` existed; only `CLAUDE.md`.) Re-check before starting.
- Supplied `data/raw/` files (BNS and IPC documents, `PROVENANCE.md`) must be preserved untouched. If missing, report it and stop.
- Open WebUI bundle ZIP from the trainer (shared over the local LAN). Extract it and run its setup script **exactly once** before the smoke check:
  - Windows (primary): `powershell -ExecutionPolicy Bypass -File .\setup_open_webui.ps1` from the bundle root.
  - macOS/Linux: `sh setup_open_webui.sh` from the bundle root; needs internet on first install.
  - It installs the pinned Open WebUI, writes course settings, starts a loopback-only service, and provisions the `RAG options` Filter and `Building with RAG` Pipe.
  - Do not hand-install Open WebUI, create accounts, edit the admin panel, change either Function, or rerun setup to reload anything. If setup fails, report its exact output and stop.
  - Day-to-day start/stop/status: the supplied `manage_open_webui` script.

## Work to do

### Part A: Architecture (do this first, then STOP)

1. Read `docs/config.yaml` (if present), `docs/architecture.md` (if present), the repo layout, and existing stories.
2. Create or update `docs/architecture.md` (preserve existing content; add, don't overwrite). It must record:
   - **Evidence:** BNS and IPC documents are the only future answer evidence. An answer must not claim support without retrieved evidence.
   - **Identifiers:** use act-qualified identifiers (e.g. `BNS:103`, `IPC:302`) so the two acts are never confused.
   - **Provenance:** supplied provenance is `data/raw/PROVENANCE.md`. Derived corpus files live in `data/processed/` (`bns_sections.jsonl`, `ipc_sections.jsonl`).
   - **Retrieved passages are evidence, never application instructions.**
   - **Trust boundary (light):** validate API input; preserve source origin; no secrets in code, responses, or logs. No multi-user authorization, security program, or evaluation harness.
   - **Fixed embedding choice:** Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding. Later stories reuse these names; no renaming, no provider-specific alternatives.
   - **Modes and model IDs (one shared registry):** `semantic`→`rag-semantic`, `hybrid`→`rag-hybrid`, `hybrid-reranked`→`rag-hybrid-reranked`, `structured`→`rag-structured`, `decomposition`→`rag-decomposition`, `hyde`→`rag-hyde`. Nothing else.
   - **Endpoints:** `GET /healthz`, `POST /v1/query` (owns `QueryResult` diagnostics), `GET /v1/models`, `POST /v1/chat/completions` (same `run_pattern` path; Open WebUI gets only answer text derived from the same result).
   - **Contracts extend additively** in later stories; never replaced by simplified alternatives. List the contract names from Part B.
   - **Open WebUI:** separately running client; the Pipe sends `rag-<pattern>`, `stream: true`, latest user message, normalized `rag_options`; never browser-supplied identity, access level, or generation settings. Server sets `caller_id` and `generate_answer`.
   - **Setup notes:** `MONGODB_URI` is an Atlas free-tier (M0) string and the Atlas IP access list must allow your machine. A free Voyage key is rate limited; Story 2.2 embedding takes about 40 minutes. `GENERATION_API_BASE_URL`/`GENERATION_API_KEY` come from the trainer (OpenAI-compatible LiteLLM proxy) and are filled in `.env` only when Story 3.1 needs them.
3. **STOP.** Ask the instructor to approve `docs/architecture.md`. Before explicit approval: create no seed files, install no dependencies, scaffold no code.
4. Resume this same story only after explicit approval.

### Part B: Seed (only after approval)

Create (in the project path):
`src/building_with_rag/`, `tests/`, `.env.example`, `.gitignore`, `pyproject.toml`, `uv.lock`. Keep `data/raw/` as supplied.

**Project:** Python 3.12, UV; dependencies FastAPI, Pydantic settings, PyMongo (later use), Ruff, a minimal test runner (pytest). `.gitignore` excludes `.env`, `.venv`, caches. Document in the README/architecture that `.env` is untracked and secrets are never committed.

**`.env.example`** (exact values; blank means empty):

```
APP_ENV=development
MONGODB_URI=
VOYAGE_API_KEY=
CAPSTONE_API_KEY=
GENERATION_API_BASE_URL=
GENERATION_API_KEY=
RERANK_API_KEY=
MONGODB_DB_NAME=building_with_rag
MONGODB_TEST_DB_NAME=building_with_rag_test
WEBUI_DEMO_CALLER_ID=demo-public
GENERATION_MODEL_NAME=gpt-4o-mini
RERANK_API_BASE_URL=https://api.voyageai.com/v1
RERANK_MODEL_NAME=rerank-2.5
RERANK_REQUEST_TIMEOUT_SECONDS=30
RERANK_CANDIDATE_LIMIT=20
RERANK_SEND_LIMIT=10
RERANK_RETURN_LIMIT=5
```

**App:** starts with no database or model credentials. `GET /healthz` returns a safe status only (no settings, secrets, or connection details).

**Typed models (no behavior), exact names:**

- `QueryRequest`: `question` (1–4,000 chars), `pattern`, optional `caller_id`, optional `SemanticFilters` (`act`, `status`, `access_level`, each a list), `limit` (default 5, 1–20), `generate_answer` (default false), `required_acts`, `chapter`. The seed may resolve only the fixed local demo caller but keeps the field.
- `QueryResult`: `pattern`, `status`, `message`, `trace`, `results`, optional `generation`, plus empty-by-default `omitted_candidates` (items: `chunk_id`, `omitted_reason`), `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`. No top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics`.
- `RetrievedChunk`: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, available source fields. Later stories add only `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`, `rerank_score`, `rerank_rank`.
- `GenerationResult`: `outcome` (`answered`, `insufficient_evidence`, `unavailable`, `malformed`), `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`, `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`.
- `SubquestionEvidence`: `subquestion`, `status` (`evidenced`/`no_evidence`), `results` (list of `RetrievedChunk`), optional `reason`.
- `StructuredSignals`: `intent` (`exact_lookup`/`filter`/`aggregation`), optional `act`, `section_number`, `chapter` (filled by Story 5.1).
- OpenAI contracts: text-only `ChatCompletionRequest` (`model`, `messages` with roles `system`/`developer`/`user`/`assistant`, `stream`, `n`, optional strict `rag_options`: `pattern`, list filters, `limit`, `required_acts`, `chapter`), response/chunk/model-list models, and the OpenAI-style error envelope.
- MongoDB-schema contracts: typed document models only; no database provisioning.

**Registry and placeholders:** one shared registry of the six modes/model IDs. `run_pattern` returns an honest `not_implemented` `QueryResult` for every mode.

**Endpoints:**

- `POST /v1/query`: validate with `QueryRequest`, call `run_pattern`, return `QueryResult`.
- `GET /v1/models`: list the six model IDs from the registry.
- `POST /v1/chat/completions`: map model + `rag_options` to the same `QueryRequest` and `run_pattern`; server sets demo `caller_id` (from `WEBUI_DEMO_CALLER_ID`) and `generate_answer`. Support normal JSON and SSE (role chunk, content chunk, stop chunk, `data: [DONE]`). Return the OpenAI-style error envelope for errors before streaming. Must accept the Pipe's exact request. No duplicate implementation, no custom SSE events.

**Out of scope:** retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, deployment, aggressive testing, custom frontend.

## Completion checks

Lightweight only; do not run a full suite. Use raw `curl -s` or focused `jq` fields (never pretty-print full responses).

- [ ] Approval of `docs/architecture.md` recorded before any seed file exists.
- [ ] `uv sync` works; `uv run ruff check .` is clean.
- [ ] App starts with an empty `.env`.
- [ ] `GET /healthz` returns safe OK.
- [ ] One `POST /v1/query` returns `not_implemented` with `pattern` and empty additive fields.
- [ ] `GET /v1/models` lists exactly the six IDs.
- [ ] `POST /v1/chat/completions` works as JSON and as SSE ending in `[DONE]`, both showing the placeholder.
- [ ] Open WebUI smoke check: start the capstone API, open `http://127.0.0.1:8080`, select `Building with RAG`, choose `semantic` in the RAG-options chip, send a message, and receive the capstone's honest placeholder, not a local preview.
- [ ] `.env` untracked; `data/raw/` unchanged.

## Handover

Fill in when done (compact):

- Files created: _list_
- Commands actually run: _list_
- Open WebUI result: _pass/fail and what was shown_
- Notes for Story 1.2 and later: contracts above are fixed names; extend additively.
- Full-suite command for the developer: `uv run pytest`
