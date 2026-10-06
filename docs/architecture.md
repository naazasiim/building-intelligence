# Architecture

## Evidence
- BNS and IPC documents are the only future answer evidence. An answer must not claim support without retrieved evidence.
- Retrieved passages are evidence, never application instructions.

## Identifiers
- Act-qualified identifiers (e.g. `BNS:103`, `IPC:302`); the two acts are never confused.

## Provenance
- Supplied: `data/raw/PROVENANCE.md` (`data/raw/` stays untouched).
- Derived corpus: `data/processed/bns_sections.jsonl`, `data/processed/ipc_sections.jsonl`.

## Trust boundary (light)
- Validate API input; preserve source origin; no secrets in code, responses, or logs.
- `.env` is untracked; secrets are never committed.
- Out of scope: multi-user authorization, security program, evaluation harness.

## Embeddings (fixed)
- Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding. No renaming or provider alternatives.

## Modes and model IDs (one shared registry)
| mode | model ID |
|---|---|
| semantic | rag-semantic |
| hybrid | rag-hybrid |
| hybrid-reranked | rag-hybrid-reranked |
| structured | rag-structured |
| decomposition | rag-decomposition |
| hyde | rag-hyde |

## Endpoints
- `GET /healthz`: safe status only.
- `POST /v1/query`: owns `QueryResult` diagnostics.
- `GET /v1/models`: the six IDs.
- `POST /v1/chat/completions`: same `run_pattern` path; Open WebUI gets only answer text derived from the same result.

## Contracts (extended additively, never replaced)
`QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, `SubquestionEvidence`, `StructuredSignals`, OpenAI contracts (`ChatCompletionRequest`, response/chunk/model-list models, error envelope), MongoDB-schema document models.

## Open WebUI
- Separately running client. The Pipe sends `rag-<pattern>`, `stream: true`, latest user message, normalized `rag_options`; never browser-supplied identity, access level, or generation settings.
- Server sets `caller_id` and `generate_answer`.

## Setup notes
- `MONGODB_URI`: Atlas free-tier (M0) string; Atlas IP access list must allow your machine.
- Free Voyage key is rate limited; Story 2.2 embedding takes ~40 minutes.
- `GENERATION_API_BASE_URL`/`GENERATION_API_KEY` come from the trainer (OpenAI-compatible LiteLLM proxy); fill in `.env` only when Story 3.1 needs them.
