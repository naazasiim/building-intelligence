# Manual tests

## Story 1.1 — Architecture and Project Seed

What it adds: FastAPI project seed with health, query, model-listing, and OpenAI-compatible chat endpoints — all returning honest `not_implemented` placeholders.

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Health

```bash
curl -s http://127.0.0.1:8000/healthz
```

Expected: `{"status":"ok"}`.

### List models

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

Expected: six lines: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.

### Query — each RAG mode (semantic)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic"}'
```

Expected (Story 2.3, needs `.env` credentials and Story 2.2 data): `"status":"ok"` with ranked passages in `results` and a populated `trace`. Without credentials: HTTP 503 `retrieval_not_ready`.

### Query — hybrid

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid", "limit": 3}'
```

Expected (Story 4.1, needs `chunk_text_index`): `"status":"ok"` with fused passages; without the index, HTTP 503 `retrieval_not_ready`.

### Query — hybrid-reranked

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid-reranked`.

### Query — structured

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "structured"}'
```

Expected: `"status":"not_implemented"`, message references `structured`.

### Query — decomposition

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "decomposition"}'
```

Expected: `"status":"not_implemented"`, message references `decomposition`.

### Query — hyde

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hyde"}'
```

Expected: `"status":"not_implemented"`, message references `hyde`.

### Query — empty question (failure)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "semantic"}'
```

Expected: 422 validation error (question below min_length 1).

### Chat completions — JSON (non-streaming)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "What is theft?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content contains `not implemented yet`.

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "What is theft?"}], "stream": true}'
```

Expected: SSE `data:` frames with `delta` role then content, ending with `data: [DONE]`.

### Chat completions — invalid model (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: 400 with `"type":"invalid_request_error"`, `"code":"model_not_found"`.

### Chat completions — no user message (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "system", "content": "You are helpful."}]}'
```

Expected: 400 with `"code":"missing_user_message"`.

## Story 2.1 — Document Ingestion

What it adds: PDF extraction script that produces inspectable JSONL section-level corpora from the BNS and IPC bare act PDFs.

Prerequisite: Story 1.1 complete, `data/raw/` PDFs and `PROVENANCE.md` present, `uv sync` done.

### Run extraction

```bash
uv run python scripts/extract_sections.py
```

Expected: prints parser name/version, record counts (~358 BNS, ~500 IPC), count of `needs_review` flags, count of empty-text records. Both `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl` exist.

### Re-run (skip)

```bash
uv run python scripts/extract_sections.py
```

Expected: prints "BNS corpus up to date — skipping" and "IPC corpus up to date — skipping". No records appended or overwritten.
## Story 2.2 — MongoDB, Chunks, Embeddings, and Vector Index

What it adds: an ingestion runner that loads the JSONL corpora into MongoDB as `sources`, `sections`, `chunks`, and `embeddings`, embeds every chunk with Voyage, and creates the Atlas `vector_index` on `embeddings.vector`.

Prerequisite: Story 2.1 complete (both JSONL files present), and `.env` holds `MONGODB_URI` (Atlas cluster), `MONGODB_DB_NAME`, `VOYAGE_API_KEY`. First run takes roughly 35–40 minutes on a free Voyage key.

### Run ingestion

```bash
uv run python -m building_with_rag.ingestion.ingest
```

Expected: steps 1–10 print in order. `sources=2`, `sections=858` with the supplied PDFs, `chunks` > 0, and `embeddings == chunks: True`. `bns:1 chunks` shows one or more with `all linked: True`, sample vector length 1024, index status `READY`, and the sample query for "punishment for theft" prints `chunk_id`, `section_id`, heading, and score (or `vector query pending — index not ready`).

### Re-run (skip)

```bash
uv run python -m building_with_rag.ingestion.ingest
```

Expected: sources and sections report skipped, chunks report all skipped with 0 inserted/replaced, `Embeddings: 0 inserted, <n> skipped, 0 deleted` (no Voyage calls), and the existing `vector_index` is reused rather than recreated.

### Missing MongoDB URI (failure)

```bash
MONGODB_URI= uv run python -m building_with_rag.ingestion.ingest
```

Expected: stops at step 1 with `MongoDB unavailable: MONGODB_URI is empty. Set it in .env and re-run.` Nothing is written and no Voyage call is made.
## Story 2.3 — Semantic Retrieval

What it adds: `POST /v1/query` with `pattern: "semantic"` embeds the question, runs a MongoDB vector search, and returns ranked source passages.

Prerequisite: Story 2.2 data ingested, `MONGODB_URI` and `VOYAGE_API_KEY` set in `.env`, API running on port 8000.

```bash
# Success
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'

# No results (IPC is repealed, so these filters match nothing)
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "theft", "pattern": "semantic", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'

# Failure (unsupported filter value)
curl -s -w ' HTTP %{http_code}\n' http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "theft", "pattern": "semantic", "filters": {"access_level": ["restricted"]}}'
```

Expected: first returns `"pattern":"semantic"`, `"status":"ok"`, at most 3 `results` in non-increasing `score` order, and a populated `trace`. Second returns HTTP 200, `"status":"no_results"`, empty `results`. Third returns HTTP 422 validation error. Without credentials the semantic query returns HTTP 503 `retrieval_not_ready`. Output includes full passage text; truncate `text` when inspecting.


## Story 3.1 — Grounded Answer Generation

What it adds: `POST /v1/query` with `pattern: "semantic"` and `generate_answer: true` returns a grounded answer with citations, or an honest `insufficient_evidence`/`unavailable`/`malformed` outcome in `generation`.

Prerequisite: Story 2.3 prerequisites plus `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` set in `.env`, API running on port 8000. The `jq` filters below truncate passage text.

```bash
# Answerable
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}' \
  | jq '{status, g: (.generation | {outcome, model, text: (.text[:300]), claims: [.claims[] | {t: .text[:80], e: .evidence_labels}], citations: [.citations[] | {label, chunk_id, section_id, act}]})}'

# Unsupported question
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the GST rate on restaurant services?", "pattern": "semantic", "limit": 5, "generate_answer": true}' \
  | jq '{status, g: (.generation | {outcome, text, citations}), results: (.results | length)}'

# Generation not requested
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "theft", "pattern": "semantic", "limit": 3}' | jq '{status, generation}'
```

Expected: first returns `"status":"ok"`, `generation.outcome` `"answered"`, non-empty `text`, claims with evidence labels, and citations whose `section_id` prefix matches `act` (`bns:` ↔ `BNS_2023`). Second returns `"insufficient_evidence"`, empty `text`, no citations, while `results` are still returned. Third returns `"generation": null`. With `GENERATION_API_KEY` empty (restart the API), the first request returns HTTP 200 with `outcome` `"unavailable"` and empty `text`; `results` stay present. Other modes on `/v1/query` still return `not_implemented`; chat `rag-semantic` is real since Story 3.2.

## Story 3.2 — Streamed Answers with Confidence

Run in Open WebUI (`Building with RAG` model, Pipe re-pasted) or the streamed curl form from the story (bounded with `head -c 1500`).

1. "What is the punishment for theft under the BNS?" → `DRAFT — checking evidence`, labelled text, `Evidence check passed — confidence: high`, `Sources:` lines. `/v1/query` (`generate_answer: true`) agrees on outcome and citations.
2. "What is the GST rate on restaurant services?" → one insufficient-evidence sentence, no confidence; `/v1/query` shows `insufficient_evidence` with `results` intact.
3. Stream has role chunk, content chunks, `finish_reason: "stop"`, `[DONE]` last. With `CAPSTONE_API_KEY` set, a wrong Bearer on chat → 401 `invalid_api_key`; unset → works.
4. `rag-hybrid-reranked` on chat and `/v1/query` → `not_implemented`; `generate_answer` omitted → `generation: null`.

Offline: `uv run pytest tests/test_streamed_confidence.py`

## Story 4.1 — Hybrid Search

Adds `hybrid` mode (`rag-hybrid`): Atlas Search keyword results fused with semantic results by Reciprocal Rank Fusion. Needs the API running and `.env` set (`MONGODB_URI`, `VOYAGE_API_KEY`; `GENERATION_*` for answers).

```bash
# Create/reuse the keyword index (once)
uv run python -m building_with_rag.ingestion.keyword_index

# Success: hybrid query with generation
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "criminal breach of trust", "pattern": "hybrid", "limit": 5, "generate_answer": true}'

# Failure: unsupported scope
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "theft", "pattern": "hybrid", "required_acts": ["BNS_2023"]}'

# Chat, streaming
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"model": "rag-hybrid", "messages": [{"role": "user", "content": "criminal breach of trust"}], "stream": true}'
```

Expected: index command ends `chunk_text_index: READY (queryable)`. Query returns `"status":"ok"`, `"pattern":"hybrid"`, results with `fused_rank`/`semantic_rank`/`keyword_rank` and `score == fused_score`, and a cited answer when generation is configured. Unsupported scope returns HTTP 422 `invalid_request`; a missing keyword index returns 503 `retrieval_not_ready`. Chat streams DRAFT text, confidence/Sources footer, and ends with `data: [DONE]`.
