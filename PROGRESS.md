# Progress

Running log of what's done, so any new session (or person) can pick up where the last one stopped.
`PLAN.md` is the plan; this file is the state. Update it after every task.

## Status

- **Current phase:** 0 (Scaffold): **done.** [PR #1](https://github.com/Riz695/rig-assist/pull/1) merged into `main` on 2026-10-21 (merged by Claude at the user's request, with a merge commit so the test-first history is kept).
- **Plan update:** [PR #2](https://github.com/Riz695/rig-assist/pull/2) merged on 2026-10-21: v1 described accurately; pivot adopted (olmOCR 2 ingestion, fine-tuning, MLflow, LangGraph, no LlamaIndex, Docker Cloud Sandboxes for long jobs). `PLAN.md` on `main` is the plan to follow.
- **Phase 1 in progress** (started 2026-10-22) on branch `phase-1-api` (pushed). Done: chunking, BM25, text-layer parser, OCR client + page cache + routing, ingestion pipeline + `make ingest`. OCR done on a rented GPU (all 982 pages cached in `data/parsed/`, scanned manual searchable). API endpoints done. **Next:** `Dockerfile` (multi-stage, non-root) + `docker-compose.yml`, then the Phase 1 acceptance check (compose up → upload via Swagger → search) and `docs/cloud-jobs.md` (`sbx` verification).
- **Blockers for Phase 1:**
  - [x] Docker installed: it's **Docker Desktop** 29.9.2 (context `desktop-linux`), compose v5.6.0; `hello-world` works. Fine for this project
  - [x] Manuals (PDFs) placed in `data/raw/`: 6 files, 982 pages; `MarineOffshoreLogOpsManual.pdf` (146 pages) has no text layer
  - [x] v1 repo: https://github.com/Riz695/offshore-drilling-rag (linked in `PLAN.md` via PR #2)
  - [x] OCR: **olmOCR 2 only** (user's decision). **Done 2026-10-23 on a rented RunPod RTX 4090 with vLLM 0.11.2** (`docs/ocr-gpu.md`); hosted providers were all unavailable (details below).
    - **Parasail:** user's key works (`/v1/models` lists 96 models) but **no olmOCR among them** (only Qwen2.5-VL-72B, Qwen3-VL-8B/235B). Inference returns 403 "add a valid payment method" for any model ID, so the ID can't be tested without a card.
    - **Cirrascale** (Ai2 Endpoint Platform, `https://ai2endpoints.cirrascale.ai/api`, model `olmOCR-2-7B-1025` per the olmocr README): their page still lists olmOCR 2 (pay-as-you-go, card, daily billing), but the API host **times out** (also outside the sandbox), and on 2026-10-23 the user confirmed the sign-up site `https://ai2endpoints.cirrascale.ai` doesn't load in their browser either. (Their "Get Started" button is an enterprise sales form, not the API.) **Treat as unavailable.**
    - **FriendliAI:** dedicated endpoints only (rent a GPU by the hour), no serverless.
    - DeepInfra deprecated it 2026-05-07; no Hugging Face inference provider; not on OpenRouter.
    - Fallback in PLAN.md: official olmocr pipeline with vLLM on a rented GPU (estimate cost, ask the user first). olmOCR needs an NVIDIA GPU with >= 12 GB (README: tested on RTX 4090, L40S, A100, H100) and ~30 GB disk. vLLM serves an OpenAI-compatible `/v1`, so our client works unchanged (`OCR_BASE_URL=http://<pod>:8000/v1`, model `allenai/olmOCR-2-7B-1025-FP8`). Estimate on a RunPod RTX 4090 ($0.34/h community, $0.69/h secure, per-second billing, price snapshots mid-2026): ~1 h including setup → **about $0.35–0.70, under $2 with margin**. **2026-10-23: user chose this (option 2).** Steps in `docs/ocr-gpu.md` (RunPod RTX 4090, PyTorch template + web terminal, `pip install vllm==0.11.2`, `vllm serve allenai/olmOCR-2-7B-1025-FP8 --max-model-len 16384 --limit-mm-per-prompt '{"video": 0}'`, key via `VLLM_API_KEY`). Waiting for the user to deploy the pod. OCR is sequential (one page at a time): measure s/page first; add concurrency only if the full 982-page run would be slow.
    - 2026-10-23: user put the key in `.env` and also in `.env.example`; caught before any commit (not in tracked files or history), `.env.example` restored, pre-commit hook `no-secrets-in-env-example` added (`de94f5e`).
  - [ ] `sbx`: **installed** (v0.49.1) and user is in the `kvm` group (checked 2026-10-22). `sbx diagnose`: platform checks pass (KVM, mkfs, storage); daemon not running yet, login not checked. Still to do: `sbx daemon start`, `sbx login`, then the verification for `docs/cloud-jobs.md`. Only needed before the full-corpus jobs

## Phase 0: Scaffold

Branch `phase-0-scaffold`. Acceptance: `make lint` + `make test` pass locally and in GitHub Actions ✅; `GET /health` returns `{"status": "ok", "version": ...}` ✅.

| Task | Commits | Notes file (Obsidian) |
|---|---|---|
| Git + public GitHub repo, topics, `.gitignore` | `77ccf41` | `Tasks/P0 - Git and GitHub setup.md` |
| uv project, ruff, pytest, mypy, pre-commit | `988db67` | `Tasks/P0 - Python project setup with uv.md` |
| `config.py` (pydantic-settings) + `.env.example` | `2a4d836` test → `0778a64` feat | `Tasks/P0 - Settings with pydantic-settings.md` |
| FastAPI app + `GET /health` | `6467346` test → `e378441` feat | `Tasks/P0 - FastAPI app and health endpoint.md` |
| `Makefile` | `57fd3dc` | `Tasks/P0 - Makefile.md` |
| `ci.yml` (lint + tests on 3.11 and 3.12) | `6c699fe` | `Tasks/P0 - CI with GitHub Actions.md` |

## Phase 1: Port v1 + OCR ingestion + API + Docker

Branch `phase-1-api`.

| Task | Commits | Notes file (Obsidian) |
|---|---|---|
| Chunking (line-aware, section detection, chunk metadata) | `c10b3f2` test → `788ed99` feat | `Tasks/P1 - Chunking.md` |
| BM25 retriever (ported from v1, JSONL persistence) | `a9f5448` test → `d31c90c` feat | `Tasks/P1 - BM25 retriever.md` |
| Text-layer parser (pdfplumber + tables, scanned-page detection, header/footer stripping) | `a986d0d` test → `09f5fdf` feat | `Tasks/P1 - Text-layer parser.md` |
| OCR client (olmOCR 2), parsed-page cache, per-page routing, OCR settings | `f8b7146` test → `21f9d08` feat | `Tasks/P1 - OCR client and page cache.md` |
| Secret guard: pre-commit hook blocking values in `.env.example` | `de94f5e` chore | (in `Concepts/Environment variables and config.md`) |
| Index store + ingestion pipeline + `make ingest` | `ebf3b91` test → `6e77488` feat → `84d2795` refactor → `b444800` feat → `f53d4c9` fix | `Tasks/P1 - Ingestion pipeline and index store.md` |
| API: `POST/GET /documents`, `POST /search`; BM25 Lucene IDF fix | `d59dd9c` test → `ada9523` test → `5ba332e` fix → `d202314` feat | `Tasks/P1 - Documents and search API.md` |
| OCR on a rented GPU: guide, concurrent OCR (`OCR_CONCURRENCY`), unfinished pages reported missing (token limit, timeouts, 524) | `44dc645`, `c350001` docs; `ab48067` test → `22babc9` test → `7cfb56e` feat; `9044ddc` → `67ee4f1` test → `3f3e458` feat; `30671a9` test → `967e501` feat | `Tasks/P1 - OCR on a rented GPU.md` |

## Decisions made during the project (beyond PLAN.md)

- 2026-10-20: Applied plan fixes #2 (fastembed instead of sentence-transformers), #3 (Qdrant Cloud free tier in prod), #5 (Langfuse Cloud, not self-hosted), #7 (`llama-index-core` for chunking only; PDF parser chosen in Phase 1), #9 (LLM-as-judge as primary answer metric). Already written into `PLAN.md`.
- Not applied yet, still worth handling in their phase:
  - Phase 7: scale-to-zero vs SQLite job store / on-disk BM25 index / background tasks (cap max replicas at 1, load index from Blob at startup, document the limits).
  - Phase 5: check CrewAI + LlamaIndex dependency resolution early.
  - Phase 4: Azure OpenAI v1 endpoint lets one `openai` client serve both providers. Check current docs.
- `LLM_PROVIDER` defaults to `fake`, so a fresh clone runs offline and free. `LLM_MODEL` has no default; it's required for `openai`/`azure`.
- Azure Blob settings are left out of `config.py` until Phase 7 (auth method undecided).
- Test client uses `httpx2`: Starlette 1.9 deprecates `httpx` for `TestClient`.
- Ruff excludes `*.md` (it reformats Python inside Markdown code blocks, e.g. in `PLAN.md`).
- `/health` is a liveness check only: it does not check Qdrant or the index.
- CI actions are pinned to commit SHAs; uv pinned to `0.13.3`.
- 2026-10-21: v1 already has hybrid BM25 + dense + RRF, citation checks, FastAPI, Streamlit, HF Space; it has **no eval and no OCR**. v2's README must not claim v1's features as new.
- 2026-10-21: Pivot adopted (PR #2): A OCR ingestion (olmOCR 2 / Mistral OCR 3, via API: laptop has no NVIDIA GPU), B LangGraph instead of CrewAI, C fine-tuning the embedder (new Phase 5), D MLflow, E drop LlamaIndex. Phases renumbered 0–9 (deploy is now Phase 8).
- 2026-10-21: Docker: install Docker Engine locally (not required to be Desktop). Cloud Sandboxes are CPU-only microVMs ($0.07–$1.12/h, ≤24 h sessions), used only for long CPU jobs.
- 2026-10-21: OCR is **olmOCR 2 only**; Mistral OCR dropped. Planning estimate: ~2,000 input tokens (page image at 1288 px + prompt) and 1,000–2,000 output tokens per page, plus 20% for retries → about $0.0005–0.0007 per page at Parasail's $0.10 / $0.20 per 1M tokens. Scanned manual (146 pages) ≈ $0.07–0.11; full corpus (982 pages) ≈ $0.50–0.70. Confirm with real usage on the first pages.
- 2026-10-22: Chunker packs **whole lines** up to 500 cl100k tokens (overlap 50, line-aligned) instead of v1's raw token windows, so Markdown table rows are never split. Chunks never span pages. `section` = heading in effect at the chunk's first line (conservative regex: Markdown `#`, `N.N…` numbered, `Chapter/Section/Appendix`).
- 2026-10-22: `doc_id` and `doc_title` are passed in by the caller (ingestion pipeline decides the slug scheme).
- 2026-10-22: BM25 keeps `rank-bm25` (v1 parity; unmaintained since 2022 but tiny; `bm25s` is the fallback). Index saved as chunks in JSON Lines and rebuilt on load (no pickle). Zero-score chunks are dropped; `k < 1` raises `ValueError`. `ScoredChunk(chunk, score)` in `retrieval/models.py` is the shared result type.
- 2026-10-22: Text layer read with **pdfplumber only** (v1 also used pypdf). Pages with < 10 letters/digits count as having no text layer. Header/footer stripping counts only pages with text. Whole corpus parses in 56 s locally (no sandbox needed for this step); 146 scanned pages + 3 near-empty pages in the OSH manual have no text.
- 2026-10-22: olmOCR request format, from the olmocr v0.4.27 source (`pipeline.py`, `prompts.py`): `build_no_anchoring_v4_yaml_prompt`, page PNG with longest side 1288 px, `max_tokens` 8000, temperature 0.1 rising on retries, response = YAML front matter (`primary_language`, `is_rotation_valid`, `rotation_correction`, `is_table`, `is_diagram`) + Markdown with **HTML tables**; valid only if `finish_reason == "stop"`.
- 2026-10-22: OCR settings: `OCR_MODE=off|missing|all` (default `off`), `OCR_BASE_URL` (required unless off), `OCR_API_KEY` (optional, for self-hosted vLLM), `OCR_MODEL` (default `allenai/olmOCR-2-7B-1025`; provider IDs differ), `DATA_PARSED_DIR=data/parsed` (gitignored).
- 2026-10-22: OCR client uses the `openai` SDK against any OpenAI-compatible URL, SDK retries off (`max_retries=0`); `ocr_with_retries` does 3 attempts with 2 s / 4 s backoff and olmOCR's rising temperature. `finish_reason != "stop"` is treated as a failure. Cache key = (SHA-256 of PDF bytes, page, parser), one JSON per page, written atomically. Ingestion stops on a page that fails all retries; finished pages stay cached.
- 2026-10-22: `openai` pinned to 3.28.0 (released 2026-10-09), not the newest 3.36.1 (released that day): openai ships several releases a week; prefer ones out ~2 weeks.
- 2026-10-22: **Page numbers are PDF page positions (1-based), not printed page numbers** (e.g. MarineOffshore PDF page 49 is printed "42"). The eval set must use PDF page numbers.
- 2026-10-23: Index on disk = `IndexStore`: `data/index/documents/<doc_id>.json` (DocumentInfo) + `data/index/chunks/<doc_id>.jsonl`, written atomically, chunks first. Re-ingesting replaces a document's files (no duplicates). BM25 is rebuilt in memory from all chunks (140 ms); `BM25Index.save/load` removed. doc_id = slug of the file name with CamelCase split (`marine-offshore-log-ops-manual`); the store rejects non-slugs (path traversal).
- 2026-10-23: First full `make ingest` (OCR off): 6 docs, 1,133 chunks, 58 s, 1.25 GB peak RAM, index 1.9 MB. 149 pages not searchable (146 scanned + 3 in OSH manual).
- 2026-10-23: **OCR runs (olmOCR-2-7B-1025-FP8, vLLM 0.11.2, RunPod RTX 4090, `OCR_CONCURRENCY=8`)**: `missing` run: 145 scanned pages (252,668 in / 64,044 out tokens) + 3 OSH pages; `all` run: the other 834 pages in 22 min (1,427,470 in / 379,232 out tokens). Throughput ≈ 2.6 s/page at concurrency 8 vs ≈ 7 s/page sequential. Total ≈ 1.69 M input / 0.44 M output tokens (≈ $0.26 at Parasail's old list price, for comparison). All 982 pages cached (4 MB). Normal index after `OCR_MODE=missing`: 1,328 chunks; 5 pages without text, all genuinely empty (Marine p111 "SECURITY", p121 "ANNEX" dividers; OSH p4, p7, p62).
- 2026-10-23: OCR failure policy: a page whose every attempt is incomplete (token limit, client timeout, proxy HTTP 524) raises `OcrPageUnreadableError` → page reported missing (or keeps its text layer) and the run continues; connection errors / other HTTP errors still stop the run. Found on Marine p9–10: blank "Document Control Sheet" forms where olmOCR loops on empty `<tr><td></td></tr>` rows. OCR output tables are HTML, one row per line (16 of 145 scanned pages), so the chunker keeps rows intact.
- 2026-10-23: Pod setup lessons (in `docs/ocr-gpu.md`): pasting a key into RunPod's web terminal altered it (bracketed paste), so create the key on the pod and compare SHA-256 fingerprints; the web terminal drops, so start vLLM with `nohup`; the first server ran without `VLLM_API_KEY` (wrong key accepted) and was caught by a pre-OCR auth check; an editor save of `.env` wrote back an old copy and wiped `OCR_BASE_URL`.
- 2026-10-23: API: `create_app(settings)` keeps settings + `SearchIndex` (store + cached BM25, rebuilt after uploads, locked) on `app.state`. Upload = base name only, `.pdf` + `%PDF-` check (415), doc_id clash with a different file (409), sync ingestion (OCR per settings; OCR failure 502). Search: `mode: Literal["bm25"]` (422 otherwise), `top_k` 1–50, empty index 409. No upload size limit yet.
- 2026-10-23: **BM25 IDF changed to Lucene's** ln(1 + (N − n + 0.5)/(n + 0.5)): rank-bm25's Okapi IDF is 0 for a term in 1 of 2 chunks, so a fresh index with one small manual found nothing (v1 has this flaw). Spot-checked real-corpus top-3 unchanged. **Phase 2 `v1_baseline` must use Okapi's IDF** (plain `BM25Okapi`) to reproduce v1.
- Branches created before PR #1 merges have no `.pre-commit-config.yaml`, so commits there need `PRE_COMMIT_ALLOW_NO_CONFIG=1` (Markdown-only changes).

## Docker Cloud Sandboxes credit

Rules are in `PLAN.md` → "Compute". Use it for long CPU jobs; ask before any run estimated over $20.

- Credit: $250 (Docker Agentic Platform signup promo), **claimed 2026-10-21, expires around 2027-04-21** (6 months). Schedule heavy jobs (eval sweeps, fine-tuning) well before then.
- Not yet verified: `sbx` on Linux, getting code/data in, secrets, Docker/compose inside, getting results out → `docs/cloud-jobs.md` (Phase 1).

| Date | Job | Size | Hours | Cost | Credit left |
|---|---|---|---|---|---|
| | | | | | $250.00 |

OCR GPU (RunPod, not the Docker credit): 2026-10-23, RTX 4090, ~1.5 h pod time incl. setup and debugging. **Actual cost: awaiting the user's RunPod billing figure.**

## Pinned versions (as of Phase 0, plus Phase 1 additions)

Phase 1: tiktoken 0.14.0 · rank-bm25 0.2.2 · numpy 2.5.4 · pdfplumber 0.11.10 (pypdfium2 5.14.0, pillow 13.0.0) · openai 3.28.0 · OCR server: vLLM 0.11.2 + allenai/olmOCR-2-7B-1025-FP8

uv 0.13.3 · Python 3.12 (supports 3.11+) · fastapi 0.156.4 · starlette 1.9.0 · uvicorn 0.54.0 · pydantic 2.14.1 · pydantic-settings 2.16.1 · ruff 0.17.1 · pytest 9.1.1 · mypy 2.4.0 · pre-commit 4.7.0 · httpx2 2.15.1 · actions/checkout v7.0.1 · astral-sh/setup-uv v10.3.0

## Deferred / known gaps

- **Phase 2:** `v1_baseline` needs v1's exact token-window chunker (`chunk_text` in v1's `src/rag/chunking.py`, 12 lines); not ported yet.
- **Phase 2:** check embedding truncation: 500 cl100k tokens can exceed bge-small's 512 WordPiece limit. Measure on the corpus; lower the chunk size for dense retrieval if many chunks are cut.
- Section labels are imperfect on some manuals (running headers like cds-guide's "Section 7 Definitions 1-7", headings wrapped over two lines). Header/footer stripping (from v1's parser) should fix the first.

- `make eval`, `make up`, `make down` are placeholders that exit with an error until their phase.
- OCR output has tables as HTML, which may be one long line: the chunker would cut it by tokens. Check real olmOCR output; convert HTML tables to Markdown if needed.
- OCR: rotated pages not handled (olmOCR returns `is_rotation_valid`/`rotation_correction`; its pipeline re-renders rotated). Cache key doesn't include model/prompt version. Check both once real output exists.
- A PDF deleted from `data/raw/` stays in the index (`IndexStore` has no `remove`). Add with the delete/upload API if needed.
- No readiness check yet (would check Qdrant/index); add when those dependencies exist.
