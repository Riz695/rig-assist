# Progress

Running log of what's done, so any new session (or person) can pick up where the last one stopped.
`PLAN.md` is the plan; this file is the state. Update it after every task.

## Status

- **Current phase:** 0 (Scaffold): **done.** [PR #1](https://github.com/Riz695/rig-assist/pull/1) merged into `main` on 2026-10-21 (merged by Claude at the user's request, with a merge commit so the test-first history is kept).
- **Plan update:** [PR #2](https://github.com/Riz695/rig-assist/pull/2) merged on 2026-10-21: v1 described accurately; pivot adopted (olmOCR 2 ingestion, fine-tuning, MLflow, LangGraph, no LlamaIndex, Docker Cloud Sandboxes for long jobs). `PLAN.md` on `main` is the plan to follow.
- **Phase 1 in progress** (started 2026-10-22) on branch `phase-1-api` (pushed). Done: chunking, BM25. **Next:** parsers (pdfplumber + olmOCR client with a fake OCR client in tests) + parsed-page cache, tests first.
- **Blockers for Phase 1:**
  - [x] Docker installed: it's **Docker Desktop** 29.9.2 (context `desktop-linux`), compose v5.6.0; `hello-world` works. Fine for this project
  - [x] Manuals (PDFs) placed in `data/raw/`: 6 files, 982 pages; `MarineOffshoreLogOpsManual.pdf` (146 pages) has no text layer
  - [x] v1 repo: https://github.com/Riz695/offshore-drilling-rag (linked in `PLAN.md` via PR #2)
  - [ ] OCR: **olmOCR 2 only** (user's decision). DeepInfra deprecated it on 2026-05-07; first verify a working provider with a 1-page test call (Parasail, then Cirrascale; fallback: rented GPU + vLLM). The user needs an API key for whichever works, in `.env` only
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
- Branches created before PR #1 merges have no `.pre-commit-config.yaml`, so commits there need `PRE_COMMIT_ALLOW_NO_CONFIG=1` (Markdown-only changes).

## Docker Cloud Sandboxes credit

Rules are in `PLAN.md` → "Compute". Use it for long CPU jobs; ask before any run estimated over $20.

- Credit: $250 (Docker Agentic Platform signup promo), **claimed 2026-10-21, expires around 2027-04-21** (6 months). Schedule heavy jobs (eval sweeps, fine-tuning) well before then.
- Not yet verified: `sbx` on Linux, getting code/data in, secrets, Docker/compose inside, getting results out → `docs/cloud-jobs.md` (Phase 1).

| Date | Job | Size | Hours | Cost | Credit left |
|---|---|---|---|---|---|
| | | | | | $250.00 |

## Pinned versions (as of Phase 0, plus Phase 1 additions)

Phase 1: tiktoken 0.14.0 · rank-bm25 0.2.2 · numpy 2.5.4

uv 0.13.3 · Python 3.12 (supports 3.11+) · fastapi 0.156.4 · starlette 1.9.0 · uvicorn 0.54.0 · pydantic 2.14.1 · pydantic-settings 2.16.1 · ruff 0.17.1 · pytest 9.1.1 · mypy 2.4.0 · pre-commit 4.7.0 · httpx2 2.15.1 · actions/checkout v7.0.1 · astral-sh/setup-uv v10.3.0

## Deferred / known gaps

- **Phase 2:** `v1_baseline` needs v1's exact token-window chunker (`chunk_text` in v1's `src/rag/chunking.py`, 12 lines); not ported yet.
- **Phase 2:** check embedding truncation: 500 cl100k tokens can exceed bge-small's 512 WordPiece limit. Measure on the corpus; lower the chunk size for dense retrieval if many chunks are cut.
- Section labels are imperfect on some manuals (running headers like cds-guide's "Section 7 Definitions 1-7", headings wrapped over two lines). Header/footer stripping (from v1's parser) should fix the first.

- `make ingest`, `make eval`, `make up`, `make down` are placeholders that exit with an error until their phase.
- No readiness check yet (would check Qdrant/index); add when those dependencies exist.
