# Phase 7: Demo UI (React + TypeScript), live node progress, page images

## Context

Phase 6 (PR #8) is merged. Phase 7 builds the demo UI. The user will record one run-through of the finished project for recruiters, so the UI must show a **real model's reasoning** (a RunPod Qwen pod or Azure later; no scripted fake). Decisions made in conversation (2026-10-27):

- **React + TypeScript instead of Streamlit** (user's choice, to show full-stack range to AI-engineer recruiters). PLAN.md says Streamlit and acceptance says `localhost:8501`, so PLAN.md is updated in the first commit (the user approved the deviation).
- **Live node-by-node progress** in Diagnose: triage ✓ → retrieval ✓ → diagnosis … while the job runs.
- **Citations show both** the snippet and the real manual page as an image.
- When the LLM is down, the UI says so plainly (job error shown, e.g. "LLM unreachable: start a pod") instead of failing silently.

Branch `phase-7-ui` (created from `main` at `6c01254` and pushed). Teaching mode: explain each task, write the Obsidian notes, then wait for "continue". Tests first where marked, with separate `test:`/`feat:` commits, pushing after each.

## Part A: API changes (Python, tests first)

**A1. Live progress in the job store.**
- `jobs.py`: add a `current_node TEXT` column (added to older DBs through `_ADDED_COLUMNS`, the same pattern as `trace_url`). Add `JobStore.node_started(job_id, node)` and `JobStore.node_finished(job_id, step)`, which appends to `steps` and clears `current_node`. Both are allowed only while the job is `running`, enforced with a conditional UPDATE like `_move`. `finish()` still writes the final steps.
- `agents/graph.py`: `TroubleshootingGraph.run(..., on_event=None)`. The `_timed` wrapper calls `on_event("start", node)` before the work and `on_event("end", step)` after it. That's the one place every node already passes through. I'll check the current LangGraph docs on `graph.stream(stream_mode="updates")` as an alternative and explain the choice: stream only reports node ends, not starts.
- `run_job` passes a callback that writes to the store. A failed store write must not kill the job: log it and keep going.
- `JobResponse` gains `current_node: str | None`. `steps` now fill in while the job is `running`.
- Tests: the steps grow while the job runs (a fake graph calls `on_event`), `current_node` is set and cleared, updates are refused after done/failed, the old DB gets the new column, and the API returns partial steps for a running job.

**A2. Page image endpoint.** `GET /documents/{doc_id}/pages/{page}` returns `image/png`.
- Map `doc_id` to the PDF through `IndexStore` (`DocumentInfo.filename` under `settings.data_raw_dir`). Render with **pypdfium2**, which comes through pdfplumber today: make it a direct, pinned dependency. Use a fixed width (~1000 px) and an LRU cache of rendered pages. Set `Cache-Control` so the browser caches too.
- Errors: unknown doc gives 404, page < 1 or > pages gives 404, PDF missing from disk gives 404 with a clear message.
- Tests use a tiny generated PDF fixture (the existing test fixtures, see `tests/test_api_documents.py`).

**A3. CORS: none needed.** nginx proxies `/api/*` to FastAPI in compose, so the browser sees one origin. `npm run dev` uses Vite's dev proxy for the same reason. I'll note this in the teaching notes as the reason we don't add CORS.

## Part B: Frontend (`ui/`, React + TS)

**B0. Toolchain.** Node isn't installed on the laptop. Install the current Node LTS through **nvm** (user-level, no sudo), after checking the docs and getting the user's OK. Scaffold with Vite's `react-ts` template in `ui/`. Pin versions that are at least 2 weeks old with exact versions in `package.json`, plus a committed `package-lock.json`. Tooling: ESLint, Prettier, `tsc --noEmit`. Add `make ui-dev`, `make ui-test`, `make ui-lint`, `make ui-types`.

**B1. API types from OpenAPI.** Use `openapi-typescript` to generate `ui/src/api/schema.d.ts` from the FastAPI OpenAPI JSON. A small script dumps it from `create_app` with fakes, so no server is needed. A thin typed client (`ui/src/api/client.ts`) uses `fetch` and `openapi-fetch`, or a hand-written wrapper over the generated types; I'll decide after checking the docs. CI regenerates the types and fails if they differ, so the UI can't drift from the API.

**B2. App shell + Search tab.** Tabs: Search / Ask / Diagnose. Search has a query box, a mode selector (`bm25 | dense | hybrid | hybrid_rerank`) and top_k, and shows results with doc, page, section, score and a snippet. Every result opens the citation viewer. The header shows API health and the number of indexed documents.

**B3. Citation viewer.** A side panel or modal with doc title and page, the snippet, and the page image from A2, with a loading state. One component is reused by all three tabs.

**B4. Ask tab.** Question in. Out: the answer, with `[doc p.N]` labels rendered as clickable chips, the refusal state ("Not found in the manuals" plus the reason), the citation list, and usage and latency (tokens, cost if known, retrieval/LLM seconds).

**B5. Diagnose tab.**
- Form with description and optional equipment ID, and example scenarios to pick from: 2–3 taken from `eval/scenarios.jsonl`, including an unsafe-temptation one, so the recording shows a block.
- After submitting, poll `GET /jobs/{id}` every second with TanStack Query (`refetchInterval`, which stops at done/failed). Show a **node timeline**: triage → retrieval → diagnosis → safety → report. Each node shows ✓, the current one has a spinner, a node skipped by the missing-info route is greyed out, and each shows its seconds and tokens.
- When done, render the **work order**:
  - equipment and symptoms;
  - probable causes with likelihood badges and evidence chips;
  - recommended actions;
  - **blocked actions** (struck through, red, labelled "blocked by safety review");
  - safety warnings;
  - an escalation banner;
  - open questions;
  - citations;
  - usage, plus a Langfuse trace link if present;
  - the disclaimer.
- When failed: the error in plain words, with a hint when it looks like an LLM connection error.

**B6. Frontend tests.** Vitest + React Testing Library + MSW (fake API). Cover:
- the Search request and its rendering;
- Ask citations clickable, and the refused state;
- Diagnose: the polling sequence queued → running (partial steps) → done renders the timeline and the work order;
- a blocked action appears only under blocked;
- a failed job shows the error;
- the citation viewer requests the right page URL.

Tests first for the work-order rendering rules and the timeline state logic, which are pure functions in `ui/src/lib/` (e.g. `timelineState(steps, currentNode, status)`).

**B7. Docker + compose.** `ui/Dockerfile` is multi-stage: a Node build stage (pinned by digest), then `nginxinc/nginx-unprivileged` (pinned) serving `dist/` with `location /api/ { proxy_pass http://api:8000/; }`. It runs non-root. Add a `ui` service in `docker-compose.yml` on port **8080**, which `depends_on` `api`. CI gets a `ui` job (npm ci, lint, types check, test, build) and the docker job builds the UI image. Add `make up` already covers it.

## Part C: Docs and bookkeeping

- First commit: the design doc in `docs/superpowers/specs/2026-10-27-phase-7-ui-design.md` (this plan, condensed) and a **PLAN.md** Phase 7 edit: React + TS, port 8080, live progress, page images. Also update the PLAN Architecture diagram label "Streamlit UI" → "React UI" and the Repository structure `ui/` entry.
- PROGRESS.md: mark Phase 6 merged (`6c01254`). Add a "Phase 7" section with the order of work, a task table filled after each task, and these decisions. Mark Phase 6 merged in Obsidian `00 Index.md`.
- Obsidian notes per task (`Tasks/P7 - ...`), with new concept notes for React, TypeScript, Vite, the OpenAPI codegen, TanStack Query polling, MSW, nginx reverse proxy and same-origin/CORS. Check for existing concept notes first.

## Order of work (one task at a time, stop for "continue" after each)

1. Design doc + PLAN.md/PROGRESS.md updates (docs commit)
2. A1 live progress (test → feat)
3. A2 page image endpoint (test → feat)
4. B0 Node + Vite scaffold + lint/format/make targets (chore)
5. B1 OpenAPI types + client + CI drift check
6. B6/B5 logic: timeline and work-order rules as pure functions (test → feat)
7. B2 + B3 Search tab + citation viewer (test → feat)
8. B4 Ask tab (test → feat)
9. B5 Diagnose tab (test → feat)
10. B7 Docker + compose + CI
11. Live check with a real model (RunPod pod, quote the price first), screenshots, Phase 7 acceptance, PR → stop for review

## Verification

- `make lint` and `make test` (Python), and `make ui-lint`, `make ui-test` and `npm run build` (UI), all green locally and in CI.
- `make up`, then open `http://localhost:8080`:
  - Search in every mode; a citation opens the snippet and the page image;
  - Ask with the fake LLM shows the flow;
  - Diagnose with a real LLM (pod `docs/llm-gpu.md`, ~$1/h, ask first) shows the nodes ticking live, a work order with a blocked action on an unsafe-temptation scenario, and a working trace link;
  - with the LLM down, the failed job shows a clear message.
- `curl localhost:8000/documents/drilling-manual/pages/46 -o p.png` gives a valid PNG; an unknown doc gives 404.

## Not in scope (stays deferred)

- The safety node's over-blocking (12/17) and the human-in-the-loop stretch. The over-blocking will be visible in the recording. Raise it with the user again before recording.
- SSE/WebSockets: polling is enough at 1 s and keeps the API stateless.
