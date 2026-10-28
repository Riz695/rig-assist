# Phase 8: Azure Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy Rig Assist to Azure Container Apps on a free-trial subscription, private to the user's IP, with Bicep infrastructure, keyless GitHub OIDC deploys that swap image tags, a recorded demo, and a tested teardown.

**Architecture:** Three Container Apps in one resource group: `ui` (nginx, external ingress, IP allowlist) proxies `/api` to `api` (internal ingress, 0–1 replica), which talks to `qdrant` (internal, ephemeral disk). Init containers copy data from a private Blob container (index, manuals, Qdrant snapshot) with a user-assigned managed identity; Key Vault holds the LLM and Langfuse keys. The LLM is Qwen on a RunPod pod during sessions only.

**Tech Stack:** Bicep, Azure Container Apps / Container Registry (Basic) / Storage / Key Vault / Log Analytics, azure-cli 2.91.0, azure-identity 1.26.0, azure-storage-blob 12.31.0, GitHub Actions (OIDC), FastAPI, nginx, Qdrant v1.19.2, Playwright (recording, outside the repo).

**Spec:** `docs/superpowers/specs/2026-10-28-phase-8-azure-design.md`

## Global Constraints

- Free-trial subscription only; never upgrade. Anti-goal: any charge beyond the free trial and ~$1 of RunPod credit.
- No secret in Git, an image, a workflow log or a Claude transcript; keys reach Azure via Key Vault, written by a script that reads `.env` and never prints values (if blocked, the user runs it).
- Only the user's IP reaches the site; the IP and other session settings are deploy-time parameters from the laptop, never committed or stored as repo variables.
- One API replica max (`maxReplicas: 1`), `minReplicas` 0 normally, 1 during sessions.
- CI/CD only swaps image tags; full Bicep is applied from the laptop.
- `deploy.yml` runs only if the repo variable `AZURE_DEPLOY_ENABLED` is `true`; otherwise it is skipped, never failed.
- Versions pinned: azure-identity 1.26.0, azure-storage-blob 12.31.0, azure-cli 2.91.0, actionlint-py 1.7.12.25, shellcheck-py 0.11.0.1 (each ≥ 2 weeks old on 2026-10-28); Bicep resource API versions and GitHub actions (by SHA) checked against current docs at implementation time.
- Tests first for logic (`/ready`, Blob fetch): separate `test:` and `feat:` commits; push after every commit. Fake clients in tests; no network.
- Teaching mode: after each task, explain, write the Obsidian note, update PROGRESS.md, wait for "continue".
- Before any billable or account action (RunPod pod, Azure sign-up), read stock/price and ask; terminate pods after; log time and cost.

## Review Focus

- **A file already in the destination but changed in Blob** (an earlier partial copy, or republished data): the fetch must re-download it, not skip it by name. Test in Task 2 (`test_a_changed_blob_is_downloaded_again`).
- **Qdrant reachable but the collection missing** (snapshot restore failed): `/ready` returns 503 naming the collection, so the platform doesn't route traffic to a half-ready API. Test in Task 1 (`test_ready_is_503_when_the_collection_is_missing`).
- **Init container with a missing prefix or no permission:** it must exit non-zero with a clear message so the revision fails visibly, instead of the API starting with an empty index. Test in Task 2 (`test_an_empty_prefix_exits_1_with_the_prefix_named`, `test_an_auth_error_exits_1_without_a_traceback`).
- **Local compose after the nginx Host-header change:** `docker compose up` and CI's `docker` job keep working with the new `API_HOST` default. Check in Task 3 (CI `docker` job + local `/api/health`).
- **Deploys switched off** (`AZURE_DEPLOY_ENABLED` unset, e.g. after the trial lapses): every push still gets a green CI and a skipped (not failed) deploy. Check in Task 6 (a branch push before the variable exists shows the deploy job skipped).

---

### Task 1: Readiness endpoint `GET /ready` (tests first)

**Files:**
- Modify: `src/rig_assist/api/routes/health.py`, `src/rig_assist/api/schemas.py`
- Modify (generated): `ui/openapi.json`, `ui/src/api/schema.d.ts` (`make ui-types`)
- Test: `tests/test_health.py`

**Interfaces:**
- Consumes: `get_search_index(request) -> SearchIndex`; `SearchIndex.is_empty() -> bool`, `SearchIndex.dense.count() -> int` (0 when the collection doesn't exist), `SearchIndex.dense.collection: str`; Qdrant down raises `qdrant_client.http.exceptions.ResponseHandlingException` (as handled in `routes/search.py`).
- Produces: `ReadyResponse(status: Literal["ready"], chunks: int, vectors: int)`; `GET /ready` → 200 `ReadyResponse`, or 503 `{"detail": <reason>}`. Used by Task 4's readiness probe and Task 6's deploy check.

- [ ] **Step 1: Write the failing tests** in `tests/test_health.py` (build the app with `create_app(...)` and override `get_search_index` with a fake whose `is_empty()` / `dense.count()` / `dense.collection` are scripted):

```python
def test_ready_is_200_with_counts_when_index_and_vectors_exist(): ...
    # is_empty False, bm25 has 3 chunks, dense.count() == 3
    # → 200, {"status": "ready", "chunks": 3, "vectors": 3}
def test_ready_is_503_when_nothing_is_indexed(): ...
    # is_empty True → 503, detail contains "No documents are indexed"
def test_ready_is_503_when_the_collection_is_missing(): ...
    # dense.count() == 0, collection "rig_assist_chunks" → 503, detail contains "rig_assist_chunks"
def test_ready_is_503_when_qdrant_is_unreachable(): ...
    # dense.count() raises ResponseHandlingException → 503, detail contains "Qdrant"
def test_ready_is_documented_in_openapi(): ...
    # "/ready" in paths and "ReadyResponse" in components.schemas
```

- [ ] **Step 2: Run** `uv run pytest tests/test_health.py -q` → the five new tests FAIL (404 / missing schema).
- [ ] **Step 3: Commit** `test: GET /ready (index loaded, collection present, Qdrant reachable)` and push.
- [ ] **Step 4: Implement** `ReadyResponse` in `schemas.py` and `ready()` in `routes/health.py` (summary "Readiness check"; `chunks` = number of BM25 chunks, `vectors` = `dense.count()`; reasons reuse the wording of `searchable_retriever` / `qdrant_unavailable` in `api/dependencies.py`, plus the collection name for the missing-collection case).
- [ ] **Step 5: Regenerate the UI types** `make ui-types` and run `uv run pytest tests/test_health.py tests/test_openapi_export.py -q && make lint && make ui-lint` → all pass.
- [ ] **Step 6: Commit** `feat: GET /ready for the platform's readiness probe` and push.

---

### Task 2: Blob fetch for init containers (tests first)

**Files:**
- Create: `src/rig_assist/deploy/__init__.py`, `src/rig_assist/deploy/blob_fetch.py`
- Modify: `pyproject.toml`, `uv.lock` (add `azure-identity==1.26.0`, `azure-storage-blob==12.31.0` to the main dependencies: the API image runs this)
- Test: `tests/test_blob_fetch.py`

**Interfaces:**
- Produces:
  - `class BlobSource(Protocol)`: `list(prefix: str) -> list[BlobInfo]`, `download(name: str, dest: Path) -> None`
  - `BlobInfo(name: str, size: int, md5: bytes | None)` (frozen dataclass; `md5` = the blob's `content_settings.content_md5`)
  - `FetchReport(downloaded: int, skipped: int)` (frozen dataclass)
  - `fetch_prefix(source: BlobSource, prefix: str, dest: Path) -> FetchReport`: blob `prefix + "x/y"` → `dest / "x/y"`; skip only when the local file exists with the same size **and** MD5 (no MD5 on the blob → download); raises `FetchError` when the prefix lists nothing.
  - `class FetchError(Exception)`
  - `AzureBlobSource(account_url: str, container: str)` using `DefaultAzureCredential(managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID"))`
  - CLI `python -m rig_assist.deploy.blob_fetch --account-url URL --container NAME --map PREFIX=DEST [--map ...]` → exit 0 and one line per map (`index/ → /data/index: 5 downloaded, 0 skipped`); exit 1 with a one-line message (no traceback) on `FetchError` or an Azure auth/HTTP error (`azure.core.exceptions.ClientAuthenticationError`, `HttpResponseError`).
  - Used by Task 4 (init containers for `api` and `qdrant`) and Task 5 (publish layout).

- [ ] **Step 1: Write the failing tests** with a `FakeSource` (in-memory dict name → bytes, MD5 computed with `hashlib.md5`) and `tmp_path`:

```python
def test_blobs_under_the_prefix_land_under_dest_keeping_subfolders(): ...
    # {"index/chunks/a.json", "index/bm25.pkl", "raw/x.pdf"}, prefix "index/" → 2 files under dest, raw/ untouched
def test_an_unchanged_file_is_skipped(): ...
    # second run → FetchReport(downloaded=0, skipped=2), source.download not called
def test_a_changed_blob_is_downloaded_again(): ...
    # local file same name, different bytes → downloaded=1
def test_a_blob_without_md5_is_always_downloaded(): ...
def test_an_empty_prefix_raises_fetch_error_naming_it(): ...
def test_an_empty_prefix_exits_1_with_the_prefix_named(capsys): ...   # main([...]) with the fake source injected
def test_an_auth_error_exits_1_without_a_traceback(capsys): ...       # fake raises ClientAuthenticationError
def test_the_cli_reports_each_map(capsys): ...
```

  (`main(argv, source_factory=...)` takes the source factory as a parameter so tests inject the fake.)
- [ ] **Step 2: Run** `uv run pytest tests/test_blob_fetch.py -q` → FAIL (`ModuleNotFoundError: rig_assist.deploy`).
- [ ] **Step 3: Commit** `test: Blob fetch for init containers (subfolders, skip unchanged by size+MD5, clear failures)` and push.
- [ ] **Step 4: Add the dependencies** `uv add azure-identity==1.26.0 azure-storage-blob==12.31.0`; implement `blob_fetch.py` per the interfaces (download to a temp file in the same folder, then rename, so an interrupted copy never leaves a half file under the final name).
- [ ] **Step 5: Run** `uv run pytest tests/test_blob_fetch.py -q && make lint && make test` → all pass.
- [ ] **Step 6: Commit** `feat: Blob fetch CLI for init containers (managed identity)` and push.

---

### Task 3: Images ready for Container Apps

**Files:**
- Modify: `Dockerfile` (bake the models), `ui/nginx/default.conf.template`, `ui/Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` (docker job), `.env.example` (document `MODELS_DIR` default change if any)

**Interfaces:**
- Produces: API image with the embedding (`BAAI/bge-small-en-v1.5`) and reranker (`Xenova/ms-marco-MiniLM-L-6-v2`) models under `/app/models` and `ENV MODELS_DIR=/app/models`; UI image honouring `API_HOST` (Host header sent upstream; default `api:8000` for compose). Task 4 sets `API_UPSTREAM=http://<api app name>` and `API_HOST=<api app name>` in Azure.

- [ ] **Step 1: Bake the models** in the builder stage (download with the project's own loaders `load_text_embedding(..., cache_dir="/app/models")` and `FastEmbedCrossEncoder(..., cache_dir="/app/models")`, forcing the model to load), copy `/app/models` into the runtime stage, `ENV MODELS_DIR=/app/models`.
- [ ] **Step 2: Verify offline start:** `docker build -t rig-assist-api:check . && docker run --rm --network none rig-assist-api:check python -c "from rig_assist.config import Settings; from rig_assist.retrieval.embedder import load_text_embedding; s=Settings(_env_file=None); list(load_text_embedding(s.embedding_model, str(s.models_dir)).embed(['ok']))"` → exits 0 with no download.
- [ ] **Step 3: nginx Host header:** `proxy_set_header Host ${API_HOST};` (replacing `$host`); `ENV API_HOST=api:8000` in `ui/Dockerfile`; `API_HOST: api:8000` in compose; CI's docker job passes `API_HOST` matching its network alias.
- [ ] **Step 4: Verify** `make up` → `curl -s localhost:8080/api/health` returns `{"status":"ok",...}` and `curl -s localhost:8080/api/ready` returns 200; image size noted (expect ~+150 MB).
- [ ] **Step 5: Commit** `feat: images for Container Apps (models baked in, configurable upstream Host header)`, push, and confirm CI (incl. `docker` job) is green.

---

### Task 4: Bicep infrastructure

**Files:**
- Create: `infra/main.bicep`, `infra/modules/{monitoring,registry,storage,keyvault,identity,environment,apps}.bicep`, `infra/README.md` (parameters and what each module creates)
- Modify: `.github/workflows/ci.yml` (job `infra`: `az bicep build --file infra/main.bicep` + `az bicep lint`), `Makefile` (`infra-lint`)

**Interfaces:**
- Consumes: Task 1 `/ready`, `/health`; Task 2 CLI (init containers run the API image with `python -m rig_assist.deploy.blob_fetch`); Task 3 images and `API_HOST`.
- Produces: `main.bicep` parameters `location`, `namePrefix` (default `rigassist`), `allowedIp` (string, required, no default), `apiImage`, `uiImage`, `qdrantImage` (default the compose digest), `llmBaseUrl` (default `''`), `llmModel`, `apiMinReplicas` (default 0); outputs `uiUrl`, `registryLoginServer`, `registryName`, `storageAccountName`, `keyVaultName`, `apiAppName`, `uiAppName`, `identityClientId`. Blob container `data` with prefixes `index/`, `raw/`, `qdrant/` (Task 5 uploads to exactly these).

- [ ] **Step 1: Install the CLI** `uv tool install azure-cli==2.91.0` (record in PROGRESS: outside the repo); `az bicep install`; check current API versions for `Microsoft.App/containerApps`, `managedEnvironments`, `ContainerRegistry`, `Storage`, `KeyVault`, `ManagedIdentity`, `OperationalInsights` on Microsoft Learn and use the newest GA ones.
- [ ] **Step 2: Write the modules** to the spec's architecture:
  - Log Analytics workspace (30-day retention) + Container Apps environment (Consumption).
  - ACR Basic, admin user disabled.
  - Storage account: `allowBlobPublicAccess: false`, `minimumTlsVersion: TLS1_2`, shared-key access disabled; container `data`.
  - Key Vault: RBAC mode, secrets `llm-api-key`, `langfuse-public-key`, `langfuse-secret-key` written later by Task 5's script (not by Bicep).
  - User-assigned identity with AcrPull (registry), Storage Blob Data Reader (account), Key Vault Secrets User (vault).
  - `qdrant` app: internal ingress on 6333, 0.5 vCPU / 1 GiB, `minReplicas` 1 / `maxReplicas` 1, EmptyDir volume `/qdrant/snapshots`, init container fetching `qdrant/` there, restore of `rig_assist_chunks` from that file at start (verify the mechanism, `--snapshot` flag or recover API, against the Qdrant v1.19 docs).
  - `api` app: internal ingress on 8000, 1 vCPU / 2 GiB, `minReplicas` = `apiMinReplicas`, `maxReplicas` 1, EmptyDir `/app/data`, init container fetching `index/` → `/app/data/index` and `raw/` → `/app/data/raw`. Env: `QDRANT_URL=http://<qdrant app>`, `LLM_PROVIDER=openai_compatible`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_THINKING=off`. Secrets as Key Vault references. Liveness `/health`, readiness `/ready`, startup probe generous enough for model load.
  - `ui` app: external ingress on 8080, `ipSecurityRestrictions` allowing only `allowedIp/32`, `API_UPSTREAM`, `API_HOST`, 0.25 vCPU / 0.5 GiB, min 0 / max 1.
- [ ] **Step 3: Verify** `az bicep build --file infra/main.bicep && az bicep lint --file infra/main.bicep` → no errors or warnings (or each remaining warning justified in `infra/README.md`).
- [ ] **Step 4: Commit** `feat: Bicep infrastructure for Azure Container Apps` (+ CI `infra` job), push, CI green.

---

### Task 5: Laptop scripts: publish data, bootstrap, secrets, sessions, teardown

**Files:**
- Create: `scripts/azure/{publish_data.sh,bootstrap.sh,set_secrets.py,session.sh,teardown.sh}`, `docs/deployment.md`
- Modify: `Makefile` (`azure-bootstrap`, `azure-publish-data`, `azure-secrets`, `azure-session-start`, `azure-session-stop`, `azure-teardown`, all documented in `make help`)

**Interfaces:**
- Consumes: Task 4 outputs and parameters; local compose Qdrant (`http://localhost:6333`) and `data/index`, `data/raw`.
- Produces:
  - `bootstrap.sh RG LOCATION`: creates the resource group, builds and pushes the first images (`az acr build`), deploys `main.bicep` with `allowedIp` = the caller's current public IP (looked up at run time, never written to disk), creates the Entra app registration + service principal + federated credentials for `repo:Riz695/rig-assist:ref:refs/heads/main` and `…:ref:refs/heads/phase-8-azure`, grants **AcrPush** on the registry and **Container Apps Contributor** on the resource group, and sets GitHub repo variables `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `ACR_NAME`, `API_APP`, `UI_APP` (identifiers, not secrets) via `gh variable set`. Idempotent (safe to rerun).
  - `publish_data.sh`: snapshot `rig_assist_chunks` from local Qdrant (REST `POST /collections/rig_assist_chunks/snapshots`, download it), upload `data/index` → `index/`, `data/raw` → `raw/`, snapshot → `qdrant/` with `az storage blob upload-batch --auth-mode login --overwrite`; the user gets Storage Blob Data Contributor on the account in `bootstrap.sh` for this.
  - `set_secrets.py`: reads `LLM_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` from `.env` via `Settings`, writes them to Key Vault with `azure-keyvault-secrets` or `az keyvault secret set --file` from a temp file deleted after; prints only names and SHA-256 fingerprints (first 8 hex).
  - `session.sh start BASE_URL` / `session.sh stop`: sets `LLM_BASE_URL` and `minReplicas` 1 (start) / 0 (stop) on `api` with `az containerapp update`, refreshes the UI allowlist to the current IP, waits until `/ready` passes through the UI URL (start).
  - `teardown.sh`: deletes the resource group (waits), the Entra app registration, sets `AZURE_DEPLOY_ENABLED=false`, then verifies `az group exists` is `false` and `az ad app list --display-name` is empty; prints what it checked.
  - `docs/deployment.md`: prerequisites (free-trial sign-up, `uv tool install azure-cli==2.91.0`, `! az login`), the command order, the cost note (free trial; ~$5/month scale-to-zero on pay-as-you-go), the Qdrant-inside-Azure change and why, dropped PLAN items (fine-tuned ONNX publish; Blob ingestion), session and teardown steps, troubleshooting.
- [ ] **Step 1: Write the scripts** (`set -euo pipefail`; every resource name derived from the Bicep outputs, none hard-coded twice). Run `shellcheck scripts/azure/*.sh` (via `uvx shellcheck-py==0.11.0.1` or the system binary) → clean.
- [ ] **Step 2: Dry checks without Azure:** `bash -n` on each script; `uv run python scripts/azure/set_secrets.py --dry-run` prints the three names with fingerprints and no values.
- [ ] **Step 3: Commit** `feat: Azure laptop scripts (bootstrap, data publish, secrets, sessions, teardown) and docs/deployment.md`, push.

---

### Task 6: Deploy workflow (OIDC, tag swap, verify through Azure)

**Files:**
- Create: `.github/workflows/deploy.yml`
- Modify: `.github/workflows/ci.yml` (lint workflows with `actionlint`)

**Interfaces:**
- Consumes: Task 5's repo variables; Task 1's `/ready` (via the platform's readiness state).
- Produces: workflow `deploy` with triggers `workflow_run` (workflows: [CI], types: [completed], branches: [main]), `push` (branches: [phase-8-azure], for pre-merge testing: GitHub only allows `workflow_dispatch` and `workflow_run` for workflow files already on `main`, while `push` runs from the branch's own file), and `workflow_dispatch` (usable after the merge); job condition `vars.AZURE_DEPLOY_ENABLED == 'true' && (github.event_name != 'workflow_run' || github.event.workflow_run.conclusion == 'success')`; `permissions: id-token: write, contents: read`. The `push` trigger and the branch federated credential are removed after the merge is verified (Task 8).

> Spec correction: the design said branch testing would use a manual `workflow_dispatch` trigger; GitHub doesn't allow that before the file is on `main`, so a branch-limited `push` trigger is used instead (same branch federated credential).

- [ ] **Step 1: Write `deploy.yml`:** checkout the CI run's `head_sha` (or the pushed/dispatched ref); `azure/login` (pinned by SHA) with the client/tenant/subscription variables; `az acr login`; build and push `api` and `ui` images tagged `${{ sha }}`; `az containerapp update --image` for `api` then `ui`; verify: wait (bounded, 10 min) until each app's latest revision is `Provisioned`, running and its replica healthy (readiness passed), else fail and print `az containerapp logs show --tail 50`; then `curl` the UI URL from the runner and **expect HTTP 403** (proves the IP restriction from a foreign IP).
- [ ] **Step 2: Verify** `actionlint .github/workflows/*.yml` (via `uvx actionlint-py==1.7.12.25`, pinned in the CI step too) → clean.
- [ ] **Step 3: Commit and push** `feat: deploy workflow (OIDC, image tag swap, verify through Azure)`. Expected on GitHub: CI green; `deploy` triggered by the branch push with its job **skipped**, because `AZURE_DEPLOY_ENABLED` doesn't exist yet (Review Focus).

---

### Task 7: First deployment and acceptance checks (live; stop before billable steps)

**Files:**
- Modify: `PROGRESS.md` (accounts, resource names, costs), `docs/deployment.md` (anything that differed)

- [ ] **Step 1: Ask the user to sign up** for the Azure free trial now (30-day clock starts), then run `! uv tool run --from azure-cli==2.91.0 az login` (or `! az login`). Record the sign-up date in PROGRESS.md. If refused or no Container Apps capacity nearby: stop the Azure part (user's fallback) and record the demo locally.
- [ ] **Step 2: Run** `make azure-bootstrap`, `make azure-publish-data`, `make azure-secrets` (the user runs the last one if the permission system blocks it). Expected: Bicep outputs printed, `qdrant` revision healthy (collection restored: check its log line), `api` at 0 replicas.
- [ ] **Step 3: Enable deploys** `gh variable set AZURE_DEPLOY_ENABLED --body true`; push a commit to `phase-8-azure` (e.g. the PROGRESS update). Expected: `deploy` green, images tagged with the branch head, runner `curl` → 403.
- [ ] **Step 4: Session for the checks:** read RunPod stock, quote the price, wait for the user's go; create the pod with the key from the RunPod secret (as in the safety fix); `make azure-session-start BASE_URL=…`; Playwright (scratchpad, not the repo) on the UI URL: Search (hybrid_rerank hits with page image), Ask (cited answer), Diagnose (live node timeline → work order, Langfuse trace link); 0 console errors / CSP violations. `make azure-session-stop`; terminate the pod; `list-pods` empty; log time and cost.
- [ ] **Step 5: Commit** `docs: first Azure deployment and acceptance checks`, push; open the Phase 8 PR into `main` (summary, evidence, what remains: merge-triggered deploy, recording, teardown); **stop for the user's review and merge.**

---

### Task 8: After the merge: main deploy, recording, teardown, closing PR

**Files:**
- Modify: `PROGRESS.md`, `docs/deployment.md` (teardown verified), `README.md` only if a deployment line is needed now (Phase 9 owns the README)

- [ ] **Step 1: Verify the merge-triggered deploy:** after the user merges, `gh run list --workflow deploy` shows a `workflow_run` run for the merge commit, green, both apps on the merge SHA (`az containerapp show --query properties.template.containers[0].image`). Then delete the `phase-8-azure` federated credential (`az ad app federated-credential delete`) and drop the `push: phase-8-azure` trigger from `deploy.yml` (in the closing PR).
- [ ] **Step 2: Recording session:** pod (stock + price + user's go), `make azure-session-start`, warm-up run, then a Playwright script with `record_video_dir` walks Search → Ask → Diagnose on the UI URL at a readable pace; save the video outside Git (path in PROGRESS.md); `make azure-session-stop`; terminate the pod; log cost.
- [ ] **Step 3: Teardown for real:** `make azure-teardown` → its checks pass (resource group gone, app registration gone, `AZURE_DEPLOY_ENABLED=false`); a push or dispatch afterwards shows `deploy` skipped. Remind the user about the RunPod secret.
- [ ] **Step 4: Phase 8 acceptance** in PROGRESS.md (public URL worked from the user's IP and refused others; merge redeployed via OIDC; teardown tested; $0 Azure; RunPod cost); `make lint && make test && make ui-test`; commit, push a branch `phase-8-closing`, open the closing PR; **stop for review.**
