# Long jobs in Docker Cloud Sandboxes (`sbx`)

Docker Cloud Sandboxes are CPU microVMs (no GPU) that live at most 24 hours. Rig Assist uses
them only for long CPU jobs (eval sweeps, fine-tuning, hard-negative mining) so the laptop
isn't tied up. Every job also runs locally with the same `make` target. Budget rules are in
`PLAN.md` → "Compute"; log every run in `PROGRESS.md`.

Verified on 2026-10-23 with `sbx` v0.49.1 (Pop!_OS 24.04 host, KVM): everything below was
run for real, in two sandboxes, ~8 minutes total at the smallest size.

## What a sandbox gives you

| Item | Value (shell agent, default size) |
|---|---|
| Image | `docker/sandbox-templates:shell-docker` (Ubuntu 26.04, x86_64) |
| Size | 2 CPUs, 3.8 GiB RAM, 20 GB disk (`--cpus` / `--memory` for bigger shapes) |
| Tools | Python 3.14, **uv 0.13.4**, git, make, curl, **Docker 29.9 + Compose v5.6** |
| User | `agent` (uid 1000), workdir `/home/agent/workspace` |
| Lifetime | TTL set at creation (`--ttl`), extendable up to 24 h after creation |
| Network | the account's **cloud** policy (default here: deny-all) + per-sandbox allow rules |

Measured: create 8 s · resume after stop 4 s · copy in 26 MB 21 s (~1.2 MB/s) · `uv sync`
(incl. Python 3.12 download) 8 s · full test suite 12 s · `make ingest` from the OCR cache 54 s
(laptop: 58 s; 2 CPUs aren't faster, so only use sandboxes for jobs that are *long*) · the index
it produced was **byte-for-byte identical** to the laptop's.

## 1. Create a sandbox

```bash
sbx --cloud create shell --name rig-job --ttl 2h --on-timeout delete --no-progress \
  --allow-network github.com --allow-network objects.githubusercontent.com \
  --allow-network release-assets.githubusercontent.com \
  --allow-network pypi.org --allow-network files.pythonhosted.org \
  --allow-network openaipublic.blob.core.windows.net \
  --allow-network registry-1.docker.io --allow-network auth.docker.io \
  --allow-network production.cloudflare.docker.com --allow-network production.cloudfront.docker.com
sbx --cloud ls                      # STATUS ● running; note the sbx_… ID
```
The verification used the `sbx_…` ID in every command; `sbx --help` says the name works too.
- `--ttl … --on-timeout delete` caps the cost if you forget it.
- **Network:** the account default stays deny-all; open only what the job needs, per sandbox:
  GitHub (uv downloads Python 3.12 builds from it), PyPI (packages), `openaipublic.blob.core.windows.net`
  (tiktoken's tokenizer file), Docker Hub (images, **including `production.cloudfront.docker.com`**,
  where layers are actually served; without it pulls fail with a DNS error).
  Add a host later with `sbx --cloud policy allow network --sandbox rig-job HOST`.
- Bigger jobs: `--cpus 8 --memory 16g` (must be a billable shape). Estimate the cost first;
  ask before anything over $20.

## 2. Code and data in

```bash
git archive --format=tar.gz -o /tmp/src.tar.gz HEAD          # committed code only: no .env, no data
tar -czf /tmp/data.tar.gz data/raw data/parsed               # manuals + OCR cache (gitignored)
sbx --cloud cp /tmp/src.tar.gz rig-job:/home/agent/workspace/src.tar.gz
sbx --cloud cp /tmp/data.tar.gz rig-job:/home/agent/workspace/data.tar.gz
sbx --cloud exec rig-job sh -c 'cd ~/workspace && mkdir -p rig-assist \
  && tar -xzf src.tar.gz -C rig-assist && tar -xzf data.tar.gz -C rig-assist'
```
`git archive` ships exactly the committed code (so results trace to a commit). The manuals go
only into your own sandbox, never into an image or repo (PLAN.md rule).

## 3. Run the job (same `make` target as locally)

```bash
sbx --cloud exec rig-job sh -c 'cd ~/workspace/rig-assist && uv sync --locked && make test'
sbx --cloud exec -e OCR_MODE=missing -e OCR_BASE_URL=https://unused.invalid/v1 rig-job \
  sh -c 'cd ~/workspace/rig-assist && make ingest'
```
- `-e KEY=VALUE` passes **non-secret** settings.
- The unreachable `OCR_BASE_URL` makes ingestion use only the copied OCR cache: if a page
  weren't cached it would fail instead of calling (and paying for) OCR.
- `sbx --cloud exec -it rig-job bash` opens a shell for debugging.

## 4. Secrets: via the proxy, never in files

```bash
# Before creating the sandbox (secrets attach at creation time):
sbx --cloud secret set openai --sandbox rig-job                     # built-in service (prompts; not exercised yet)
sbx --cloud secret set-custom --sandbox rig-job --host api.example.com --env OCR_API_KEY
```
- Inside the sandbox the env var holds only a **placeholder** (`sbx-cs-…`); the real value
  never enters the sandbox. The proxy sets the header (default `Authorization: Bearer <secret>`)
  on requests to that exact host, so the openai client works unchanged.
- **Secrets attach when a sandbox is created.** Adding one to a running sandbox (even after
  stop/resume) has no effect: create the secret first, then the sandbox.
- Verified with a dummy value against `httpbin.org/headers`: placeholder in env, real value only
  in the outgoing header, not in `env`.
- Prefer `--command`/`--ref` (or the interactive prompt) over `--value`, which lands in shell history.

## 5. Docker and Compose inside

```bash
sbx --cloud exec rig-job sh -c 'docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.20.0 \
  && curl -s localhost:6333/'
```
Verified with `qdrant/qdrant:latest` (which was 1.20.0): up in 4 s, accepted a collection. `docker compose` is available too, so the
Phase 3 eval stack (`docker compose up qdrant`) can run inside the sandbox. Pin image versions.

## 6. Results out

```bash
sbx --cloud exec rig-job sh -c 'cd ~/workspace/rig-assist && tar -czf ~/workspace/out.tar.gz eval/results mlruns'
sbx --cloud cp rig-job:/home/agent/workspace/out.tar.gz ./
```
Copy results (eval tables, the MLflow file store `mlruns/`) back as one tarball, then commit
what belongs in Git (`eval/results/*.md`) and keep the rest local.

## 7. Stop or delete

```bash
sbx --cloud stop rig-job      # pause: memory + disk kept, resumes in ~4 s, no compute billed
sbx --cloud run shell --name rig-job --detach   # resume
sbx --cloud ttl rig-job       # time left (hard ceiling: 24 h after creation)
sbx --cloud rm --force rig-job                  # delete for good
sbx --cloud ls && sbx --cloud secret ls         # confirm nothing is left
```
Then log the job in `PROGRESS.md` (date, job, size, hours, cost, credit left).

## Gotchas found during verification

- `sbx diagnose` reports "Global network policy: not initialized" until a cloud policy exists;
  the effective default is deny-all. We left it that way and used per-sandbox rules.
- Docker Hub needs `production.cloudfront.docker.com` (layer downloads), not only `registry-1.docker.io`.
- Secrets don't reach a sandbox that already exists (see §4).
- Cloud sandbox names can't contain periods.
