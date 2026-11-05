# Infrastructure

Bicep for Rig Assist on Azure Container Apps. Everything lives in one resource group. It is
applied from the laptop (`scripts/azure`); CI/CD only swaps image tags, and CI compiles and
lints it (`make infra-lint`). Setup steps are in `docs/deployment.md`.

## Parameters (`main.bicep`)

| Parameter | Default | Meaning |
|---|---|---|
| `location` | the resource group's | Region for every resource |
| `namePrefix` | `rigassist` | Start of every resource name |
| `allowedIp` | required | The only IP allowed to reach the UI; passed at deploy time, never committed |
| `deployApps` | `true` | `false` creates only the supporting resources (see below) |
| `apiImage`, `uiImage` | `''` | Images in the registry; required when `deployApps` is `true` |
| `qdrantImage` | the compose digest | Same pinned Qdrant as `docker-compose.yml` |
| `llmBaseUrl` | `http://llm-not-started.invalid/v1` | The LLM server. `Settings` refuses an empty URL, so between sessions this is a name that can never resolve |
| `llmModel` | `Qwen/Qwen3.6-35B-A3B-FP8` | Model the LLM server serves |
| `apiMinReplicas` | `0` | `1` keeps the API warm during a session |

Outputs: `uiUrl`, `registryLoginServer`, `registryName`, `storageAccountName`, `keyVaultName`,
`apiAppName`, `uiAppName`, `identityClientId`.

## Modules

| Module | Creates |
|---|---|
| `monitoring` | Log Analytics workspace (30-day retention) |
| `registry` | Container Registry (Basic), admin user off |
| `storage` | Storage account (no public access, no account keys, TLS 1.2) and the private Blob container `data` |
| `keyvault` | Key Vault in RBAC mode; its secrets are written by `scripts/azure/set_secrets.py`, not Bicep |
| `identity` | One user-assigned identity with AcrPull, Storage Blob Data Reader and Key Vault Secrets User |
| `environment` | Container Apps environment with the Consumption workload profile |
| `apps` | The `qdrant`, `api` and `ui` Container Apps |

## Two-stage deployment

The apps need images in the registry, data in Blob and secrets in Key Vault before they can
start, and those need the registry, storage and vault to exist. So `deployApps=false` runs
first, then the images, data and secrets go in, then `deployApps=true` adds the apps.

Blob layout in `data`: `index/`, `raw/`, and `qdrant/rig_assist_chunks.snapshot` (the name
Qdrant's `--snapshot` option is given).

## Choices that are easy to break

- **Workload-profiles environment:** init containers in a "Consumption only" environment
  can't use a managed identity, and `blob_fetch` needs one.
- **`allowInsecure: true` on `api` and `qdrant`:** both have internal ingress only and speak
  plain HTTP inside the environment; without it, HTTP is redirected to HTTPS.
- **`QDRANT_URL` ends in `:80`:** with no port, qdrant-client uses 6333, which the ingress
  doesn't serve.
- **`API_UPSTREAM` is the full internal name:** nginx's resolver ignores the search list in
  `resolv.conf`, so a short app name wouldn't resolve.
- **Qdrant sets `command` as well as `args`:** Container Apps' `args` replace the image's CMD,
  which is what starts Qdrant.
- **Init containers count toward the replica's CPU and memory**, and the total must be a valid
  Consumption pair: `qdrant` 0.75 vCPU / 1.5 GiB, `api` 1.25 vCPU / 2.5 GiB.
- **`Microsoft.Storage` API `2026-04-01`,** not the newest GA (`2026-09-01`): Bicep 0.48.1 has
  no types for the newer one, so it couldn't validate the properties.
