# OCR on a rented GPU (olmOCR 2 + vLLM on RunPod)

No hosted API serves olmOCR 2 any more (checked 2026-10-23: DeepInfra deprecated it,
Parasail no longer lists it, Cirrascale's endpoint doesn't load). So we run the model
ourselves for about an hour on a rented GPU, OCR every page we need, and shut it down.

Rig Assist's OCR client speaks the OpenAI-compatible API, and vLLM serves exactly
that, so **no code changes**: only `OCR_BASE_URL`, `OCR_API_KEY` and `OCR_MODEL` in `.env`.

Results are cached in `data/parsed/` (one file per page), so a page is never paid for twice,
and the pod can be deleted as soon as the run ends.

## Cost

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB (olmOCR needs ≥ 12 GB; tested by AI2 on 4090, L40S, A100, H100) |
| Price | ~$0.34/h Community Cloud, ~$0.69/h Secure Cloud, billed per second (RunPod, mid-2026 prices: check the console) |
| Time | ~15 min setup (install vLLM, download the 10 GB model) + OCR time (measured on the first pages) |
| Estimate | **about $0.35–0.70 for an hour; under $2 with margin** |

Disk is billed separately while the pod exists, a few cents for an hour. **Terminate** the pod when done
(step 7); a stopped pod still bills its disk.

## Why this setup

- **Model:** `allenai/olmOCR-2-7B-1025-FP8`, the version AI2 recommends for inference (FP8 = 8-bit weights,
  half the memory of 16-bit, ~same accuracy).
- **vLLM 0.11.2:** the exact version olmOCR pins (`pyproject.toml`) and builds its Docker image on.
- **Server flags** (from olmOCR's own `pipeline.py`): `--max-model-len 16384` (page image + up to 8,000
  output tokens fit) and `--limit-mm-per-prompt '{"video": 0}'` (skips the video encoder, freeing GPU
  memory for the KV cache).
- **Security:** RunPod's proxy URL is **public**. vLLM requires an API key on every request when
  `VLLM_API_KEY` is set, so nobody else can use the GPU. The key is set as a pod environment
  variable, so it never appears in a command line or the logs.
- **Limit:** RunPod's proxy closes any request after **100 seconds** (HTTP 524). A normal page takes
  far less; a page where the model loops would fail and be retried (with a higher temperature).

## Steps

### 1. Make an API key for the server (your own terminal)
```bash
python3 -c "import secrets; print('olm-' + secrets.token_urlsafe(24))"
```
Keep it for steps 3 and 6. It only protects this short-lived pod.

### 2. RunPod account
Sign up at https://www.runpod.io and add credit (the smallest top-up is plenty).

### 3. Deploy the pod
In the console: **Pods → Deploy**.
1. GPU: **RTX 4090** (Community Cloud is cheaper; Secure Cloud is more reliable). 1 GPU.
2. Template: an official **RunPod PyTorch** template (it has a web terminal).
3. **Edit Template** (or "Customize deployment"):
   - **Container Disk: 50 GB** (vLLM + PyTorch ≈ 10 GB, model 10 GB, plus room).
   - **Expose HTTP Ports:** add **8000**.
   - **Environment Variables:** add `VLLM_API_KEY` = the key from step 1.
4. Deploy **On-Demand** (Spot can be interrupted mid-run).

### 4. Start vLLM (pod web terminal)
**Connect → Web Terminal** (or Jupyter → Terminal), then:
```bash
: "${VLLM_API_KEY:?VLLM_API_KEY is not set: add it in the pod's environment variables}"
pip install "vllm==0.11.2"
vllm serve allenai/olmOCR-2-7B-1025-FP8 \
  --port 8000 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"video": 0}'
```
The first line stops right away if the key is missing, rather than starting a server open to anyone.
The install takes a few minutes; the first start downloads the model (10 GB). It's ready when the
log shows `Application startup complete`. Leave the terminal open.

### 5. Find the server URL
**Connect → HTTP Service [Port 8000]**. The URL looks like
`https://<POD_ID>-8000.proxy.runpod.net`.

### 6. Point Rig Assist at it (`.env`, never `.env.example`)
```bash
OCR_BASE_URL=https://<POD_ID>-8000.proxy.runpod.net/v1
OCR_API_KEY=<the key from step 1>
OCR_MODEL=allenai/olmOCR-2-7B-1025-FP8
```
Leave `OCR_MODE=off` in `.env`; each run sets it explicitly. Check before any OCR spend:
```bash
uv run python -c "
import openai; from rig_assist.config import Settings; s = Settings()
c = openai.OpenAI(base_url=s.ocr_base_url, api_key=s.ocr_api_key.get_secret_value())
print([m.id for m in c.models.list()])"
```
It should print `['allenai/olmOCR-2-7B-1025-FP8']`.

### 7. Run, then terminate
```bash
OCR_MODE=missing make ingest   # OCR only pages without a text layer (149 pages)
OCR_MODE=all make ingest       # OCR every page, for the Phase 3 parser comparison
```
Then in RunPod: **Terminate** the pod (not just Stop) so nothing more is billed.
Log the hours and cost in `PROGRESS.md`.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` / KV cache too large at startup | add `--gpu-memory-utilization 0.85` (olmOCR's advice: also lower `--max-model-len` if needed) |
| HTTP 401 from Rig Assist | `OCR_API_KEY` in `.env` doesn't match the pod's `VLLM_API_KEY` |
| HTTP 404 on the model | `OCR_MODEL` must equal the name `vllm serve` printed (`allenai/olmOCR-2-7B-1025-FP8`) |
| HTTP 524 | the request took > 100 s (proxy limit); Rig Assist retries it automatically |
| Connection refused / 502 | vLLM isn't up yet (still downloading) or the terminal running it was closed |
