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
  `VLLM_API_KEY` is set, so nobody else can use the GPU. Always check that a wrong key is refused
  before sending any page (step 6).
- **Limit:** RunPod's proxy closes any request after **100 seconds** (HTTP 524). A normal page takes
  far less; a page where the model loops would fail and be retried (with a higher temperature).

## Steps

### 1. RunPod account
Sign up at https://www.runpod.io and add credit (the smallest top-up is plenty).

### 2. Deploy the pod
In the console: **Pods → Deploy**.
1. GPU: **RTX 4090** (Community Cloud is cheaper; Secure Cloud is more reliable). 1 GPU.
2. Template: an official **RunPod PyTorch** template (it has a web terminal).
3. **Edit Template** (or "Customize deployment"):
   - **Container Disk: 50 GB** (vLLM + PyTorch ≈ 10 GB, model 10 GB, plus room).
   - **Expose HTTP Ports:** add **8000**.
4. Deploy **On-Demand** (Spot can be interrupted mid-run).

### 3. Install vLLM (pod web terminal)
**Connect → Web Terminal**, then:
```bash
pip install "vllm==0.11.2"
vllm --version        # 0.11.2
```

### 4. Create the server key and start vLLM (same terminal)
Create the key **on the pod** and copy it to your laptop afterwards. Pasting a key *into* the
browser terminal can add invisible characters (bracketed-paste markers), and the key then
silently differs.
```bash
export VLLM_API_KEY="olm-$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
echo "$VLLM_API_KEY"; printf %s "$VLLM_API_KEY" | sha256sum | cut -c1-8
nohup vllm serve allenai/olmOCR-2-7B-1025-FP8 --port 8000 --max-model-len 16384 \
  --limit-mm-per-prompt '{"video": 0}' > ~/vllm.log 2>&1 &
```
- `nohup … &` runs vLLM in the background: the web terminal disconnects easily, and anything
  started in the foreground dies with it.
- Note the key and the 8-character fingerprint printed under it.
- Watch startup with `tail -n 20 ~/vllm.log` (re-run it; a closed terminal doesn't matter).
  The first start downloads the model (10 GB); it's ready at `Application startup complete`.
- A new terminal doesn't have `$VLLM_API_KEY` any more, but the running server keeps it.
  To change the key, `pkill -f "vllm serve"` and repeat this step.

### 5. Find the server URL
**Connect → HTTP Service [Port 8000]**. The URL looks like
`https://<POD_ID>-8000.proxy.runpod.net`.

If that link doesn't appear, build the URL from the pod ID shown on the pod's card. RunPod's
"Waiting for service to respond" page (HTTP 502) means the port is exposed but vLLM isn't up yet.

### 6. Point Rig Assist at it (`.env`, never `.env.example`)
Paste the key from step 4 into `.env` in a text editor (close the file before Rig Assist or
another tool edits `.env`, or an editor save can write back an old copy):
```bash
OCR_BASE_URL=https://<POD_ID>-8000.proxy.runpod.net/v1
OCR_API_KEY=<the key printed in step 4>
OCR_MODEL=allenai/olmOCR-2-7B-1025-FP8
```
Leave `OCR_MODE=off` in `.env`; each run sets it explicitly. Check before any OCR spend:
```bash
uv run python -c "
import hashlib, openai; from rig_assist.config import Settings; s = Settings()
k = s.ocr_api_key.get_secret_value()
print('fingerprint', hashlib.sha256(k.encode()).hexdigest()[:8])   # must match step 4
try:
    openai.OpenAI(base_url=s.ocr_base_url, api_key='wrong', max_retries=0).models.list()
    print('NOT PROTECTED: stop and fix VLLM_API_KEY')
except openai.AuthenticationError:
    print('wrong key refused: protected')
print([m.id for m in openai.OpenAI(base_url=s.ocr_base_url, api_key=k).models.list()])"
```
It should print the same fingerprint as step 4, `wrong key refused: protected`, and
`['allenai/olmOCR-2-7B-1025-FP8']`.

### 7. Run, then terminate
```bash
OCR_MODE=missing OCR_CONCURRENCY=8 make ingest   # OCR only pages without a text layer (149)
OCR_MODE=all OCR_CONCURRENCY=8 make ingest       # every page, for the Phase 3 comparison
OCR_MODE=missing make ingest                     # rebuild the normal index (all from cache)
```
`OCR_CONCURRENCY=8` keeps 8 pages in flight: vLLM batches them at nearly the same per-request
speed, so throughput is several times higher than one page at a time. Pages are cached as they
finish, so an interrupted run resumes where it stopped. A page the model can never finish (for
example a blank form it fills with empty table rows until the token limit) is reported as missing
after 3 attempts instead of stopping the run. The `all` run overwrites the index with OCR text for
every page, so finish with the `missing` run to restore the normal index.

Then in RunPod: **Terminate** the pod (not just Stop) so nothing more is billed.
Log the hours and cost in `PROGRESS.md`.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` / KV cache too large at startup | add `--gpu-memory-utilization 0.85` (olmOCR's advice: also lower `--max-model-len` if needed) |
| HTTP 401 from Rig Assist | `OCR_API_KEY` in `.env` doesn't match the server's key: compare fingerprints (steps 4 and 6) |
| A wrong key is accepted | vLLM started without `VLLM_API_KEY`: `pkill -f "vllm serve"` and redo step 4 |
| Web terminal says "connection closed" | reopen it; vLLM keeps running if started with `nohup` (step 4) |
| HTTP 404 on the model | `OCR_MODEL` must equal the name `vllm serve` printed (`allenai/olmOCR-2-7B-1025-FP8`) |
| HTTP 524 | the request took > 100 s (proxy limit); Rig Assist retries it automatically |
| Connection refused / 502 | vLLM isn't up yet (still downloading), or it was started without `nohup` and the terminal closed |
| A page takes minutes | the model is looping (often blank forms/tables); it's retried with a higher temperature, then reported missing |
