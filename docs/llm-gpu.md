# An open LLM on a rented GPU (Qwen3.6-35B-A3B + vLLM on RunPod)

The answer eval (and later the agent comparison) needs many LLM calls. Gemini's free tier
allows 20 requests per day per model, so instead we rent a GPU for an hour or two, serve an
open-weights model with vLLM, and point Rig Assist at it. vLLM speaks the OpenAI-compatible
API, so this is configuration only: `LLM_PROVIDER=openai_compatible`, `LLM_BASE_URL`,
`LLM_API_KEY`, `LLM_MODEL` in `.env`. The prompts (with manual excerpts) go only to our own pod.

Same procedure as the OCR run (`docs/ocr-gpu.md`); the lessons from it are built in.

## Choices (checked 2026-10-25)

| Item | Choice | Why |
|---|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` | Official Qwen FP8 weights, Apache-2.0. Mixture of experts: 35B parameters, ~3B active per token, so it's fast. 37.5 GB of weights |
| GPU | **RTX 6000 Ada, 48 GB** (Community ~$0.74/h); fallback **L40S, 48 GB** (~$0.79/h) | 37.5 GB doesn't fit a 24 GB card; Ada GPUs run FP8 natively (the cheaper A6000 is Ampere, without native FP8) |
| vLLM | **0.31.0** (2026-10-05) | Newest release at least two weeks old; the model card asks for ≥ 0.19 |
| Disk | Container disk **90 GB** | model 37.5 GB + vLLM/PyTorch ~10 GB + room |
| Context | `--max-model-len 32768` | Our prompts are ~3k tokens + up to 4k output; the model's 262k default wouldn't fit next to the weights |

Prices: RunPod pricing page, 2026-10-25; check the console. **Estimate: about 1.5 h (setup
~30 min, mostly the 37.5 GB download; eval ~30–60 min) ≈ $1.10–1.60**, plus a few cents of disk.

## Route used on 2026-10-25: create the pod through the RunPod MCP

With the official RunPod Claude Code plugin (`claude plugin install runpod@runpod`, then
`/mcp` → runpod → Sign in with Runpod; OAuth, no API key stored), Claude creates the pod with
vLLM's official image, so there is no web terminal and no `pip install`:

- **Stock first:** on 2026-10-25 Community Cloud had no 48 GB card; **Secure Cloud** had the
  RTX 6000 Ada (LOW in US-IL-1, US-KS-2, US-WA-1) at **$0.99/h** (`list-gpu-types`,
  `include: AVAILABILITY`, `product: POD`). Read both clouds: a cheap Community price means
  nothing without stock.
- **Pod:** `cloud: SECURE`, `gpu: {id: "NVIDIA RTX 6000 Ada Generation", minCudaVersion: "12.8"}`,
  `image: vllm/vllm-openai:v0.31.0` (the image's CUDA is 12.9; a newer host driver is fine),
  `entrypoint: ["vllm", "serve"]`, `cmd: [model, --host 0.0.0.0, --port 8000,
  --max-model-len 32768, --reasoning-parser qwen3, --language-model-only]`,
  `env: {VLLM_API_KEY: ...}`, `disk: 90`, `ports: ["8000/http"]`.
- **Key:** generated on the laptop straight into `.env` (`LLM_API_KEY`) and passed as the
  pod's `VLLM_API_KEY`. It does appear in that tool call: acceptable for a pod that is
  terminated after the run. Use the manual steps below to keep a key out of any transcript.
- **Boot:** the image was cached on the host; the 34.9 GiB of weights downloaded in 18 s.
  Then step 5 below, then step 6.

## Manual steps (web terminal)

### 1. Deploy the pod
RunPod console → **Pods → Deploy**:
1. GPU: **RTX 6000 Ada** (or L40S), 1 GPU, **Community Cloud**, **On-Demand** (not Spot).
2. Template: **RunPod PyTorch** (has a web terminal).
3. **Edit Template**: Container Disk **90 GB**; **Expose HTTP Ports: 8000**.
4. If the page has a **CUDA version** filter, pick **12.8 or newer** (recent vLLM wheels need a
   recent driver).

### 2. Install vLLM (pod web terminal)
**Connect → Web Terminal**:
```bash
pip install "vllm==0.31.0"
vllm --version        # 0.31.0
```

### 3. Create the key and start the server
Create the key **on the pod** (pasting into the web terminal can alter it), then start vLLM in
the background (`nohup`), because the web terminal drops:
```bash
export VLLM_API_KEY="llm-$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
echo "$VLLM_API_KEY"; printf %s "$VLLM_API_KEY" | sha256sum | cut -c1-8
nohup vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 --max-model-len 32768 \
  --reasoning-parser qwen3 --language-model-only > ~/vllm.log 2>&1 &
```
- `--reasoning-parser qwen3`: the model thinks before answering (`<think>…</think>`); the parser
  moves that out of the answer text, so citations are checked on the answer only.
- `--language-model-only`: skips the vision encoder (we send text), leaving more GPU memory.
- `--default-chat-template-kwargs '{"enable_thinking": false}'` (used for `make pairs`, Phase 5): turns
  thinking off server-side for bulk generation that doesn't need it. Output fell to 41–79 tokens
  per call and 1,160 query requests took 2.5 min at 16 in flight. Leave it off for `/ask` and the
  judge, which were measured with thinking on.
- Boot time varies by host: 4 min on a host that had compiled the model before, 7.5 min on a fresh
  one (`torch.compile` + CUDA graph capture). A 502 from the proxy during that time is normal;
  read the pod log before assuming a failure.
- Copy the key and its 8-character fingerprint. Watch startup with `tail -n 20 ~/vllm.log`; the
  first start downloads 37.5 GB. Ready at `Application startup complete`.

### 4. URL and `.env`
**Connect → HTTP Service [Port 8000]** gives `https://<POD_ID>-8000.proxy.runpod.net`.
In `.env` (a text editor; close it before tools edit `.env`):
```bash
LLM_PROVIDER=openai_compatible
LLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8
LLM_BASE_URL=https://<POD_ID>-8000.proxy.runpod.net/v1
LLM_API_KEY=<the key from step 3>
```

### 5. Check before sending any manual text
```bash
uv run python -c "
import hashlib, openai; from rig_assist.config import Settings; s = Settings()
k = s.llm_api_key.get_secret_value()
print('fingerprint', hashlib.sha256(k.encode()).hexdigest()[:8])   # must match step 3
try:
    openai.OpenAI(base_url=s.llm_base_url, api_key='wrong', max_retries=0).models.list()
    print('NOT PROTECTED: stop and fix VLLM_API_KEY')
except openai.AuthenticationError:
    print('wrong key refused: protected')
print([m.id for m in openai.OpenAI(base_url=s.llm_base_url, api_key=k).models.list()])"
```
Expect the same fingerprint, `wrong key refused: protected`, and `['Qwen/Qwen3.6-35B-A3B-FP8']`.

### 6. Run, then terminate
```bash
make eval-answers          # resumes from saved answers if interrupted
```
Then **Terminate** the pod in RunPod (a stopped pod still bills its disk) and log the hours and
cost in `PROGRESS.md`.

## Thinking on or off
Qwen3.6 thinks by default. Thinking tokens count against `MAX_ANSWER_TOKENS` (4,096) and the
RunPod proxy closes requests after 100 s. If answers get cut off (`finish_reason=length`) or
time out (HTTP 524), restart the server with thinking off:
```bash
pkill -f "vllm serve"
nohup vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 --max-model-len 32768 \
  --reasoning-parser qwen3 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}' > ~/vllm.log 2>&1 &
```
(`VLLM_API_KEY` must still be set in that terminal, or the server starts unprotected: step 5
catches that.) Whichever mode is used is recorded with the results.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` at startup | lower `--max-model-len` to 16384, or add `--gpu-memory-utilization 0.95` |
| CUDA / driver version error on start | redeploy on a pod with CUDA ≥ 12.8 (step 1.4) |
| HTTP 401 | `LLM_API_KEY` ≠ the server's key: compare fingerprints (steps 3 and 5) |
| A wrong key is accepted | vLLM started without `VLLM_API_KEY`: `pkill -f "vllm serve"`, redo step 3 |
| HTTP 404 on the model | `LLM_MODEL` must equal the name `vllm serve` printed |
| HTTP 524 | a request took > 100 s (proxy limit): see "Thinking on or off" |
| 502 / connection refused | vLLM isn't up yet (still downloading), or wasn't started with `nohup` |
