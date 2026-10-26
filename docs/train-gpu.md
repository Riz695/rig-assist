# Training the embedder on a rented GPU

`make train` works on the CPU, but on a 14 GB laptop one step takes ~85 s (one epoch of
96 steps ≈ 2.3 h). On a RunPod RTX 4090 the same epoch takes ~40 s. This is how the Phase 5
runs were made (2026-10-26): pod `ancmyoqg7g0oh1`, ~37 min, ≈ $0.55.

## 1. Pod (RunPod MCP or console)
- GPU: RTX 4090, Secure Cloud ($0.89/h on 2026-10-26; Community had no stock), CUDA floor 12.8.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (official "Runpod Pytorch 2.8.0"
  template); container disk 40 GB; ports `22/tcp` only.
- SSH without touching the account's keys: make a throwaway key and pass its public half as
  the `PUBLIC_KEY` env var (RunPod's images add it to `authorized_keys`).

```bash
ssh-keygen -t ed25519 -N "" -f /tmp/pod_key
# create the pod with env PUBLIC_KEY="$(cat /tmp/pod_key.pub)", then read ssh.direct
```

## 2. Code and data
The repo is public; the data (index + mined pairs, manual text) is copied over SSH and never
leaves the pod otherwise. No `.env` is needed (the defaults match the served config).

```bash
tar czf train_data.tgz data/index data/training/mined
scp -i /tmp/pod_key -P <port> train_data.tgz root@<ip>:/root/
ssh -i /tmp/pod_key -p <port> root@<ip>
git clone https://github.com/Riz695/rig-assist.git && cd rig-assist && git checkout <commit>
tar xzf /root/train_data.tgz
uv venv -p 3.12 .venv
# Same pins as the train group, but torch from PyPI (CUDA build) instead of the CPU index.
uv pip install -e . "sentence-transformers[train]==6.1.0" "transformers==5.19.0" "torch==2.14.1"
```
The torch download is several GB (CUDA libraries); allow ~10 min.

## 3. Train (no MLflow on the pod)
```bash
nohup .venv/bin/python training/train_embedder.py --no-tracking --epochs 3 \
  --mini-batch-size 32 --seed 0 --name ft-inbatch-e3 --no-hard-negatives \
  --out-dir /root/runs/ft-inbatch-e3 > /root/run.log 2>&1 &
```
Each run writes a folder: `run.json` (config, data stats, metrics, log history) + `model/`.

## 4. Back home: copy, log, register, delete the pod
```bash
scp -r -i /tmp/pod_key -P <port> 'root@<ip>:/root/runs/*' data/models/finetuned/
uv run --group train --group eval python training/train_embedder.py --log-run data/models/finetuned/<run>
uv run --group train --group eval python training/train_embedder.py --log-run data/models/finetuned/<best> --register
```
Then terminate the pod and confirm with a pod list that nothing is left running.
