#!/usr/bin/env bash
# One-time (idempotent) setup on a fresh RunPod pod. Repo tree is assumed to already
# be at /workspace/pcdm (uploaded by the orchestrator via runpodctl send / hf upload).
# See PLAN2.md "RunPod mechanics".
set -euo pipefail
export PYTHONUNBUFFERED=1

cd /workspace

export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/workspace/uv
export WANDB_DIR=/workspace/wandb
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$WANDB_DIR"

if [ -f /workspace/.env ]; then
  set -a
  source /workspace/.env
  set +a
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v tmux >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y tmux
fi

cd /workspace/pcdm
uv sync

uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported(), torch.version.cuda)"

if [ ! -f data/train.jsonl ]; then
  uv run hf download guychuk/pcdm-data --repo-type dataset --local-dir data
fi

# candidate cache is built lazily by pcdm/train.py per backbone (data/cache_<backbone>_L<split>.pt)

# GPU smoke: 300 steps, W&B on, then tail the log for peak memory / step time.
# Kill+resume (last.pt auto-resume) is a manual follow-up check, not scripted here.
mkdir -p logs
uv run python pcdm/train.py --name smoke_gpu --steps 300 --eval_every 150 --val_every 50 --ckpt_every 100 --eval_limit 200 --wandb \
  2>&1 | tee logs/smoke_gpu.log
echo "--- tail of logs/smoke_gpu.log (step time / peak memory) ---"
tail -n 20 logs/smoke_gpu.log
echo "pod_setup.sh done"
