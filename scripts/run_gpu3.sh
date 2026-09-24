#!/usr/bin/env bash
# Phase 3 (after the tap-20 diagnostics). LORA_FLAGS is decided from diag_tap20 vs diag_tap20_frozen.
set -x
cd "$(dirname "$0")/.."   # repo-root relative (runs/, logs/, *.py)
export PYTHONUNBUFFERED=1
export HF_REPO=guychuk/pcdm-runs
export LORA_FLAGS="${LORA_FLAGS:---lora_r 16}"  # explicit; empty would fall through to the default
mkdir -p logs runs
run_if_needed() {
  local name="$1"; shift
  if [ -f "runs/${name}/results.json" ]; then echo "skip ${name}"; return 0; fi
  echo "=== ${name}: $* ==="; "$@" 2>&1 | tee "logs/${name}.log"
}
export -f run_if_needed
schedule() {
  [ -f data_v3/train.jsonl ] || uv run --no-sync hf download guychuk/pcdm-data --repo-type dataset --local-dir data_v3
  run_if_needed main_v2        uv run --no-sync python pcdm/train.py --name main_v2 --tap_layer 20 --zscore $LORA_FLAGS --wandb --hf_repo "$HF_REPO"
  run_if_needed main_v3        uv run --no-sync python pcdm/train.py --name main_v3 --tap_layer 20 --zscore $LORA_FLAGS --data data_v3 --steps 24000 --wandb --hf_repo "$HF_REPO"
  run_if_needed abl_nohybrid   uv run --no-sync python pcdm/train.py --name abl_nohybrid --tap_layer 20 --zscore $LORA_FLAGS --data data_v3 --no_hybrid --wandb --hf_repo "$HF_REPO"
  run_if_needed main_v3_s1     uv run --no-sync python pcdm/train.py --name main_v3_s1 --tap_layer 20 --zscore $LORA_FLAGS --data data_v3 --seed 1 --wandb --hf_repo "$HF_REPO"
  run_if_needed bench          uv run --no-sync python pcdm/bench.py --model runs/main_v3 --backbone Qwen/Qwen3-1.7B-Base --name bench
  run_if_needed B_1.7B         uv run --no-sync python pcdm/baselines.py B --backbone Qwen/Qwen3-1.7B-Base --data data_v3 --name B_1.7B
  run_if_needed B_8B           uv run --no-sync python pcdm/baselines.py B --backbone Qwen/Qwen3-8B-Base --data data_v3 --name B_8B
  echo PHASE3_DONE
}
export -f schedule
timeout 9h bash -c schedule
uv run --no-sync hf upload guychuk/pcdm-runs runs runs --repo-type model || echo "warning: hf upload failed"
[ -n "${RUNPOD_POD_ID:-}" ] && runpodctl pod stop "$RUNPOD_POD_ID"
