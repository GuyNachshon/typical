#!/usr/bin/env bash
# Pod 2 half of phase 3 (shared /workspace volume with pod 1).
set -x
export PYTHONUNBUFFERED=1
export HF_REPO=guychuk/pcdm-runs
mkdir -p logs runs
run_if_needed() {
  local name="$1"; shift
  if [ -f "runs/${name}/results.json" ]; then echo "skip ${name}"; return 0; fi
  echo "=== ${name}: $* ==="; "$@" 2>&1 | tee "logs/${name}.log"
}
export -f run_if_needed
schedule() {
  until [ -f data_v3/train.jsonl ] && [ -f data_v3/eval/snli_test_hyponly.jsonl ]; do sleep 20; done
  run_if_needed abl_nohybrid   uv run --no-sync python train.py --name abl_nohybrid --tap_layer 20 --zscore --lora_r 16 --data data_v3 --no_hybrid --wandb --hf_repo "$HF_REPO"
  run_if_needed main_v3_s1     uv run --no-sync python train.py --name main_v3_s1 --tap_layer 20 --zscore --lora_r 16 --data data_v3 --seed 1 --wandb --hf_repo "$HF_REPO"
  run_if_needed B_1.7B         uv run --no-sync python baselines.py B --backbone Qwen/Qwen3-1.7B-Base --data data_v3 --name B_1.7B
  run_if_needed B_8B           uv run --no-sync python baselines.py B --backbone Qwen/Qwen3-8B-Base --data data_v3 --name B_8B
  echo POD2_DONE
}
export -f schedule
timeout 6h bash -c schedule
uv run --no-sync hf upload guychuk/pcdm-runs runs runs --repo-type model || true
[ -n "${RUNPOD_POD_ID:-}" ] && runpodctl pod stop "$RUNPOD_POD_ID"
