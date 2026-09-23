#!/usr/bin/env bash
# Diagnostic phase after main_s0 (see PLAN2.md): is the tower's memory layer the problem?
set -x
cd "$(dirname "$0")/.."   # repo-root relative (runs/, logs/, *.py)
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
  run_if_needed diag_tap20_frozen uv run python pcdm/train.py --name diag_tap20_frozen --tap_layer 20 --lora_r 0 --wandb --hf_repo "$HF_REPO"
  run_if_needed diag_tap20        uv run python pcdm/train.py --name diag_tap20 --tap_layer 20 --wandb --hf_repo "$HF_REPO"
  run_if_needed B_1.7B            uv run python pcdm/baselines.py B --backbone Qwen/Qwen3-1.7B-Base --name B_1.7B
  run_if_needed B_8B              uv run python pcdm/baselines.py B --backbone Qwen/Qwen3-8B-Base --name B_8B
  echo DIAG_DONE
}
export -f schedule
timeout 6h bash -c schedule
