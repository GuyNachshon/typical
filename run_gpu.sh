#!/usr/bin/env bash
# Full GPU run schedule (PLAN2.md "Run schedule", priority order). Idempotent: each
# run is skipped if runs/<name>/results.json already exists, so a crashed/restarted
# pod just continues. No `set -e` -- one run failing must not block the rest.
set -x
export PYTHONUNBUFFERED=1

mkdir -p logs runs
export HF_REPO=guychuk/pcdm-runs

run_if_needed() {
  local name="$1"; shift
  if [ -f "runs/${name}/results.json" ]; then
    echo "skip ${name}: runs/${name}/results.json already exists"
    return 0
  fi
  echo "=== ${name}: $* ==="
  "$@" 2>&1 | tee "logs/${name}.log"
}
export -f run_if_needed

schedule() {
  run_if_needed main_s0       uv run python train.py --name main_s0 --wandb --hf_repo "$HF_REPO"
  run_if_needed B_1.7B        uv run python baselines.py B --backbone Qwen/Qwen3-1.7B-Base --name B_1.7B
  run_if_needed B_8B          uv run python baselines.py B --backbone Qwen/Qwen3-8B-Base --name B_8B
  run_if_needed C_lora        uv run python baselines.py C --name C_lora
  run_if_needed bench         uv run python bench.py --model runs/main_s0 --backbone Qwen/Qwen3-1.7B-Base --name bench
  run_if_needed abl_frozen    uv run python train.py --name abl_frozen --lora_r 0 --wandb --hf_repo "$HF_REPO"
  run_if_needed abl_nohybrid  uv run python train.py --name abl_nohybrid --no_hybrid --wandb --hf_repo "$HF_REPO"
  run_if_needed abl_nlionly   uv run python train.py --name abl_nlionly --mix nlionly --wandb --hf_repo "$HF_REPO"
  run_if_needed abl_statenull uv run python train.py --name abl_statenull --no_cand_null --wandb --hf_repo "$HF_REPO"
  run_if_needed main_s1       uv run python train.py --name main_s1 --seed 1 --wandb --hf_repo "$HF_REPO"
  run_if_needed main_4B       uv run python train.py --name main_4B --backbone Qwen/Qwen3-4B-Base --lora_layers 10 --wandb --hf_repo "$HF_REPO"
}
export -f schedule

timeout 14h bash -c schedule

uv run hf upload guychuk/pcdm-runs runs runs --repo-type model || echo "warning: hf upload of runs/ failed"

if [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl pod stop "$RUNPOD_POD_ID"
fi
