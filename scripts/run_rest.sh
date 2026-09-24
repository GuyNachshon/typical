#!/bin/zsh
set -x
cd "$(dirname "$0")/.."   # every command below is repo-root relative
export PYTHONUNBUFFERED=1
scripts/watchdog.sh runs_B.log uv run pcdm/baselines.py B
scripts/watchdog.sh runs_cache.log uv run python pcdm/encode.py
for n in F G D E F_nobank; do
  extra=""; [[ $n == D ]] && extra="--no_null"; [[ $n == E ]] && extra="--hard_only"; [[ $n == G ]] && extra="--cand_null"
  scripts/watchdog.sh runs_${n}_eval.log uv run pcdm/train.py --name $n --eval_only ${=extra}
done
scripts/watchdog.sh runs_bench.log uv run pcdm/bench.py --model runs/F/model.pt
for s in 1 2; do
  scripts/watchdog.sh runs_F_s$s.log uv run pcdm/train.py --name F_s$s --epochs 4 --seed $s
  scripts/watchdog.sh runs_G_s$s.log uv run pcdm/train.py --name G_s$s --cand_null --epochs 4 --seed $s
done
echo REST_DONE
