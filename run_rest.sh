#!/bin/zsh
set -x
export PYTHONUNBUFFERED=1
./watchdog.sh runs_B.log uv run baselines.py B
./watchdog.sh runs_cache.log uv run python encode.py
for n in F G D E F_nobank; do
  extra=""; [[ $n == D ]] && extra="--no_null"; [[ $n == E ]] && extra="--hard_only"; [[ $n == G ]] && extra="--cand_null"
  ./watchdog.sh runs_${n}_eval.log uv run train.py --name $n --eval_only ${=extra}
done
./watchdog.sh runs_bench.log uv run bench.py --model runs/F/model.pt
for s in 1 2; do
  ./watchdog.sh runs_F_s$s.log uv run train.py --name F_s$s --epochs 4 --seed $s
  ./watchdog.sh runs_G_s$s.log uv run train.py --name G_s$s --cand_null --epochs 4 --seed $s
done
echo REST_DONE
