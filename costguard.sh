#!/bin/zsh
# Cost guard: every 10 min, check each RunPod pod; stop it if idle (no train/baselines/bench python AND GPU util < 5%)
# for 2 consecutive checks. Log to /tmp/costguard.log. Pods: id|ssh-info-file
set -a; source /Users/guynachshon/conductor/workspaces/typical/buffalo/.env; set +a
PODS="llzychzkjm3uec|/tmp/podssh7"
typeset -A idle
while true; do
  for entry in ${=PODS}; do
    pid=${entry%%|*}; sshf=${entry##*|}
    st=$(runpodctl pod get $pid 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('desiredStatus','?'))" 2>/dev/null)
    [ "$st" != "RUNNING" ] && { echo "$(date +%H:%M) $pid $st"; continue; }
    eval "$(cat $sshf)"
    info=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" root@"$IP" 'u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); n=$(ps -eo args | grep -E "python (train|baselines|bench)\.py" | grep -v grep | wc -l); w=$(ps -eo args | grep -E "p[12]_next|pod[123]_tail|pod[345]_boot|pod5_runs|pod5_n3|pod6_boot|pod6_runs|pod7_boot|pod7_runs|pod6_v3|pod6_jev2|pod6_bench2|jevbench_run|scripts/compose|pod4_post[0-9]*|pod4_redo|pod4_b8b|pod4_mmlu2|pod4_e[0-9]*|pod4_bench|pod4_ks|pod4_e3|pod4_post_e3|pod4_e3b|pod4_e3d|pod4_post_e3c|pod4_probes|pod4_e3ms|pod4_msprobe|pod4_e3d2|hf upload|run_gpu|hf download|uv sync" | grep -v grep | wc -l); echo "$u $n $w"' 2>/dev/null)
    if [ -z "$info" ]; then echo "$(date +%H:%M) $pid ssh failed"; continue; fi
    util=${info%% *}; rest=${info#* }; nproc=${rest%% *}; nwait=${rest##* }
    if [ "$nproc" -eq 0 ] && [ "${util:-0}" -lt 5 ] && [ "${nwait:-0}" -eq 0 ]; then
      idle[$pid]=$(( ${idle[$pid]:-0} + 1 ))
      echo "$(date +%H:%M) $pid IDLE (util=$util%, trainers=0, waiters=$nwait) strike ${idle[$pid]}"
      if [ ${idle[$pid]} -ge 2 ]; then echo "$(date +%H:%M) $pid STOPPING (idle 2 checks)"; runpodctl pod stop $pid >/dev/null 2>&1; idle[$pid]=0; fi
    else
      idle[$pid]=0; echo "$(date +%H:%M) $pid busy (util=$util%, trainers=$nproc, waiters=$nwait)"
    fi
  done
  sleep 600
done
