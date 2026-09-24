#!/bin/zsh
# Cost guard: every 10 min, check each RunPod pod; stop it if idle (no train/baselines/bench python AND GPU util < 5%)
# for 2 consecutive checks. Log to /tmp/costguard.log. Pods: id|ssh-info-file
set -a; source /Users/guynachshon/conductor/workspaces/typical/buffalo/.env; set +a
# PODS: one line per pod in /tmp/PODS_ACTIVE -> "<id>|<ssh-info-file>|<waiter-pattern>" (workers append; re-read every loop)
typeset -A idle
while true; do
  PODS=$(grep -v "^#" /tmp/PODS_ACTIVE 2>/dev/null | tr "\n" " ")
  for entry in ${=PODS}; do
    pid=${entry%%|*}; rest=${entry#*|}; sshf=${rest%%|*}
    st=$(runpodctl pod get $pid 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('desiredStatus','?'))" 2>/dev/null)
    [ "$st" != "RUNNING" ] && { echo "$(date +%H:%M) $pid $st"; continue; }
    eval "$(cat $sshf)"
    info=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" root@"$IP" 'u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); n=$(ps -eo args | grep -E "python (train|baselines|bench)\.py" | grep -v grep | wc -l); w=$(ps -eo args | grep -E "bash /workspace/|python scripts/|python train|jevbench_run|bench\.py|hf upload|hf download|uv sync|pip install" | grep -v grep | wc -l); echo "$u $n $w"' 2>/dev/null)
    if [ -z "$info" ]; then echo "$(date +%H:%M) $pid ssh failed"; continue; fi
    util=${info%% *}; rest=${info#* }; nproc=${rest%% *}; nwait=${rest##* }
    if [ "$nproc" -eq 0 ] && [ "${util:-0}" -lt 5 ] && [ "${nwait:-0}" -eq 0 ]; then
      idle[$pid]=$(( ${idle[$pid]:-0} + 1 ))
      echo "$(date +%H:%M) $pid IDLE (util=$util%, trainers=0, waiters=$nwait) strike ${idle[$pid]}"
      if [ ${idle[$pid]} -ge 6 ]; then echo "$(date +%H:%M) $pid STOPPING (idle 6 checks = 60 min)"; runpodctl pod stop $pid >/dev/null 2>&1; idle[$pid]=0; fi
    else
      idle[$pid]=0; echo "$(date +%H:%M) $pid busy (util=$util%, trainers=$nproc, waiters=$nwait)"
    fi
  done
  sleep 600
done
