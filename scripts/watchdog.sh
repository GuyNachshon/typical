#!/bin/zsh
# usage: ./watchdog.sh <log> <cmd...>  — reruns cmd if its log hasn't grown for 5 min (MPS stream hang)
log=$1; shift
while true; do
  "$@" > "$log" 2>&1 & pid=$!
  prev=-1; stall=0
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    sz=$(stat -f %z "$log" 2>/dev/null || echo 0)
    if [[ "$sz" == "$prev" ]]; then stall=$((stall+1)); else stall=0; fi
    prev=$sz
    if (( stall >= 5 )); then echo "[watchdog] no log growth for 5 min, restarting: $*" >&2; pkill -P $pid; kill $pid; sleep 3; break; fi
  done
  wait $pid 2>/dev/null; rc=$?
  (( stall < 5 )) && return $rc
done
