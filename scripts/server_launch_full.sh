#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/home/minjae/DR-Classification}"
mkdir -p "$ROOT/outputs/logs" "$ROOT/outputs/status"
ulimit -n 65535 2>/dev/null || true
tmux kill-session -t fullrun 2>/dev/null || true
tmux new-session -d -s fullrun "ulimit -n 65535 2>/dev/null || true; cd '$ROOT'; while [ ! -f outputs/status/environment.done ]; do echo WAITING_ENV \$(date -Is); sleep 60; done; bash scripts/run_all_baselines.sh '$ROOT' 2>&1 | tee outputs/logs/master.log"
tmux ls
tmux capture-pane -pt fullrun:0 -S -10
