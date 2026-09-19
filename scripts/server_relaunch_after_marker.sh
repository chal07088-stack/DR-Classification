#!/usr/bin/env bash
set -euo pipefail

root="${1:-/home/minjae/DR-Classification}"
marker="${2:?completion marker path is required}"
session="${3:-fullrun}"

echo "WAITING marker=${marker} before relaunching ${session}"
while [[ ! -f "${marker}" ]]; do
  sleep 2
done

echo "MARKER READY $(date -Is): ${marker}"
if tmux has-session -t "${session}" 2>/dev/null; then
  tmux kill-session -t "${session}"
fi
sleep 2
bash "${root}/scripts/server_launch_full.sh" "${root}"
echo "RELAUNCHED ${session} with current on-disk orchestration order"
