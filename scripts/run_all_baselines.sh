#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/home/minjae/DR-Classification}"
export PYTHONPATH="$ROOT/src"
exec /opt/miniforge3/bin/conda run --no-capture-output -n huang_env python "$ROOT/scripts/run_all_baselines.py" --repo "$ROOT"
