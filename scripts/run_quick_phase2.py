"""Run B0, B1-S and B1-P sequentially after the Phase 1 hard gate."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpair.benchmark.eyepacs import SPLIT_VERSION, require_phase1_hard_gate
from drpair.benchmark.quick_training import load_toml_config, phase2_analysis, train_method
from drpair.runtime import save_json

CONFIGS = {
    "B0": ROOT / "configs/benchmark/convnext_single.toml",
    "B1-S": ROOT / "configs/benchmark/huang_single.toml",
    "B1-P": ROOT / "configs/benchmark/huang_paired.toml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", choices=list(CONFIGS))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--eyepacs-only-provisional", action="store_true")
    args = parser.parse_args()
    phase1 = require_phase1_hard_gate(
        ROOT, allow_eyepacs_only_provisional=args.eyepacs_only_provisional
    )
    phase_root = ROOT / f"results/runs/phase2_{SPLIT_VERSION}"
    status_path = ROOT / f"results/phase2_{SPLIT_VERSION}_status.json"
    methods = [args.method] if args.method else list(CONFIGS)
    state = {
        "phase": 2,
        "split_version": SPLIT_VERSION,
        "scope": phase1.get("scope", "DUAL_DATASET_BENCHMARK"),
        "status": "running",
        "methods": {},
    }
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding="utf-8"))
        state["status"] = "running"
        state.pop("error", None)
    save_json(status_path, state)
    try:
        for method in methods:
            summary_path = phase_root / method / "summary.json"
            if summary_path.exists() and not args.smoke:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("status") == "complete":
                    print(f"[{method}] already complete; skipping", flush=True)
                    state["methods"][method] = "complete"
                    continue
            cfg = load_toml_config(CONFIGS[method])
            print(f"[{method}] starting smoke={args.smoke}", flush=True)
            train_method(cfg, phase_root, args.smoke, not args.no_pretrained)
            state["methods"][method] = "smoke_passed" if args.smoke else "complete"
            save_json(status_path, state)
        if not args.smoke and all((phase_root / m / "summary.json").exists() for m in CONFIGS):
            state["analysis"] = phase2_analysis(phase_root)
            state["status"] = "complete"
        elif args.smoke:
            state["status"] = "smoke_passed"
        save_json(status_path, state)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        save_json(status_path, state)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
