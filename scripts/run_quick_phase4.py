"""Run CLIP-DR and AOR-DR only after Phase 3 completion."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpair.benchmark.aor_training import train_aor
from drpair.benchmark.eyepacs import SPLIT_VERSION, require_phase1_hard_gate
from drpair.benchmark.quick_training import load_toml_config, train_method
from drpair.runtime import save_json

CONFIGS = {
    "B3": ROOT / "configs/benchmark/clip_dr.toml",
    "B4": ROOT / "configs/benchmark/aor_dr.toml",
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
    phase3 = json.loads(
        (ROOT / f"results/phase3_{SPLIT_VERSION}_status.json").read_text(encoding="utf-8")
    )
    if not args.smoke and phase3.get("status") != "complete":
        raise RuntimeError("Phase 4 hard gate requires Phase 3 status=complete")
    phase_root = ROOT / f"results/runs/phase4_{SPLIT_VERSION}"
    status_path = ROOT / f"results/phase4_{SPLIT_VERSION}_status.json"
    methods = [args.method] if args.method else list(CONFIGS)
    state = {
        "phase": 4,
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
                    state["methods"][method] = "complete"
                    continue
            cfg = load_toml_config(CONFIGS[method])
            print(f"[{method}] starting smoke={args.smoke}", flush=True)
            if method == "B3":
                train_method(cfg, phase_root, args.smoke, not args.no_pretrained)
            else:
                train_aor(cfg, phase_root, args.smoke, not args.no_pretrained)
            state["methods"][method] = "smoke_passed" if args.smoke else "complete"
            save_json(status_path, state)
        state["status"] = "smoke_passed" if args.smoke else "complete"
        save_json(status_path, state)
        if not args.smoke:
            (ROOT / "results/PHASE4_ORDINAL.md").write_text(
                "# PHASE 4 — Ordinal single-eye comparators\n\n"
                "B3 CLIP-DR and B4 AOR-DR completed under the locked capped protocol.\n",
                encoding="utf-8",
            )
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        save_json(status_path, state)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
