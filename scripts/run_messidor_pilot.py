"""Run the audited Messidor-2 single-dataset comparator block serially."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpair.benchmark.aor_training import train_aor
from drpair.benchmark.eyepacs import require_phase1_hard_gate
from drpair.benchmark.quick_training import load_toml_config, train_method
from drpair.runtime import save_json

CONFIGS = {
    "B0": ROOT / "configs/benchmark/convnext_single.toml",
    "B1-S": ROOT / "configs/benchmark/huang_single.toml",
    "B1-P": ROOT / "configs/benchmark/huang_paired.toml",
    "B2-S": ROOT / "configs/benchmark/drgcnn_single.toml",
    "B2-P": ROOT / "configs/benchmark/drgcnn_bilateral.toml",
    "B3": ROOT / "configs/benchmark/clip_dr.toml",
    "B4": ROOT / "configs/benchmark/aor_dr.toml",
}


def messidor_config(method: str) -> dict:
    cfg = load_toml_config(CONFIGS[method])
    cfg.update(
        {
            "train_csv": str(ROOT / "data/splits/messidor_train.csv"),
            "val_csv": str(ROOT / "data/splits/messidor_val.csv"),
            "test_csv": str(ROOT / "data/splits/messidor_test.csv"),
            "cache_root": str(ROOT / "data/cache/messidor_huang_512"),
            "dataset_name": "Messidor-2",
        }
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(CONFIGS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    gate = require_phase1_hard_gate(ROOT)
    phase_root = ROOT / "results/runs/messidor_pilot"
    status_path = ROOT / "results/messidor_pilot_status.json"
    methods = [args.method] if args.method else list(CONFIGS)
    state = {
        "phase": "Messidor-2 single-dataset pilot",
        "scope": gate.get("scope", "DUAL_DATASET_BENCHMARK"),
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
                if json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "complete":
                    state["methods"][method] = "complete"
                    save_json(status_path, state)
                    continue
            print(f"[{method}] starting Messidor-2 smoke={args.smoke}", flush=True)
            cfg = messidor_config(method)
            if method == "B4":
                train_aor(cfg, phase_root, args.smoke, True)
            else:
                train_method(cfg, phase_root, args.smoke, True)
            state["methods"][method] = "smoke_passed" if args.smoke else "complete"
            save_json(status_path, state)
        state["status"] = "smoke_passed" if args.smoke else "complete"
        save_json(status_path, state)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        save_json(status_path, state)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
