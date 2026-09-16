"""Dataset-balanced EyePACS + Messidor-2 joint pilot for B0/B1/B2 only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpair.benchmark.eyepacs import require_phase1_hard_gate
from drpair.benchmark.quick_training import load_toml_config, train_method
from drpair.runtime import save_json

CONFIGS = {
    "B0": ROOT / "configs/benchmark/convnext_single.toml",
    "B1-S": ROOT / "configs/benchmark/huang_single.toml",
    "B1-P": ROOT / "configs/benchmark/huang_paired.toml",
    "B2-S": ROOT / "configs/benchmark/drgcnn_single.toml",
    "B2-P": ROOT / "configs/benchmark/drgcnn_bilateral.toml",
}


def combined_split(name: str) -> Path:
    sources = [
        (ROOT / f"data/splits/eyepacs_quick20_symmetry_{name}.csv", "EyePACS"),
        (ROOT / f"data/splits/messidor_{name}.csv", "Messidor-2"),
    ]
    rows = []
    for source, dataset in sources:
        with source.open(encoding="utf-8", newline="") as stream:
            rows.extend({**row, "dataset": dataset} for row in csv.DictReader(stream))
    target = ROOT / f"data/splits/joint_eyepacs_messidor_{name}.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(CONFIGS))
    args = parser.parse_args()
    require_phase1_hard_gate(ROOT)
    splits = {name: combined_split(name) for name in ("train", "val", "test")}
    root = ROOT / "results/runs/joint_eyepacs_messidor_pilot"
    status_path = ROOT / "results/joint_eyepacs_messidor_pilot_status.json"
    methods = [args.method] if args.method else list(CONFIGS)
    state = {"phase": "dataset-balanced joint pilot", "status": "running", "methods": {}}
    save_json(status_path, state)
    try:
        for method in methods:
            summary_path = root / method / "summary.json"
            if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "complete":
                state["methods"][method] = "complete"
                continue
            cfg = load_toml_config(CONFIGS[method])
            cfg.update({
                "train_csv": str(splits["train"]), "val_csv": str(splits["val"]),
                "test_csv": str(splits["test"]),
                "cache_root": str(ROOT / "data/cache/quick20_huang_512"),
                "cache_roots": [
                    str(ROOT / "data/cache/quick20_huang_512"),
                    str(ROOT / "data/cache/messidor_huang_512"),
                ],
                "dataset_name": "Joint EyePACS+Messidor-2", "dataset_balanced": True,
            })
            print(f"[{method}] starting dataset-balanced joint training", flush=True)
            train_method(cfg, root, False, True)
            state["methods"][method] = "complete"
            save_json(status_path, state)
        state["status"] = "complete"
        save_json(status_path, state)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        save_json(status_path, state)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
