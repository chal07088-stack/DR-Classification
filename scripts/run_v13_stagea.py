"""Stage A launcher for the isolated Messidor v13 internal pilot."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpair.benchmark.quick_training import load_toml_config, train_method
from drpair.runtime import save_json

OUT = ROOT / "results/messidor_full_internal_pilot_v13_seed42"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/splits/messidor_train.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    patient_ids = sorted({row["patient_id"] for row in rows})
    expert_ids = {
        patient_id for patient_id in patient_ids
        if int(hashlib.sha256(f"42:{patient_id}".encode()).hexdigest(), 16) % 10 < 7
    }
    if not expert_ids or len(expert_ids) == len(patient_ids):
        raise RuntimeError("Invalid deterministic Expert-Train/Correction-Dev assignment")
    expert_rows = [row for row in rows if row["patient_id"] in expert_ids]
    correction_rows = [row for row in rows if row["patient_id"] not in expert_ids]
    split_dir = OUT / "01_splits"
    split_dir.mkdir(exist_ok=True)
    for name, selected in (("expert_train.csv", expert_rows), ("correction_dev.csv", correction_rows)):
        with (split_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(selected)
    cfg = load_toml_config(ROOT / "configs/benchmark/convnext_single.toml")
    cfg.update({
        "train_csv": str(split_dir / "expert_train.csv"),
        "val_csv": str(ROOT / "data/splits/messidor_val.csv"),
        "test_csv": str(ROOT / "data/splits/messidor_test.csv"),
        "cache_root": str(ROOT / "data/cache/messidor_huang_512"),
        "dataset_name": "Messidor-2 v13 Stage A", "max_epochs": 50, "patience": 10,
    })
    save_json(OUT / "00_config/stageA_split.json", {"seed": 42, "expert_patients": len(expert_ids), "correction_dev_patients": len(patient_ids) - len(expert_ids)})
    train_method(cfg, OUT / "02_stageA", False, True)


if __name__ == "__main__":
    main()
