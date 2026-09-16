"""Evaluate completed single-dataset checkpoints on the opposite dataset test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpair.benchmark.aor_dr import AORDR
from drpair.benchmark.aor_training import evaluate_aor
from drpair.benchmark.quick_training import (
    build_model,
    evaluate,
    load_toml_config,
    make_loader,
    read_rows,
)
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

RUN_ROOTS = {
    "EyePACS": {
        "B0": ROOT / "results/runs/phase2_quick20_symmetry_preserved_v2",
        "B1-S": ROOT / "results/runs/phase2_quick20_symmetry_preserved_v2",
        "B1-P": ROOT / "results/runs/phase2_quick20_symmetry_preserved_v2",
        "B2-S": ROOT / "results/runs/phase3_quick20_symmetry_preserved_v2",
        "B2-P": ROOT / "results/runs/phase3_quick20_symmetry_preserved_v2",
        "B3": ROOT / "results/runs/phase4_quick20_symmetry_preserved_v2",
        "B4": ROOT / "results/runs/phase4_quick20_symmetry_preserved_v2",
    },
    "Messidor-2": {method: ROOT / "results/runs/messidor_pilot" for method in CONFIGS},
}

DATASETS = {
    "EyePACS": {
        "test_csv": ROOT / "data/splits/eyepacs_quick20_symmetry_test.csv",
        "cache_root": ROOT / "data/cache/quick20_huang_512",
    },
    "Messidor-2": {
        "test_csv": ROOT / "data/splits/messidor_test.csv",
        "cache_root": ROOT / "data/cache/messidor_huang_512",
    },
}


def evaluate_transfer(source: str, target: str, method: str) -> dict:
    cfg = load_toml_config(CONFIGS[method])
    cfg.update({
        "test_csv": str(DATASETS[target]["test_csv"]),
        "cache_root": str(DATASETS[target]["cache_root"]),
        "dataset_name": target,
    })
    source_root = RUN_ROOTS[source][method]
    checkpoint = torch.load(source_root / method / "best.pt", map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == "B4":
        model = AORDR(
            pretrained=True,
            diffusion_batch_mul=cfg["diffusion_batch_mul"],
            sampling_steps=str(cfg["sampling_steps"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        metrics, _, _, _ = evaluate_aor(model, make_loader(read_rows(cfg["test_csv"]), cfg, False, False), device, cfg["seed"] + 20_000)
    else:
        model, paired = build_model(cfg, pretrained=method == "B3", phase_root=source_root)
        model.to(device)
        model.load_state_dict(checkpoint["model_state"])
        metrics, _, _, _ = evaluate(model, make_loader(read_rows(cfg["test_csv"]), cfg, False, paired), cfg, device)
    return {"source": source, "target": target, "method": method, "test": metrics}


def main() -> None:
    output = ROOT / "results/transfer_pilot_metrics.json"
    status = ROOT / "results/transfer_pilot_status.json"
    records = []
    save_json(status, {"status": "running", "completed": 0, "total": 14})
    for source, target in (("EyePACS", "Messidor-2"), ("Messidor-2", "EyePACS")):
        for method in CONFIGS:
            print(f"[{source} -> {target}] {method}", flush=True)
            records.append(evaluate_transfer(source, target, method))
            save_json(output, records)
            save_json(status, {"status": "running", "completed": len(records), "total": 14})
    save_json(status, {"status": "complete", "completed": len(records), "total": 14})


if __name__ == "__main__":
    main()
