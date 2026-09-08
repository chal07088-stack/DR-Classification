from pathlib import Path

import numpy as np

from drpair.data.manifest import prepare, read_manifest
from drpair.evaluation.ordinal import project
from drpair.evaluation.report import make_report
from drpair.experiments.gate_study import run_gate_study
from drpair.runtime import load_config, run_directory
from drpair.training.predictions import save_predictions


def test_cached_prediction_workflow_cpu_only(tmp_path):
    """Synthetic files test wiring; no EyePACS images or image models are loaded."""
    cfg = load_config(Path(__file__).parents[1] / "configs" / "phase1.toml")
    cfg["data"] = {
        "root": str(tmp_path / "images"),
        "manifest": str(tmp_path / "manifest.csv"),
        "fold_manifest": str(tmp_path / "folds.csv"),
        "results": str(tmp_path / "results"),
    }
    cfg["gate"]["epochs"] = 1
    cfg["gate"]["batch"] = 128
    cfg["evaluation"]["bootstrap_repeats"] = 5
    for s, split in enumerate(("train", "val", "test")):
        for p in range(15):
            for e, side in enumerate(("left", "right")):
                path = tmp_path / "images" / split / str((p + e) % 5) / f"{s * 100 + p}_{side}.jpeg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
    prepare(
        cfg["data"]["root"],
        cfg["data"]["manifest"],
        cfg["data"]["fold_manifest"],
        tmp_path / "audit.json",
        official=False,
    )
    all_rows = read_manifest(cfg["data"]["manifest"])
    run = run_directory(cfg)
    rng = np.random.default_rng(42)
    for split, prefix in [("train", "oof"), ("val", "final"), ("test", "final")]:
        rows = [r for r in all_rows if r["split"] == split]
        for kind in ("single", "paired"):
            name = f"oof_{kind}.npz" if split == "train" else f"final_{kind}_{split}.npz"
            q = project(rng.random((len(rows), 4)))
            save_predictions(run / "predictions" / name, rows, q)
    run_gate_study(cfg)
    report = make_report(cfg)
    assert report.exists()
    assert (run / "metrics" / "local_vs_joint.csv").exists()
    assert (run / "figures" / "qwk_by_gap.png").exists()
    with np.load(run / "teachers" / "val.npz") as teachers:
        assert "b_utility" in teachers.files
        assert not any("joint" in key or "oracle" in key for key in teachers.files)
