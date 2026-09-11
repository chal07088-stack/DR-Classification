from pathlib import Path

import numpy as np

from drpair.data.manifest import prepare, read_manifest, write_csv
from drpair.evaluation.multiseed import make_multiseed_report
from drpair.evaluation.ordinal import project
from drpair.evaluation.report import make_report
from drpair.experiments.arms import ARMS
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
    assert (run / "metrics" / "oof_boundary_opportunities.csv").exists()
    with np.load(run / "features" / "gate_crossfit_identity.npz") as identity:
        patients = identity["rows"][:, 0].astype(int)
        assert ((patients >= 100) & (patients < 200)).all()
    assert (run / "figures" / "qwk_by_gap.png").exists()
    with np.load(run / "teachers" / "val.npz") as teachers:
        assert "b_utility" in teachers.files
        assert not any("joint" in key or "oracle" in key for key in teachers.files)


def test_phase2_gate_workflow_does_not_require_train_oof(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs" / "phase2_seed123.toml")
    cfg["data"] = {
        "root": str(tmp_path / "images"),
        "manifest": str(tmp_path / "manifest.csv"),
        "fold_manifest": str(tmp_path / "folds.csv"),
        "results": str(tmp_path / "results"),
    }
    cfg["gate"]["epochs"] = 1
    cfg["gate"]["batch"] = 128
    cfg["evaluation"]["bootstrap_repeats"] = 5
    for split_index, split in enumerate(("train", "val", "test")):
        for patient in range(15):
            for eye, side in enumerate(("left", "right")):
                path = (
                    tmp_path
                    / "images"
                    / split
                    / str((patient + eye) % 5)
                    / f"{split_index * 100 + patient}_{side}.jpeg"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
    prepare(
        cfg["data"]["root"],
        cfg["data"]["manifest"],
        cfg["data"]["fold_manifest"],
        tmp_path / "audit.json",
        seed=123,
        official=False,
    )
    rows = read_manifest(cfg["data"]["manifest"])
    run = run_directory(cfg)
    rng = np.random.default_rng(123)
    for split in ("val", "test"):
        subset = [row for row in rows if row["split"] == split]
        for kind in ("single", "paired"):
            save_predictions(
                run / "predictions" / f"final_{kind}_{split}.npz",
                subset,
                project(rng.random((len(subset), 4))),
                seed=123,
            )
    run_gate_study(cfg, include_expert_oof=False)
    report = make_report(cfg)
    assert report.exists()
    assert not (run / "teachers" / "oof.npz").exists()
    assert '"phase": 2' in (run / "completion.json").read_text(encoding="utf-8")


def test_multiseed_aggregation(tmp_path):
    runs = []
    for index, seed in enumerate((42, 123, 128)):
        run_name = f"run_{seed}"
        runs.append((seed, run_name))
        rows = []
        for arm_index, arm in enumerate(ARMS):
            for group in ("overall", "gap_ge1", "gap_ge2"):
                rows.append(
                    {
                        "arm": arm,
                        "split": "test",
                        "group": group,
                        "qwk": 0.8 + index * 0.01 + arm_index * 0.001,
                    }
                )
        write_csv(tmp_path / run_name / "metrics" / "arm_metrics.csv", rows)
    report = make_multiseed_report(tmp_path, runs)
    assert report.exists()
    summary = (report.parent / "metrics" / "multiseed_summary.csv").read_text(encoding="utf-8")
    assert "G_teacher_b" in summary
    assert "0.81" in summary
