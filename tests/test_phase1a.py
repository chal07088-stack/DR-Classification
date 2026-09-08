import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import isotonic_regression

from drpair.data.manifest import content_audit_status, write_csv
from drpair.evaluation.calibration import crossfit_calibration, specificity_threshold
from drpair.evaluation.feasibility import make_feasibility_report
from drpair.evaluation.headroom import rollback_grid
from drpair.evaluation.ordinal import blend, decode, project
from drpair.experiments.stages import execute_stage, require_review
from drpair.runtime import file_hash, load_config, run_directory
from drpair.teachers.targets import discrete_oracle
from drpair.training.predictions import save_predictions


def test_vectorized_projection_matches_independent_pava():
    rng = np.random.default_rng(42)
    q = np.vstack([rng.random((1000, 4)), [[0.5, 0.1, 0.9, 0.5], [0, 1, 0, 1]]])
    expected = np.array([isotonic_regression(row, increasing=False).x for row in q])
    np.testing.assert_allclose(project(q), expected, atol=1e-7)
    np.testing.assert_array_equal(project(project(q)), project(q))


def test_grid_minimizes_squared_error_in_declared_policy_class():
    rng = np.random.default_rng(4)
    qs, qf = project(rng.random((40, 4))), project(rng.random((40, 4)))
    y = rng.integers(0, 5, 40)
    result = rollback_grid(qs, qf, y, [0, 0.5, 1])
    errors = []
    for mask in itertools.product([0, 0.5, 1], repeat=4):
        q, _ = blend(qs, qf, np.broadcast_to(mask, qs.shape))
        errors.append((decode(q) - y) ** 2)
    np.testing.assert_array_equal((decode(result["q"]) - y) ** 2, np.min(errors, axis=0))
    binary = discrete_oracle(qs, qf, y)
    assert np.all(result["utility"] >= binary["utility"])
    with pytest.raises(ValueError):
        rollback_grid(qs, qf, y, [0, 0.5])


@pytest.mark.parametrize("method", ["raw", "platt", "isotonic"])
def test_heldout_labels_do_not_change_calibration_or_operating_points(method):
    rng = np.random.default_rng(5)
    patient = np.repeat(np.arange(40), 2)
    folds = patient % 5
    q = project(rng.random((80, 4)))
    y = rng.integers(0, 5, 80)
    out, thresholds, records = crossfit_calibration(q, y, patient, folds, method, [0.9, 0.95])
    changed = y.copy()
    changed[folds == 0] = 4 - changed[folds == 0]
    other, other_thresholds, _ = crossfit_calibration(
        q, changed, patient, folds, method, [0.9, 0.95]
    )
    np.testing.assert_array_equal(out[folds == 0], other[folds == 0])
    np.testing.assert_array_equal(thresholds[folds == 0], other_thresholds[folds == 0])
    assert np.all(np.diff(out, axis=1) <= 1e-7)
    for record in records:
        assert not set(record["training_patients"]) & set(record["heldout_patients"])
    broken = folds.copy()
    broken[0] = 1
    with pytest.raises(ValueError, match="patient leakage"):
        crossfit_calibration(q, y, patient, broken, method, [0.95])


def test_specificity_ties_and_missing_negatives():
    scores = np.array([0.1, 0.2, 0.2, 0.9])
    threshold = specificity_threshold(np.zeros(4, bool), scores, 0.75)
    assert np.mean(scores < threshold) >= 0.75
    assert specificity_threshold(np.ones(4, bool), scores, 0.95) is None


def test_manifest_audit_is_bound_to_content(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("first", encoding="utf-8")
    report = tmp_path / "experimental_results" / "repository_validation" / "data_content_audit.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"manifest_sha256": file_hash(manifest)}), encoding="utf-8")
    assert content_audit_status(tmp_path, manifest)["image_content_duplicates_checked"]
    manifest.write_text("changed", encoding="utf-8")
    assert not content_audit_status(tmp_path, manifest)["image_content_duplicates_checked"]


def test_stage1a_never_calls_oof_gates_or_test(monkeypatch, tmp_path):
    from drpair.evaluation import feasibility
    from drpair.experiments import experts, gate_study

    calls = []

    def fake_run(cfg, jobs, **kwargs):
        calls.append((jobs, kwargs))

    def forbidden(*args, **kwargs):
        raise AssertionError("PHASE 1A attempted a test/gate operation")

    monkeypatch.setattr(experts, "run_experts", fake_run)
    monkeypatch.setattr(experts, "predict_final_experts", forbidden)
    monkeypatch.setattr(gate_study, "run_gate_study", forbidden)
    monkeypatch.setattr(feasibility, "make_feasibility_report", lambda cfg: tmp_path / "REPORT.md")
    states = []
    execute_stage({}, "1a", update=lambda *args, **kw: states.append((args, kw)))
    assert calls == [(["final_single", "final_paired"], {"evaluation_splits": ("val",)})]
    assert states[-1][0] == ("review_boundary", "awaiting_review")
    with pytest.raises(ValueError, match="explicit"):
        execute_stage({}, "1b")


def test_validation_report_and_review_guard_end_to_end(tmp_path, monkeypatch):
    from drpair.evaluation import feasibility

    cfg = load_config(Path(__file__).parents[1] / "configs" / "phase1.toml")
    cfg["data"].update(
        results=str(tmp_path / "results"),
        manifest=str(tmp_path / "manifest.csv"),
        fold_manifest=str(tmp_path / "folds.csv"),
    )
    cfg["diagnostics"].update(alpha_grid=[0, 0.5, 1], bootstrap_repeats=10)
    rows = []
    for p in range(50):
        for side, fellow in [("left", "right"), ("right", "left")]:
            label = p % 5 if side == "left" else (p // 5) % 5
            other = (p // 5) % 5 if side == "left" else p % 5
            rows.append(
                {
                    "patient": str(p),
                    "side": side,
                    "label": label,
                    "fellow_label": other,
                    "path": f"{p}_{side}.jpeg",
                    "fellow_path": f"{p}_{fellow}.jpeg",
                    "split": "val",
                }
            )
    write_csv(cfg["data"]["manifest"], rows)
    write_csv(
        cfg["data"]["fold_manifest"],
        [
            {
                "patient": r["patient"],
                "side": r["side"],
                "split": "val",
                "fold": (int(r["patient"]) // 10),
                "seed": 42,
            }
            for r in rows
        ],
    )
    run = run_directory(cfg)
    rng = np.random.default_rng(42)
    for kind in ("single", "paired"):
        save_predictions(
            run / "predictions" / f"final_{kind}_val.npz",
            rows,
            project(rng.random((100, 4))),
            seed=42,
        )
    original_loader = feasibility.load_pair

    def val_only(single, paired):
        assert "test" not in Path(single).name and "test" not in Path(paired).name
        return original_loader(single, paired)

    monkeypatch.setattr(feasibility, "load_pair", val_only)
    report = make_feasibility_report(cfg)
    assert report.is_file() and (report.parent / "grade_metrics.png").stat().st_size > 1000
    summary = require_review(cfg, True)
    assert summary["status"] == "awaiting_review" and not summary["test_evaluated"]
    assert not list((run / "predictions").glob("*test*"))
    assert (report.parent / "clinical_sensitivity_deltas.csv").is_file()
    with pytest.raises(ValueError, match="explicit"):
        require_review(cfg, False)
