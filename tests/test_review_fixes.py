import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from drpair.data.manifest import content_audit_status, patient_folds
from drpair.evaluation.ordinal import project

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from boundary_power import boundary_power, grid_search, project_rows


def rows_with_labels(labels):
    rows = []
    for patient, label in enumerate(labels):
        for side, other in [("left", "right"), ("right", "left")]:
            rows.append(
                {
                    "patient": str(patient),
                    "side": side,
                    "label": label,
                    "fellow_label": label,
                    "split": "train",
                    "path": f"train/{patient}_{side}.jpeg",
                    "fellow_path": f"train/{patient}_{other}.jpeg",
                }
            )
    return rows


def test_unstratified_fold_fallback_is_reported_not_silent():
    # Grade 0 alone clears the fold count; the merged rare stratum holds only four.
    rows = rows_with_labels([0] * 10 + [1, 2, 3, 4])
    with pytest.warns(RuntimeWarning, match="unstratified"):
        folds = patient_folds(rows, folds=5, seed=42)
    assert set(folds) == set(range(5))
    assert all(folds[i] == folds[i + 1] for i in range(0, len(rows), 2))


def test_stratified_folds_stay_silent():
    rows = rows_with_labels([g for g in range(5) for _ in range(6)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        folds = patient_folds(rows, folds=5, seed=42)
    assert set(folds) == set(range(5))


def test_content_audit_status_reflects_the_committed_audit(tmp_path):
    assert content_audit_status(tmp_path) == {"image_content_duplicates_checked": False}
    audit = tmp_path / "experimental_results" / "repository_validation" / "data_content_audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(
        json.dumps(
            {
                "images": 88702,
                "unique_content_hashes": 88702,
                "byte_duplicate_groups": 0,
                "cross_split_duplicate_groups": 0,
                "conflicting_label_groups": 0,
                "near_duplicate_images_checked": False,
            }
        ),
        encoding="utf-8",
    )
    status = content_audit_status(tmp_path)
    assert status["image_content_duplicates_checked"] is True
    assert status["image_content_audit"]["byte_duplicate_groups"] == 0
    assert status["image_content_audit"]["source"].endswith("data_content_audit.json")


def test_vectorised_projection_matches_the_reference():
    rng = np.random.default_rng(0)
    q = rng.random((512, 4)).astype(np.float32)
    np.testing.assert_allclose(project_rows(q), project(q), atol=1e-6)
    monotone = project(rng.random((64, 4)).astype(np.float32))
    np.testing.assert_allclose(project_rows(monotone), monotone, atol=1e-6)


def test_identical_experts_leave_no_headroom():
    y = np.array([0, 1, 2, 3, 4] * 4)
    q = np.tile(np.array([0.9, 0.7, 0.3, 0.1], np.float32), (len(y), 1))
    rows, summary = boundary_power(y, q, q.copy())
    assert [row["threshold_crossing"] for row in rows] == [0, 0, 0, 0]
    assert [row["probability_differs"] for row in rows] == [0, 0, 0, 0]
    assert summary["eyes_touched_binary"] == 0
    assert summary["eyes_touched_grid"] == 0
    assert summary["qwk_gap_binary_vs_paired"] == 0.0
    assert summary["qwk_gap_grid_vs_paired"] == 0.0


def test_grid_is_never_worse_than_the_binary_masks():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 5, 200)
    qs = project(rng.random((200, 4)).astype(np.float32))
    qf = project(rng.random((200, 4)).astype(np.float32))
    _, summary = boundary_power(y, qs, qf)
    # The grid contains {0,1}^4, so its per-eye error can only match or improve.
    assert summary["squared_error_grid"] <= summary["squared_error_binary"] + 1e-9
    assert summary["eyes_grid_beats_binary"] >= 0


def test_grid_can_select_intermediate_alpha():
    # The grid is strictly richer than the corners: PAVA can make a half blend
    # decode differently from both endpoints. It is not guaranteed to help, so
    # this only asserts the search can return a non-binary alpha at all.
    y = np.array([2, 2, 2, 2])
    qs = np.tile(np.array([0.95, 0.90, 0.05, 0.00], np.float32), (4, 1))
    qf = np.tile(np.array([0.55, 0.10, 0.45, 0.40], np.float32), (4, 1))
    grid = grid_search(qs, qf, y, levels=(0.0, 0.5, 1.0))
    assert grid["alpha"].shape == qs.shape
    assert set(np.unique(grid["alpha"])) <= {0.0, 0.5, 1.0}
    assert grid["error"].max() <= float(((project(qf) >= 0.5).sum(1) - y).max() ** 2)


def test_boundary_power_separates_probability_and_threshold_disagreement():
    y = np.array([0, 4] * 10)
    qs = np.array(
        [[0.2, 0.1, 0.05, 0.02] if v == 0 else [0.95, 0.9, 0.85, 0.8] for v in y], np.float32
    )
    qf = np.tile(np.array([0.6, 0.55, 0.4, 0.3], np.float32), (len(y), 1))
    rows, summary = boundary_power(y, qs, qf)
    assert all(row["probability_differs"] == len(y) for row in rows)
    assert any(row["threshold_crossing"] > 0 for row in rows)
    assert summary["qwk_single"] == 1.0
    assert summary["qwk_gap_grid_vs_paired"] > 0
    assert summary["eyes_touched_grid"] == len(y)
