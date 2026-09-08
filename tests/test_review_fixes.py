import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from drpair.data.manifest import content_audit_status, patient_folds

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from boundary_power import boundary_power


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


def test_boundary_power_reports_no_headroom_when_experts_agree():
    y = np.array([0, 1, 2, 3, 4] * 4)
    q = np.tile(np.array([0.9, 0.7, 0.3, 0.1], np.float32), (len(y), 1))
    rows, ceiling = boundary_power(y, q, q.copy())
    assert [row["expert_disagreement"] for row in rows] == [0, 0, 0, 0]
    assert not any(row["actionable"] for row in rows)
    assert ceiling["oracle_headroom_over_paired"] == 0.0
    assert ceiling["eyes_oracle_touches"] == 0


def test_boundary_power_finds_the_actionable_boundaries():
    y = np.array([0, 4] * 10)
    qs = np.array(
        [[0.2, 0.1, 0.05, 0.02] if v == 0 else [0.95, 0.9, 0.85, 0.8] for v in y], np.float32
    )
    qf = np.tile(np.array([0.6, 0.55, 0.4, 0.3], np.float32), (len(y), 1))
    rows, ceiling = boundary_power(y, qs, qf)
    assert any(row["actionable"] for row in rows)
    assert ceiling["qwk_single"] == 1.0
    assert ceiling["oracle_headroom_over_paired"] > 0
    assert ceiling["eyes_oracle_touches"] == len(y)
