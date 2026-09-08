import copy

import numpy as np
import pytest

from drpair.data.manifest import patient_folds, validate_manifest
from drpair.gates.features import fellow_predictions, gate_features


def pair_rows(n=20):
    rows = []
    for patient in range(n):
        for side, other in [("left", "right"), ("right", "left")]:
            rows.append(
                {
                    "patient": str(patient),
                    "side": side,
                    "label": patient % 5,
                    "fellow_label": patient % 5,
                    "split": "train",
                    "path": f"train/{patient}_{side}.jpeg",
                    "fellow_path": f"train/{patient}_{other}.jpeg",
                }
            )
    return rows


def test_split_assertions_and_patient_folds():
    rows = pair_rows()
    validate_manifest(rows)
    folds = patient_folds(rows)
    assert all(folds[i] == folds[i + 1] for i in range(0, len(rows), 2))
    assert set(folds) == set(range(5))
    np.testing.assert_array_equal(folds, patient_folds(rows))
    bad = copy.deepcopy(rows)
    bad[0]["split"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(bad)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(rows + [rows[0]])
    with pytest.raises(ValueError, match="Unpaired"):
        validate_manifest(rows[:-1])


def test_features_only_accept_predictions_and_fellow_identity():
    q = np.array([[0.9, 0.8, 0.3, 0.1], [0.8, 0.5, 0.2, 0.1]], np.float32)
    fellow = fellow_predictions(q, ["1", "1"], ["left", "right"])
    np.testing.assert_array_equal(fellow, q[::-1])
    features, names = gate_features(q, q, fellow)
    assert features.shape == (2, len(names))
    assert not any("true" in n or "label" in n or "gap" in n for n in names)
    assert features[0, -2] == -features[1, -2]
    with pytest.raises(KeyError):
        fellow_predictions(q[:1], ["1"], ["left"])
