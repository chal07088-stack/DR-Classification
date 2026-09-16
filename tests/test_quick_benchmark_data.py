import hashlib
import json
from collections import Counter

import pytest

from drpair.benchmark.eyepacs import (
    ACTIVE_PATIENT_FILE,
    ACTIVE_SPLIT_FILES,
    SPLIT_VERSION,
    PatientPair,
    assign_patient_splits,
    require_phase1_hard_gate,
)


def _pairs():
    strata = {
        (0, 0): 12155,
        (1, 0): 600,
        (1, 1): 842,
        (2, 0): 1998,
        (2, 1): 393,
        (2, 2): 631,
        (3, 0): 307,
        (3, 1): 183,
        (3, 2): 9,
        (4, 0): 263,
        (4, 1): 67,
        (4, 2): 115,
    }
    result = []
    patient = 1
    for (maximum, bucket), count in strata.items():
        if bucket == 0:
            left = right = maximum
        elif bucket == 1:
            left, right = maximum, maximum - 1
        else:
            left, right = maximum, max(0, maximum - 2)
        for _ in range(count):
            result.append(
                PatientPair(
                    str(patient), left, right, f"{patient}_left", f"{patient}_right"
                )
            )
            patient += 1
    return result


def test_quick20_assignment_is_exact_disjoint_and_deterministic():
    pairs = _pairs()
    first, targets = assign_patient_splits(pairs, seed=42)
    second, second_targets = assign_patient_splits(pairs, seed=42)
    assert first == second
    assert targets == second_targets == {"train": 1756, "val": 527, "test": 1229}
    assert Counter(first.values()) == Counter(targets)
    assert len(first) == len(set(first)) == 3512
    by_id = {pair.patient_id: pair for pair in pairs}
    full_symmetric_rate = sum(pair.gap == 0 for pair in pairs) / len(pairs)
    for split, total in targets.items():
        selected = [by_id[patient] for patient, value in first.items() if value == split]
        symmetric_rate = sum(pair.gap == 0 for pair in selected) / total
        assert abs(symmetric_rate - full_symmetric_rate) <= 1 / total
    quick = [by_id[patient] for patient in first]
    assert abs(sum(pair.gap == 0 for pair in quick) / len(quick) - full_symmetric_rate) <= 1 / len(quick)


def test_quick20_seed_changes_membership_without_changing_counts():
    pairs = _pairs()
    first, targets = assign_patient_splits(pairs, seed=42)
    second, _ = assign_patient_splits(pairs, seed=43)
    assert first != second
    assert Counter(first.values()) == Counter(second.values()) == Counter(targets)


def test_training_gate_rejects_blocked_messidor(tmp_path):
    audit_root = tmp_path / "results" / "data_audit"
    audit_root.mkdir(parents=True)
    inventory = {}
    for key, relative in {**ACTIVE_SPLIT_FILES, "patients": ACTIVE_PATIENT_FILE}.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
        inventory[key] = {
            "path": relative,
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
        }
    (audit_root / "eyepacs_symmetry_sampling_summary.json").write_text(
        json.dumps({
            "split_version": SPLIT_VERSION,
            "status": "passed",
            "hard_gate_passed": True,
            "active_split_files": inventory,
        }),
        encoding="utf-8",
    )
    (audit_root / "messidor_pair_join_summary.json").write_text(
        json.dumps({"status": "blocked", "reason": "official pairing missing"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="official pairing missing"):
        require_phase1_hard_gate(tmp_path)
