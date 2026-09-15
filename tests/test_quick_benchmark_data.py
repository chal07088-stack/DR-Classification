from collections import Counter

from drpair.benchmark.eyepacs import PatientPair, assign_patient_splits


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


def test_quick20_seed_changes_membership_without_changing_counts():
    pairs = _pairs()
    first, targets = assign_patient_splits(pairs, seed=42)
    second, _ = assign_patient_splits(pairs, seed=43)
    assert first != second
    assert Counter(first.values()) == Counter(second.values()) == Counter(targets)

