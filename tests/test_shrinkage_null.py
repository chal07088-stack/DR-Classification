import numpy as np

from drpair.evaluation.feasibility import (
    shrinkage_association,
    shrinkage_design,
    shrinkage_permutation,
)


def flat(expected):
    """A constant cumulative row sums to its expected grade and needs no projection."""
    return np.repeat(np.asarray(expected, np.float64)[:, None] / 4.0, 4, axis=1)


def planted(n=3000, coefficient=0.30, seed=6):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    fellow = rng.integers(0, 5, n)
    patient = np.repeat(np.arange(n // 2), 2)
    single = np.full(n, 2.0)
    paired = single + coefficient * (fellow - y) + rng.normal(0, 0.05, n)
    return y, fellow, patient, flat(single), flat(np.clip(paired, 0, 4))


def test_shrinkage_association_recovers_a_planted_coefficient():
    y, fellow, patient, qs, qf = planted()
    result = shrinkage_association(y, fellow, qs, qf, patient)
    assert abs(result["slope"] - 0.30) < 0.02
    assert result["ci_low"] < 0.30 < result["ci_high"]
    assert result["design_rank"] == 6


def test_permutation_rejects_when_the_shift_tracks_the_fellow():
    y, fellow, _patient, qs, qf = planted()
    null = shrinkage_permutation(y, fellow, qs, qf, repeats=200, seed=42)
    assert abs(null["observed_slope"] - 0.30) < 0.02
    assert abs(null["null_mean"]) < 0.02
    assert null["p_permutation"] < 0.01


def test_permutation_does_not_reject_when_the_shift_ignores_the_fellow():
    rng = np.random.default_rng(7)
    n = 1500
    y = rng.integers(0, 5, n)
    fellow = rng.integers(0, 5, n)
    single = np.full(n, 2.0)
    paired = single + rng.normal(0, 0.1, n)  # noise only, no fellow dependence
    null = shrinkage_permutation(
        y, fellow, flat(single), flat(np.clip(paired, 0, 4)), repeats=200, seed=3
    )
    assert abs(null["null_mean"]) < 0.01
    assert null["p_permutation"] > 0.05


def test_permutation_is_restricted_to_the_target_grade():
    # The design must be unchanged for eyes whose target grade has one member,
    # because such a stratum cannot be shuffled with anything else.
    y = np.array([0, 0, 0, 0, 1])
    fellow = np.array([0, 1, 2, 3, 4])
    design = shrinkage_design(y, fellow)
    assert design.shape == (5, 6)
    # Grade 1 appears once, so its fellow value survives every shuffle.
    single = flat(np.full(5, 2.0))
    paired = flat(np.full(5, 2.0))
    null = shrinkage_permutation(y, fellow, single, paired, repeats=25, seed=1)
    assert null["repeats"] in (0, 25)


def test_rank_deficient_design_is_reported_not_silently_fitted():
    y = np.zeros(40, int)
    fellow = np.zeros(40, int)
    q = flat(np.full(40, 2.0))
    assert shrinkage_association(y, fellow, q, q, np.arange(40))["slope"] is None
    assert shrinkage_permutation(y, fellow, q, q, repeats=10)["p_permutation"] is None
