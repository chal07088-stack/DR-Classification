import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from boundary_mde import (
    boundary_rows,
    design_effect,
    minimum_detectable,
    pairwise_icc,
    read_labels,
)

Z_TOTAL = float(norm.ppf(0.975) + norm.ppf(0.8))


def test_icc_is_one_for_perfectly_concordant_eyes():
    patient = np.repeat(np.arange(50), 2)
    values = np.repeat(np.arange(50) % 2, 2).astype(float)
    assert abs(pairwise_icc(values, patient) - 1.0) < 1e-9


def test_icc_is_near_zero_for_independent_eyes():
    rng = np.random.default_rng(0)
    patient = np.repeat(np.arange(6000), 2)
    values = rng.integers(0, 2, 12000).astype(float)
    assert abs(pairwise_icc(values, patient)) < 0.03


def test_icc_is_none_when_every_eye_agrees():
    patient = np.repeat(np.arange(20), 2)
    assert pairwise_icc(np.ones(40), patient) is None


def test_design_effect_refuses_to_bank_on_negative_correlation():
    capped, raw = design_effect(-0.4, 2.0)
    assert capped == 1.0 and raw < 1.0
    capped, raw = design_effect(0.5, 2.0)
    assert capped == raw == 1.5


def test_no_effect_is_detectable_below_the_discordance_floor():
    positives, deff = 400, 1.0
    floor = Z_TOTAL**2 * deff / positives
    below, _ = minimum_detectable(positives, floor * 0.9, deff, Z_TOTAL)
    at, _ = minimum_detectable(positives, floor, deff, Z_TOTAL)
    assert below is None
    assert at is not None and abs(at - floor) < 1e-9


def test_minimum_detectable_matches_a_mcnemar_simulation():
    """The closed form should deliver the power it claims, so simulate it."""
    positives, discordance, replicates = 2000, 0.10, 6000
    delta, discordant = minimum_detectable(positives, discordance, 1.0, Z_TOTAL)
    assert abs(discordant - 200.0) < 1e-9
    favouring = (delta / discordance + 1) / 2  # invert delta = d * (2*psi - 1)
    rng = np.random.default_rng(42)
    counts = rng.binomial(positives, discordance, replicates)
    b = rng.binomial(counts, favouring)
    c = counts - b
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(counts > 0, (b - c) / np.sqrt(np.maximum(counts, 1)), 0.0)
    empirical = float(np.mean(np.abs(z) > norm.ppf(0.975)))
    assert abs(empirical - 0.8) < 0.04


def test_clustering_makes_the_detectable_difference_larger():
    independent, _ = minimum_detectable(1000, 0.1, 1.0, Z_TOTAL)
    clustered, _ = minimum_detectable(1000, 0.1, 1.8, Z_TOTAL)
    assert clustered > independent
    assert abs(clustered / independent - np.sqrt(1.8)) < 1e-9


def write_manifest(path, records):
    header = "patient,side,label,fellow_label,split,path,fellow_path\n"
    lines = [header]
    for patient, left, right, split in records:
        lines.append(f"{patient},left,{left},{right},{split},a.jpeg,b.jpeg\n")
        lines.append(f"{patient},right,{right},{left},{split},b.jpeg,a.jpeg\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_rows_report_patients_not_just_eyes(tmp_path):
    manifest = tmp_path / "m.csv"
    # Mostly concordant, as bilateral DR actually is: 30 patients positive in both
    # eyes, 10 in one, 20 in neither.
    records = (
        [(i, 4, 4, "val") for i in range(30)]
        + [(30 + i, 4, 0, "val") for i in range(10)]
        + [(40 + i, 0, 0, "val") for i in range(20)]
    )
    write_manifest(manifest, records)
    patient, y, fellow = read_labels(manifest, "val")
    rows, _ = boundary_rows(patient, y, fellow)
    b3 = next(r for r in rows if r["boundary"] == 3 and r["group"] == "overall")
    assert b3["positive_eyes"] == 70
    assert b3["positive_patients"] == 40
    assert abs(b3["mean_positive_eyes_per_patient"] - 1.75) < 1e-9
    # Closed form for these counts, so the estimator is pinned, not merely signed.
    assert abs(b3["label_icc_within_patient"] - 0.657142857) < 1e-6
    assert abs(b3["design_effect"] - (1 + 0.75 * 0.657142857)) < 1e-6
    assert b3["effective_positive_eyes"] < b3["positive_eyes"]


def test_a_deliberately_discordant_cohort_gives_negative_icc_and_a_capped_deff(tmp_path):
    manifest = tmp_path / "m.csv"
    # Half the patients discordant at grade 4 drives the correlation below zero.
    records = [(i, 4, 4, "val") for i in range(10)] + [(10 + i, 4, 0, "val") for i in range(10)]
    write_manifest(manifest, records)
    patient, y, fellow = read_labels(manifest, "val")
    rows, _ = boundary_rows(patient, y, fellow)
    b3 = next(r for r in rows if r["boundary"] == 3 and r["group"] == "overall")
    assert abs(b3["label_icc_within_patient"] + 1 / 3) < 1e-9
    assert b3["design_effect_uncapped"] < 1.0
    # Anticonservative gains from negative correlation are not banked.
    assert b3["design_effect"] == 1.0
    assert b3["effective_positive_eyes"] == b3["positive_eyes"]


def test_split_filter_is_honoured(tmp_path):
    manifest = tmp_path / "m.csv"
    write_manifest(manifest, [(0, 1, 2, "val"), (1, 3, 4, "test")])
    patient, y, _ = read_labels(manifest, "test")
    assert len(y) == 2 and set(y) == {3, 4} and set(patient) == {"1"}
