"""Can a boundary-wise claim be made at all? Answered from labels, before training.

The arms are compared on the same eyes, so the clinical endpoint -- sensitivity, or
equivalently missed cases, at a matched specificity -- is a paired binary comparison
clustered by patient. Its power is governed by the number of eyes where the two arms
actually disagree, not by the size of the dataset.

Everything here comes from `data/splits/eyepacs.csv` alone: positive counts, the
patient cluster structure, and the within-patient correlation of each boundary
indicator. No predictions, no GPU. Run it first. A boundary whose minimum detectable
difference exceeds any difference the arms could produce cannot support a claim no
matter how the experiment turns out, and that is worth knowing before the schedule
runs rather than after.

METHOD. Among P positive eyes let d be the fraction where the two arms disagree, and
psi the share of those disagreements favouring the candidate. The sensitivity
difference is delta = d * (2*psi - 1), and McNemar on the discordant eyes needs

    |2*psi - 1| * sqrt(P * d / DEFF) >= z_alpha + z_power

so the smallest detectable difference is

    delta_min = (z_alpha + z_power) * sqrt(d * DEFF / P).

Because delta can never exceed d, a boundary is undetectable outright unless
d * P / DEFF >= (z_alpha + z_power)^2, which fixes a floor on the discordant eyes
needed before any effect at all is visible.

LIMITS. DEFF uses the within-patient correlation of the *label* indicator as a stand-in
for the correlation of arm disagreement, which cannot be known before predictions
exist; it is a plausible value, not a measured one. The normal approximation to
McNemar is unreliable at very small discordant counts, so counts are printed beside
every figure. This bounds detectability, not the truth of any hypothesis.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

DISCORDANCE_GRID = (0.01, 0.02, 0.05, 0.10, 0.20)
BOUNDARY_NAMES = {
    0: "any_dr_ge1",
    1: "referable_ge2",
    2: "vision_threatening_ge3",
    3: "proliferative_ge4",
}


def read_labels(manifest, split):
    rows = []
    with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            if record["split"] != split:
                continue
            rows.append((str(record["patient"]), int(record["label"]), int(record["fellow_label"])))
    if not rows:
        raise SystemExit(f"No rows for split {split!r} in {manifest}")
    patient = np.array([r[0] for r in rows])
    return patient, np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def pairwise_icc(values, patient):
    """Intraclass correlation for exactly-paired clusters; None if degenerate."""
    grouped = defaultdict(list)
    for p, v in zip(patient, np.asarray(values, float)):
        grouped[p].append(v)
    pairs = np.array([v for v in grouped.values() if len(v) == 2])
    if len(pairs) < 2:
        return None
    everything = pairs.reshape(-1)
    mean = everything.mean()
    variance = everything.var()
    if variance <= 0:
        return None
    centred = pairs - mean
    return float((centred[:, 0] * centred[:, 1]).mean() / variance)


def design_effect(icc, mean_cluster_size):
    if icc is None:
        return 1.0, None
    raw = 1.0 + (mean_cluster_size - 1.0) * icc
    # A negative correlation would help; refusing to bank on it keeps this a floor.
    return float(max(raw, 1.0)), float(raw)


def groups_from(y, fellow):
    gap = np.abs(np.asarray(y) - np.asarray(fellow))
    return {
        "overall": np.ones(len(y), bool),
        "gap0": gap == 0,
        "gap1": gap == 1,
        "gap_ge1": gap >= 1,
        "gap_ge2": gap >= 2,
    }


def minimum_detectable(positives, discordance, deff, z_total):
    """Smallest detectable sensitivity difference, or None when none exists."""
    if positives <= 0 or discordance <= 0:
        return None, 0.0
    delta = z_total * np.sqrt(discordance * deff / positives)
    # delta can never exceed the discordance rate itself.
    return (float(delta) if delta <= discordance else None), float(positives * discordance)


def boundary_rows(patient, y, fellow, alpha=0.05, power=0.8, grid=DISCORDANCE_GRID):
    z_total = float(norm.ppf(1 - alpha / 2) + norm.ppf(power))
    rows = []
    for boundary in range(4):
        indicator = (y > boundary).astype(float)
        icc = pairwise_icc(indicator, patient)
        for group, mask in groups_from(y, fellow).items():
            positive = mask & (indicator > 0)
            positives = int(positive.sum())
            positive_patients = len(np.unique(patient[positive])) if positives else 0
            mean_cluster = positives / positive_patients if positive_patients else 0.0
            deff, raw_deff = design_effect(icc, mean_cluster)
            row = {
                "boundary": boundary,
                "condition": BOUNDARY_NAMES[boundary],
                "group": group,
                "eyes": int(mask.sum()),
                "positive_eyes": positives,
                "positive_patients": positive_patients,
                "mean_positive_eyes_per_patient": round(mean_cluster, 4),
                "label_icc_within_patient": icc,
                "design_effect": deff,
                "design_effect_uncapped": raw_deff,
                "effective_positive_eyes": round(positives / deff, 1) if positives else 0.0,
                "min_discordant_eyes_for_any_effect": round(z_total**2 * deff, 1),
            }
            detectable = False
            for discordance in grid:
                delta, discordant = minimum_detectable(positives, discordance, deff, z_total)
                key = f"mde_at_discordance_{round(discordance * 100):02d}pct"
                row[key] = None if delta is None else round(delta, 4)
                row[key + "_discordant_eyes"] = round(discordant, 1)
                detectable = detectable or delta is not None
            row["any_grid_point_detectable"] = detectable
            rows.append(row)
    return rows, z_total


def main():
    parser = argparse.ArgumentParser(description="Boundary-level detectability from labels alone")
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/eyepacs.csv"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--discordance", type=float, nargs="+", default=list(DISCORDANCE_GRID))
    parser.add_argument("--group", default="overall", help="group shown in the printed table")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    patient, y, fellow = read_labels(args.manifest, args.split)
    rows, z_total = boundary_rows(
        patient, y, fellow, args.alpha, args.power, tuple(args.discordance)
    )

    print(
        f"split={args.split}  eyes={len(y)}  patients={len(np.unique(patient))}  "
        f"alpha={args.alpha}  power={args.power}  z_total={z_total:.4f}"
    )
    print(f"\ngroup shown: {args.group}\n")
    columns = [f"{round(d * 100):02d}%" for d in args.discordance]
    header = (
        f"{'condition':>22} {'pos':>7} {'pat':>7} {'ICC':>7} {'DEFF':>6} {'min_disc':>9}  "
        + " ".join(f"{c:>7}" for c in columns)
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["group"] != args.group:
            continue
        icc = (
            "-"
            if row["label_icc_within_patient"] is None
            else f"{row['label_icc_within_patient']:.3f}"
        )
        cells = []
        for d in args.discordance:
            value = row[f"mde_at_discordance_{round(d * 100):02d}pct"]
            cells.append("  none" if value is None else f"{value:.4f}")
        print(
            f"{row['condition']:>22} {row['positive_eyes']:>7} {row['positive_patients']:>7} "
            f"{icc:>7} {row['design_effect']:>6.2f} "
            f"{row['min_discordant_eyes_for_any_effect']:>9.1f}  "
            + " ".join(f"{c:>7}" for c in cells)
        )

    dead = [f"{r['condition']}/{r['group']}" for r in rows if not r["any_grid_point_detectable"]]
    if dead:
        print("\nNo grid point yields a detectable difference for:")
        for name in dead:
            print(f"  {name}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "alpha": args.alpha,
                    "power": args.power,
                    "z_total": z_total,
                    "discordance_grid": list(args.discordance),
                    "endpoint": "sensitivity difference between two arms on the same eyes",
                    "caveat": (
                        "DEFF uses the label indicator correlation as a stand-in for the "
                        "correlation of arm disagreement, which is unknown before predictions "
                        "exist. Bounds detectability, not the truth of any hypothesis."
                    ),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
