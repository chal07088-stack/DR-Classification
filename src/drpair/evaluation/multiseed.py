"""Aggregate the locked seed-42 result and PHASE 2 replication seeds."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from drpair.data.manifest import write_csv
from drpair.experiments.arms import ARMS, PROPOSED
from drpair.runtime import save_json

GROUPS = ("overall", "gap_ge1", "gap_ge2")


def _test_qwk(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (row["arm"], row["group"]): float(row["qwk"])
        for row in rows
        if row["split"] == "test" and row["group"] in GROUPS
    }


def make_multiseed_report(results_root, runs):
    """Write per-seed and mean/SD QWK without treating folds as training seeds."""
    results_root = Path(results_root)
    output = results_root / "phase2_multiseed_v1"
    metrics_dir = output / "metrics"
    per_seed = []
    for seed, run_name in runs:
        values = _test_qwk(results_root / run_name / "metrics" / "arm_metrics.csv")
        for arm in ARMS:
            for group in GROUPS:
                per_seed.append(
                    {"seed": seed, "arm": arm, "group": group, "qwk": values[(arm, group)]}
                )
    write_csv(metrics_dir / "per_seed_qwk.csv", per_seed)

    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["arm"], row["group"])].append(row["qwk"])
    summary = []
    for arm in ARMS:
        for group in GROUPS:
            values = np.asarray(grouped[(arm, group)], dtype=float)
            summary.append(
                {
                    "arm": arm,
                    "group": group,
                    "n_seeds": len(values),
                    "mean_qwk": float(values.mean()),
                    "sd_qwk": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    write_csv(metrics_dir / "multiseed_summary.csv", summary)

    def result(arm, group):
        row = next(r for r in summary if r["arm"] == arm and r["group"] == group)
        return f"{row['mean_qwk']:.4f} +/- {row['sd_qwk']:.4f}"

    lines = [
        "# PHASE 2A multi-seed replication",
        "",
        "Locked EyePACS / ConvNeXt-S protocol. Values are mean +/- sample SD across training seeds.",
        "Patient bootstrap intervals remain in each seed report and are not substitutes for seed SD.",
        "",
        "| Arm | Overall QWK | Gap >= 1 QWK | Gap >= 2 QWK |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {result(arm, 'overall')} | {result(arm, 'gap_ge1')} | "
            f"{result(arm, 'gap_ge2')} |"
        )
    lines += [
        "",
        f"Proposed arm remained fixed as `{PROPOSED}` before the replication runs.",
        "Train-expert OOF was not repeated because the deployed gates are fitted by patient-level",
        "cross-fitting on official validation predictions. Final experts and gates were retrained per seed.",
    ]
    output.mkdir(parents=True, exist_ok=True)
    report = output / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(
        output / "completion.json",
        {
            "phase": 2,
            "stage": "2a_multiseed",
            "status": "complete",
            "proposed": PROPOSED,
            "seeds": [seed for seed, _ in runs],
        },
    )
    return report
