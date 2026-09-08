"""Boundary-level feasibility check to run before committing GPU-days to PHASE 1.

Reads saved single/paired predictions and reports, for every ordinal boundary:

* how many label events exist at that boundary,
* how often each expert actually crosses the decision threshold,
* how often the experts disagree at each boundary decision,
* binary rollback intervention counts and the resulting diagnostic QWK difference.

Threshold event counts do not measure all ranking or calibration opportunities.
The oracle minimizes per-eye squared grade error over 16 masks, not QWK; its
reported QWK is not an upper bound on continuous mixing. Use PHASE 1A diagnostics
for calibration, finite grids and clustered intervals before drawing conclusions.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from drpair.evaluation.metrics import qwk
from drpair.evaluation.ordinal import decode, project
from drpair.teachers.targets import discrete_oracle
from drpair.training.predictions import load_pair


def boundary_power(y, qs, qf, threshold=0.5):
    """Per-boundary event counts, expert disagreement, and oracle interventions."""
    y = np.asarray(y)
    target = y[:, None] > np.arange(4)
    fires_single = project(qs) >= threshold
    fires_paired = project(qf) >= threshold
    oracle = discrete_oracle(qs, qf, y, threshold)
    rows = []
    for b in range(4):
        disagreement = fires_single[:, b] != fires_paired[:, b]
        intervention = oracle["mask"][:, b] > 0
        rows.append(
            {
                "boundary": b,
                "label_positive": int(target[:, b].sum()),
                "label_positive_rate": float(target[:, b].mean()),
                "single_fires": int(fires_single[:, b].sum()),
                "paired_fires": int(fires_paired[:, b].sum()),
                "expert_disagreement": int(disagreement.sum()),
                "expert_disagreement_rate": float(disagreement.mean()),
                "oracle_interventions": int(intervention.sum()),
                "oracle_intervention_rate": float(intervention.mean()),
                "actionable": bool(disagreement.any() and intervention.any()),
            }
        )
    diagnostic = {
        "n_eyes": len(y),
        "qwk_single": qwk(y, decode(qs, threshold)),
        "qwk_paired_default": qwk(y, decode(qf, threshold)),
        "qwk_discrete_oracle": qwk(y, decode(oracle["q"], threshold)),
        "eyes_oracle_helps": int((oracle["utility"] > 0).sum()),
        "eyes_oracle_touches": int((oracle["mask"] > 0).any(axis=1).sum()),
    }
    base, top = diagnostic["qwk_paired_default"], diagnostic["qwk_discrete_oracle"]
    diagnostic["binary_qwk_delta_vs_paired"] = (
        None if base is None or top is None else float(top - base)
    )
    return rows, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True, help="single-expert .npz")
    parser.add_argument("--paired", type=Path, required=True, help="paired-expert .npz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    s, f = load_pair(args.single, args.paired)
    rows, diagnostic = boundary_power(s["y"], s["q"], f["q"], args.threshold)
    report = {
        "threshold": args.threshold,
        "binary_rollback_diagnostic": diagnostic,
        "boundaries": rows,
    }

    header = (
        f"{'b':>2} {'pos':>8} {'s_fire':>8} {'p_fire':>8} {'disagree':>9} {'oracle':>8}  verdict"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        verdict = "observed" if row["actionable"] else "NO OBSERVED BINARY EVENT"
        print(
            f"{row['boundary']:>2} {row['label_positive']:>8} {row['single_fires']:>8} "
            f"{row['paired_fires']:>8} {row['expert_disagreement']:>9} "
            f"{row['oracle_interventions']:>8}  {verdict}"
        )
    print()
    print(json.dumps(diagnostic, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    dead = [row["boundary"] for row in rows if not row["actionable"]]
    if dead:
        print(
            f"\nBoundaries without observed binary events: {dead}; inspect calibration too.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
