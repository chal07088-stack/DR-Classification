"""Boundary-level feasibility check to run before committing GPU-days to PHASE 1.

Reads saved single/paired predictions and reports, for every ordinal boundary:

* how many label events exist at that boundary,
* how often each expert actually crosses the decision threshold,
* how often the two experts disagree -- the only eyes a rollback gate can move,
* how often the discrete oracle intervenes, and the QWK ceiling it buys.

A boundary the experts never cross, or never disagree on, cannot support a
boundary-wise claim no matter how the gate is trained. Reading this off the OOF
dump costs minutes; discovering it after the full schedule costs GPU-days.
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
    ceiling = {
        "n_eyes": len(y),
        "qwk_single": qwk(y, decode(qs, threshold)),
        "qwk_paired_default": qwk(y, decode(qf, threshold)),
        "qwk_discrete_oracle": qwk(y, decode(oracle["q"], threshold)),
        "eyes_oracle_helps": int((oracle["utility"] > 0).sum()),
        "eyes_oracle_touches": int((oracle["mask"] > 0).any(axis=1).sum()),
    }
    base, top = ceiling["qwk_paired_default"], ceiling["qwk_discrete_oracle"]
    ceiling["oracle_headroom_over_paired"] = (
        None if base is None or top is None else float(top - base)
    )
    return rows, ceiling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True, help="single-expert .npz")
    parser.add_argument("--paired", type=Path, required=True, help="paired-expert .npz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    s, f = load_pair(args.single, args.paired)
    rows, ceiling = boundary_power(s["y"], s["q"], f["q"], args.threshold)
    report = {"threshold": args.threshold, "ceiling": ceiling, "boundaries": rows}

    header = (
        f"{'b':>2} {'pos':>8} {'s_fire':>8} {'p_fire':>8} {'disagree':>9} {'oracle':>8}  verdict"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        verdict = "actionable" if row["actionable"] else "NO HEADROOM"
        print(
            f"{row['boundary']:>2} {row['label_positive']:>8} {row['single_fires']:>8} "
            f"{row['paired_fires']:>8} {row['expert_disagreement']:>9} "
            f"{row['oracle_interventions']:>8}  {verdict}"
        )
    print()
    print(json.dumps(ceiling, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    dead = [row["boundary"] for row in rows if not row["actionable"]]
    if dead:
        print(f"\nBoundaries with nothing for a gate to act on: {dead}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
