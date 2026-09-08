"""Diagnostic headroom for boundary rollback, measured under the deployed decoder.

Reads saved single/paired predictions and reports, for every ordinal boundary:

* how many label events exist there,
* how often each expert crosses the decision threshold,
* how often the two experts' boundary probabilities differ at all, and how often
  they land on opposite sides of the threshold,
* how often a rollback policy intervenes, over two policy classes: the sixteen
  binary masks, and a continuous alpha grid.

WHAT THIS IS NOT
----------------
None of these numbers upper-bound the QWK a learned gate can reach.

1. Both searches minimise per-eye squared grade error greedily. QWK does not
   decompose per eye -- its denominator depends on the marginal distribution of
   predictions -- so the per-eye argmin is not the QWK argmax.
2. The binary search covers alpha in {0,1}^4 only. PAVA couples the four
   boundaries after blending, so a continuous alpha reaches projected patterns no
   binary mask reaches. `eyes_grid_beats_binary` counts exactly that; the grid is
   itself only a finite sample of [0,1]^4.
3. A boundary where both experts sit on the same side of the threshold is not
   immovable: changing alpha at another boundary shifts the projected value here
   through PAVA. `probability_differs` is the wider, more honest count.

Read a near-zero headroom as a stop signal. Do NOT read a positive headroom as
evidence a gate can learn it, and do not treat any fixed number as a pass mark --
pair it with patient-clustered bootstrap CIs, the count of eyes a policy actually
touches, and the effect at the clinical boundaries.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from drpair.evaluation.metrics import qwk
from drpair.evaluation.ordinal import check_probabilities, decode, project
from drpair.teachers.targets import discrete_oracle
from drpair.training.predictions import load_pair

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def project_rows(q):
    """Vectorised decreasing isotonic projection; agrees with ordinal.project."""
    out = np.empty_like(q)
    for i in range(4):
        left = [
            np.stack([q[:, a : b + 1].mean(1) for b in range(i, 4)], 1).max(1) for a in range(i + 1)
        ]
        out[:, i] = np.stack(left, 1).min(1)
    return out


def grid_search(qs, qf, y, levels=GRID, threshold=0.5, chunk=2048):
    """Per-eye best alpha on a continuous grid, by squared grade error.

    Combinations are visited in order of increasing total alpha, so ties resolve
    to the weakest intervention rather than an arbitrary one.
    """
    combos = np.asarray(list(itertools.product(levels, repeat=4)), np.float32)
    combos = combos[np.argsort(combos.sum(1), kind="stable")]
    y = np.asarray(y)
    alpha = np.zeros_like(qs)
    error = np.empty(len(y))
    for start in range(0, len(y), chunk):
        window = slice(start, min(start + chunk, len(y)))
        s, f, target = qs[window], qf[window], y[window]
        raw = f[:, None, :] + combos[None, :, :] * (s - f)[:, None, :]
        n, k, _ = raw.shape
        pred = (project_rows(raw.reshape(n * k, 4)) >= threshold).sum(1).reshape(n, k)
        squared = (pred - target[:, None]) ** 2
        pick = squared.argmin(1)
        alpha[window] = combos[pick]
        error[window] = squared[np.arange(n), pick]
    return {"alpha": alpha, "error": error, "q": project_rows(qf + alpha * (qs - qf))}


def boundary_power(y, qs, qf, threshold=0.5, levels=GRID):
    y = np.asarray(y)
    qs = check_probabilities(qs).astype(np.float32)
    qf = check_probabilities(qf).astype(np.float32)
    target = y[:, None] > np.arange(4)
    fires_single, fires_paired = project(qs) >= threshold, project(qf) >= threshold
    binary = discrete_oracle(qs, qf, y, threshold)
    grid = grid_search(qs, qf, y, levels, threshold)
    base_error = (decode(qf, threshold) - y) ** 2
    binary_error = base_error - binary["utility"]

    rows = []
    for b in range(4):
        crossing = fires_single[:, b] != fires_paired[:, b]
        rows.append(
            {
                "boundary": b,
                "label_positive": int(target[:, b].sum()),
                "label_positive_rate": float(target[:, b].mean()),
                "single_fires": int(fires_single[:, b].sum()),
                "paired_fires": int(fires_paired[:, b].sum()),
                "probability_differs": int((np.abs(qs[:, b] - qf[:, b]) > 1e-6).sum()),
                "threshold_crossing": int(crossing.sum()),
                "threshold_crossing_rate": float(crossing.mean()),
                "binary_interventions": int((binary["mask"][:, b] > 0).sum()),
                "grid_interventions": int((grid["alpha"][:, b] > 0).sum()),
            }
        )

    summary = {
        "n_eyes": len(y),
        "grid_levels": list(levels),
        "qwk_single": qwk(y, decode(qs, threshold)),
        "qwk_paired_default": qwk(y, decode(qf, threshold)),
        "qwk_binary_rollback": qwk(y, decode(binary["q"], threshold)),
        "qwk_grid_rollback": qwk(y, decode(grid["q"], threshold)),
        "eyes_touched_binary": int((binary["mask"] > 0).any(axis=1).sum()),
        "eyes_touched_grid": int((grid["alpha"] > 0).any(axis=1).sum()),
        "eyes_grid_beats_binary": int((grid["error"] < binary_error).sum()),
        "squared_error_paired": float(base_error.mean()),
        "squared_error_binary": float(binary_error.mean()),
        "squared_error_grid": float(grid["error"].mean()),
        "not_an_upper_bound": (
            "Per-eye greedy squared-error search over a finite alpha set. QWK does "
            "not decompose per eye, so this bounds neither a learned gate nor QWK."
        ),
    }
    for name, key in (("binary", "qwk_binary_rollback"), ("grid", "qwk_grid_rollback")):
        base, top = summary["qwk_paired_default"], summary[key]
        summary[f"qwk_gap_{name}_vs_paired"] = (
            None if base is None or top is None else float(top - base)
        )
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Boundary rollback headroom diagnostic")
    parser.add_argument("--single", type=Path, required=True, help="single-expert .npz")
    parser.add_argument("--paired", type=Path, required=True, help="paired-expert .npz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--levels", type=float, nargs="+", default=list(GRID))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    s, f = load_pair(args.single, args.paired)
    rows, summary = boundary_power(s["y"], s["q"], f["q"], args.threshold, tuple(args.levels))

    header = (
        f"{'b':>2} {'pos':>7} {'s_fire':>7} {'p_fire':>7} "
        f"{'q_diff':>7} {'cross':>7} {'bin_iv':>7} {'grid_iv':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['boundary']:>2} {row['label_positive']:>7} {row['single_fires']:>7} "
            f"{row['paired_fires']:>7} {row['probability_differs']:>7} "
            f"{row['threshold_crossing']:>7} {row['binary_interventions']:>7} "
            f"{row['grid_interventions']:>8}"
        )
    print()
    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {"threshold": args.threshold, "summary": summary, "boundaries": rows}
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
