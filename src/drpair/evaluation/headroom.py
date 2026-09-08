"""Label-aware rollback diagnostics, not learnable results or QWK upper bounds."""

import itertools

import numpy as np

from .ordinal import blend, decode, project


def rollback_grid(qs, qf, y, values=(0, 0.25, 0.5, 0.75, 1), threshold=0.5):
    values = sorted({float(v) for v in values})
    if not values or values[0] != 0 or values[-1] != 1 or any(not np.isfinite(v) for v in values):
        raise ValueError("Alpha grid must include 0 and 1 and stay within [0, 1]")
    y = np.asarray(y)
    target = y[:, None] > np.arange(4)
    best_q = project(qf)
    base_error = (decode(best_q, threshold) - y) ** 2
    best_error = base_error.copy()
    best_alpha = np.zeros_like(qs)
    best_amount = np.zeros(len(y))
    best_brier = ((best_q - target) ** 2).mean(1)
    for mask in sorted(itertools.product(values, repeat=4), key=lambda m: (sum(m), m))[1:]:
        a = np.broadcast_to(mask, qs.shape)
        q, _ = blend(qs, qf, a)
        error = (decode(q, threshold) - y) ** 2
        amount = sum(mask)
        brier = ((q - target) ** 2).mean(1)
        better = (error < best_error) | ((error == best_error) & (amount < best_amount))
        better |= (error == best_error) & (amount == best_amount) & (brier < best_brier - 1e-9)
        best_q[better], best_alpha[better] = q[better], a[better]
        best_error[better], best_amount[better], best_brier[better] = (
            error[better],
            amount,
            brier[better],
        )
    return {
        "q": best_q,
        "alpha": best_alpha,
        "utility": (base_error - best_error).astype(np.float32),
    }
