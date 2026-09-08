"""Labels supervise prediction interventions, never enter inference features."""

import itertools

import numpy as np

from drpair.evaluation.ordinal import blend, decode, project


def loss_guided_teacher(qs, qf, y, temperature=1.0):
    target = np.asarray(y)[:, None] > np.arange(4)

    def bce(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -(target * np.log(q) + (1 - target) * np.log1p(-q))

    utility = bce(qf) - bce(qs)
    alpha = -np.expm1(-np.maximum(utility, 0) / temperature)
    return {"utility": utility.astype(np.float32), "alpha": alpha.astype(np.float32)}


def counterfactual_teacher(qs, qf, y, threshold=0.5, temperature=1.0):
    """One-boundary U_b only; each candidate is projected and decoded in full."""
    if temperature <= 0:
        raise ValueError("Teacher temperature must be positive")
    y = np.asarray(y)
    base_error = (decode(qf, threshold) - y) ** 2
    utility = np.zeros_like(qs, dtype=np.float32)
    for b in range(4):
        candidate = np.array(qf, copy=True)
        candidate[:, b] = qs[:, b]
        utility[:, b] = base_error - (decode(candidate, threshold) - y) ** 2
    # U_b is the teacher label. This fixed link maps positive utility to a soft target.
    alpha = -np.expm1(-np.maximum(utility, 0) / temperature)
    return {"utility": utility, "alpha": alpha.astype(np.float32)}


def joint_effect(qs, qf, y, alpha, threshold=0.5):
    """Evaluation only: deployed simultaneous rollback, projection, and decode."""
    q, raw = blend(qs, qf, alpha)
    base_error = (decode(qf, threshold) - np.asarray(y)) ** 2
    error = (decode(q, threshold) - np.asarray(y)) ** 2
    return {"utility": (base_error - error).astype(np.float32), "q": q, "raw": raw}


def discrete_oracle(qs, qf, y, threshold=0.5):
    """Enumerate all 16 rollback masks under the deployed PAVA/decoder policy.

    Primary utility: squared ordinal error. Ties: fewer interventions, then
    smaller ordinal Brier loss. Local one-boundary effects remain diagnostics.
    This is a discrete oracle, not an upper bound on all continuous soft blends.
    """
    y = np.asarray(y)
    target = y[:, None] > np.arange(4)
    base = project(qf)
    base_error = (decode(base, threshold) - y) ** 2
    masks = sorted(itertools.product([0.0, 1.0], repeat=4), key=lambda a: (sum(a), a))
    best_error = base_error.copy()
    best_count = np.zeros(len(y))
    best_brier = ((base - target) ** 2).mean(axis=1)
    alpha = np.zeros_like(qs)
    chosen_q = base.copy()
    for mask in masks[1:]:
        a = np.broadcast_to(mask, qs.shape)
        q, _ = blend(qs, qf, a)
        error = (decode(q, threshold) - y) ** 2
        brier = ((q - target) ** 2).mean(axis=1)
        count = sum(mask)
        better = (error < best_error) | ((error == best_error) & (count < best_count))
        better |= (error == best_error) & (count == best_count) & (brier < best_brier - 1e-9)
        alpha[better] = a[better]
        chosen_q[better] = q[better]
        best_error[better], best_count[better], best_brier[better] = (
            error[better],
            count,
            brier[better],
        )
    return {
        "mask": alpha.astype(np.float32),
        "utility": (base_error - best_error).astype(np.float32),
        "q": chosen_q,
        "base_error": base_error.astype(np.float32),
    }
