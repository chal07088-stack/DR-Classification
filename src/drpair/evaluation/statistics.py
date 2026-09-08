import numpy as np
from scipy.stats import norm

from .ordinal import project


def adjust_pvalues(pvalues, method="bh"):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    rank = np.arange(1, len(p) + 1)
    if method == "bh":
        ordered = np.minimum.accumulate((p[order] * len(p) / rank)[::-1])[::-1]
    elif method == "holm":
        ordered = np.maximum.accumulate(p[order] * (len(p) - rank + 1))
    else:
        raise ValueError(method)
    result = np.empty_like(p)
    result[order] = np.minimum(ordered, 1)
    return result


def qwk_from_confusion(confusion):
    n = confusion.sum(axis=(-2, -1))
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2
    expected = confusion.sum(-1)[..., :, None] * confusion.sum(-2)[..., None, :]
    denominator = (expected * weights).sum(axis=(-2, -1))
    numerator = (confusion * weights).sum(axis=(-2, -1)) * n
    return (
        np.divide(
            numerator,
            denominator,
            out=np.full_like(denominator, np.nan, dtype=float),
            where=denominator > 0,
        )
        * (-1)
        + 1
    )


def paired_bootstrap_delta(y, reference, candidate, patient, repeats=1000, seed=42):
    patient = np.asarray(patient)
    ids, inverse = np.unique(patient, return_inverse=True)
    if len(ids) < 2:
        return {"delta": None, "ci_low": None, "ci_high": None, "valid_replicates": 0}
    matrices = []
    for pred in (reference, candidate):
        cm = np.zeros((len(ids), 5, 5), dtype=np.int64)
        np.add.at(cm, (inverse, np.asarray(y), pred), 1)
        matrices.append(cm)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        indices = rng.integers(0, len(ids), len(ids))
        draws.append(
            qwk_from_confusion(matrices[1][indices].sum(0))
            - qwk_from_confusion(matrices[0][indices].sum(0))
        )
    draws = np.asarray(draws)
    draws = draws[np.isfinite(draws)]
    point = qwk_from_confusion(matrices[1].sum(0)) - qwk_from_confusion(matrices[0].sum(0))
    return {
        "delta": float(point) if np.isfinite(point) else None,
        "ci_low": float(np.quantile(draws, 0.025)) if len(draws) else None,
        "ci_high": float(np.quantile(draws, 0.975)) if len(draws) else None,
        "valid_replicates": len(draws),
    }


def locality_analysis(y, fellow_y, qs, qf, patient, threshold=0.5):
    """Descriptive harm-vs-gap slopes, clustered by patient, with BH/Holm.

    Grade indicators adjust different boundary prevalence. Normal Wald inference
    is asymptotic and not a mechanistic/causal proof; sparse cells are disclosed.
    """
    y = np.asarray(y)
    target = y[:, None] > np.arange(4)
    d = ((project(qf) >= threshold) != target).astype(float)
    d -= ((project(qs) >= threshold) != target).astype(float)
    gap = np.abs(y - np.asarray(fellow_y))
    x = np.column_stack([np.ones(len(y)), gap] + [(y == g).astype(float) for g in range(1, 5)])
    unique, inverse = np.unique(patient, return_inverse=True)
    bread = np.linalg.pinv(x.T @ x)
    coef = bread @ x.T @ d
    residual = d - x @ coef
    influence = np.zeros((len(unique), 4))
    for b in range(4):
        score = np.zeros((len(unique), x.shape[1]))
        np.add.at(score, inverse, x * residual[:, b, None])
        influence[:, b] = score @ bread[:, 1]
    correction = len(unique) / max(1, len(unique) - 1) * (len(y) - 1) / max(1, len(y) - x.shape[1])
    covariance = influence.T @ influence * correction
    rows = []
    for b in range(4):
        se = float(np.sqrt(max(covariance[b, b], 0)))
        value = float(coef[1, b])
        p = float(2 * norm.sf(abs(value / se))) if se > 1e-12 else 1.0
        rows.append(
            {
                "boundary": b,
                "slope": value,
                "cluster_se": se,
                "p": p,
                "ci_low": value - 1.96 * se,
                "ci_high": value + 1.96 * se,
                "harmful_count": int((d[:, b] > 0).sum()),
                "beneficial_count": int((d[:, b] < 0).sum()),
                "n_patients": len(unique),
            }
        )
    for method in ("bh", "holm"):
        for row, value in zip(rows, adjust_pvalues([r["p"] for r in rows], method)):
            row[method] = float(value)
    interactions = []
    for b in range(1, 4):
        delta = float(coef[1, b] - coef[1, 0])
        se = float(np.sqrt(max(covariance[b, b] + covariance[0, 0] - 2 * covariance[b, 0], 0)))
        p = float(2 * norm.sf(abs(delta / se))) if se > 1e-12 else 1.0
        interactions.append({"boundary_vs_b0": b, "slope_difference": delta, "se": se, "p": p})
    for method in ("bh", "holm"):
        for row, value in zip(interactions, adjust_pvalues([r["p"] for r in interactions], method)):
            row[method] = float(value)
    return rows, interactions
