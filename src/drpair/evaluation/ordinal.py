import numpy as np


def check_probabilities(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.all(np.isfinite(q)):
        raise ValueError("Expected finite N x 4 cumulative probabilities")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Probabilities outside [0,1]")
    return q


def project(q):
    """Euclidean projection onto decreasing probabilities, exact 4-point PAVA."""
    q = check_probabilities(q)
    # The PAVA min-max identity vectorizes all eyes for the diagnostic mask grid.
    means = {(a, b): q[:, a : b + 1].mean(1) for a in range(4) for b in range(a, 4)}
    out = [
        np.minimum.reduce(
            [np.maximum.reduce([means[a, b] for b in range(i, 4)]) for a in range(i + 1)]
        )
        for i in range(4)
    ]
    return np.column_stack(out).astype(np.float32)


def violations(q):
    q = check_probabilities(q)
    return np.any(np.diff(q, axis=1) > 1e-7, axis=1)


def decode(q, threshold=0.5):
    return (project(q) >= threshold).sum(axis=1)


def blend(qs, qf, alpha):
    qs, qf = check_probabilities(qs), check_probabilities(qf)
    alpha = np.asarray(alpha)
    if qs.shape != qf.shape or alpha.shape != qs.shape:
        raise ValueError("Prediction and alpha shapes must match")
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("Invalid alpha")
    raw = qf + alpha * (qs - qf)
    return project(raw), raw.astype(np.float32)


def class_probabilities(q):
    q = project(q)
    return np.column_stack([1 - q[:, 0], q[:, :-1] - q[:, 1:], q[:, -1]])
