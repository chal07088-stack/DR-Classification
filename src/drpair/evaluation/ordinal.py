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
    out = np.empty_like(q)
    for i, row in enumerate(q):
        blocks = []
        for v in row:
            blocks.append([float(v), 1])
            while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
                right, left = blocks.pop(), blocks.pop()
                n = left[1] + right[1]
                blocks.append([(left[0] * left[1] + right[0] * right[1]) / n, n])
        out[i] = np.concatenate([np.repeat(v, n) for v, n in blocks])
    return out.astype(np.float32)


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
