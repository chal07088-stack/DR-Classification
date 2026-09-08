import numpy as np

from drpair.evaluation.ordinal import check_probabilities


def fellow_predictions(qs, patient, side):
    lookup = {(str(p), str(s)): i for i, (p, s) in enumerate(zip(patient, side))}
    if len(lookup) != len(qs):
        raise ValueError("Duplicate prediction identity")
    indices = [lookup[(str(p), "right" if s == "left" else "left")] for p, s in zip(patient, side)]
    return qs[indices]


def entropy(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -(q * np.log(q) + (1 - q) * np.log1p(-q))


def gate_features(qs, qf, fellow_q, threshold=0.5):
    qs, qf, fellow_q = map(check_probabilities, (qs, qf, fellow_q))
    if qs.shape != qf.shape or qs.shape != fellow_q.shape:
        raise ValueError("Unaligned expert probabilities")
    signed = qs.sum(axis=1, keepdims=True) - fellow_q.sum(axis=1, keepdims=True)
    blocks = {
        "single": qs,
        "fusion": qf,
        "fellow_single": fellow_q,
        "shift": qf - qs,
        "absolute_shift": abs(qf - qs),
        "single_margin": abs(qs - threshold),
        "fusion_margin": abs(qf - threshold),
        "single_entropy": entropy(qs),
        "fusion_entropy": entropy(qf),
        "boundary_conflict": ((qs >= threshold) != (qf >= threshold)).astype(float),
        "signed_asymmetry": signed,
        "absolute_asymmetry": abs(signed),
    }
    names = [f"{name}_{b}" for name, x in blocks.items() for b in range(x.shape[1])]
    return np.concatenate(list(blocks.values()), axis=1).astype(np.float32), names
