import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, roc_auc_score

from .ordinal import decode, project, violations


def qwk(y, pred):
    if len(y) == 0 or len(np.unique(np.concatenate([y, pred]))) < 2:
        return None
    value = float(cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(5))))
    return value if np.isfinite(value) else None


def grade_metrics(y, q, threshold=0.5):
    y = np.asarray(y)
    pred = decode(q, threshold)
    if not len(y):
        return {"n_eyes": 0, "qwk": None}
    severe = y >= 3
    return {
        "n_eyes": len(y),
        "qwk": qwk(y, pred),
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)
        ),
        "g3_f1": float(f1_score(y == 3, pred == 3, zero_division=0)),
        "mae": float(np.mean(np.abs(y - pred))),
        "under_rate": float(np.mean(pred < y)),
        "over_rate": float(np.mean(pred > y)),
        "ge2_error_rate": float(np.mean(np.abs(y - pred) >= 2)),
        "severe_n": int(severe.sum()),
        "severe_under_rate": float(np.mean(pred[severe] <= 2)) if severe.any() else None,
        "monotonic_violations": int(violations(q).sum()),
    }


def subgroup_metrics(y, q, fellow_y, patient, threshold=0.5):
    gap = np.abs(np.asarray(y) - np.asarray(fellow_y))
    groups = {
        "overall": np.ones(len(y), bool),
        "gap0": gap == 0,
        "gap1": gap == 1,
        "gap_ge1": gap >= 1,
        "gap_ge2": gap >= 2,
    }
    return [
        {
            "group": group,
            "n_patients": len(np.unique(np.asarray(patient)[mask])),
            **grade_metrics(np.asarray(y)[mask], q[mask], threshold),
        }
        for group, mask in groups.items()
    ]


def boundary_metrics(y, qs, qf, threshold=0.5):
    target = np.asarray(y)[:, None] > np.arange(4)
    s, f = project(qs), project(qf)
    s_error, f_error = (s >= threshold) != target, (f >= threshold) != target
    rows = []
    for b in range(4):
        rows.append(
            {
                "boundary": b,
                "positive": int(target[:, b].sum()),
                "negative": int((~target[:, b]).sum()),
                "harmful_flip": int((~s_error[:, b] & f_error[:, b]).sum()),
                "beneficial_flip": int((s_error[:, b] & ~f_error[:, b]).sum()),
                "brier_single": float(np.mean((s[:, b] - target[:, b]) ** 2)),
                "brier_fusion": float(np.mean((f[:, b] - target[:, b]) ** 2)),
                "auc_single": float(roc_auc_score(target[:, b], s[:, b]))
                if len(np.unique(target[:, b])) == 2
                else None,
                "auc_fusion": float(roc_auc_score(target[:, b], f[:, b]))
                if len(np.unique(target[:, b])) == 2
                else None,
            }
        )
    return rows


def expert_boundary_diagnostics(y, q, threshold=0.5, patient=None):
    """Threshold event counts and ranking diagnostics, including silent boundaries."""
    y, q = np.asarray(y), project(q)
    rows = []
    for b in range(4):
        target, predicted = y > b, q[:, b] >= threshold
        positive, negative = int(target.sum()), int((~target).sum())
        tp = int((target & predicted).sum())
        fp = int((~target & predicted).sum())
        row = {
            "boundary": b,
            "positive": positive,
            "negative": negative,
            "predicted_positive": int(predicted.sum()),
            "tp": tp,
            "fp": fp,
            "fn": positive - tp,
            "tn": negative - fp,
            "recall": tp / positive if positive else None,
            "precision": tp / int(predicted.sum()) if predicted.any() else None,
            "auc": float(roc_auc_score(target, q[:, b])) if positive and negative else None,
            "average_precision": float(average_precision_score(target, q[:, b]))
            if positive
            else None,
            "brier": float(np.mean((q[:, b] - target) ** 2)),
            "p_min": float(q[:, b].min()),
            "p_max": float(q[:, b].max()),
            "p50": float(np.median(q[:, b])),
            "positive_p50": float(np.median(q[target, b])) if positive else None,
            "silent_at_threshold": positive > 0 and not predicted.any(),
        }
        if patient is not None:
            row["positive_patients"] = len(np.unique(np.asarray(patient)[target]))
        rows.append(row)
    return rows


def alpha_summary(alpha):
    return [
        {
            "boundary": b,
            "mean": float(alpha[:, b].mean()),
            "sd": float(alpha[:, b].std()),
            "p10": float(np.quantile(alpha[:, b], 0.1)),
            "p50": float(np.quantile(alpha[:, b], 0.5)),
            "p90": float(np.quantile(alpha[:, b], 0.9)),
            "fraction_ge_0_5": float(np.mean(alpha[:, b] >= 0.5)),
            "fraction_le_0_01": float(np.mean(alpha[:, b] <= 0.01)),
            "fraction_ge_0_99": float(np.mean(alpha[:, b] >= 0.99)),
        }
        for b in range(4)
    ]
