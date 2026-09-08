from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error, roc_auc_score


def monotonic_project_np(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 0.0, 1.0)
    return np.minimum.accumulate(q, axis=1)


def decode(q: np.ndarray, threshold: float = 0.5, monotonic: bool = True) -> np.ndarray:
    if monotonic:
        q = monotonic_project_np(q)
    return (q >= threshold).sum(axis=1).astype(int)


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3, 4]))


def safe_auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, score))


def grade_metrics(y_true: np.ndarray, q: np.ndarray) -> Dict[str, float]:
    pred = decode(q)
    out = {
        "qwk": qwk(y_true, pred),
        "macro_f1": float(f1_score(y_true, pred, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "pm1_accuracy": float(np.mean(np.abs(y_true - pred) <= 1)),
        "under_rate": float(np.mean(pred < y_true)),
        "severe_under_rate": float(np.mean((y_true >= 3) & (pred <= 2))),
        "ge2_error_rate": float(np.mean(np.abs(y_true - pred) >= 2)),
    }
    for b in range(4):
        yb = (y_true >= b + 1).astype(int)
        out[f"b{b}_auc"] = safe_auc(yb, q[:, b])
    return out


def subgroup_metrics(y_true: np.ndarray, q: np.ndarray, gap: np.ndarray, prefix: str) -> Dict[str, float]:
    out = {}
    groups = {
        "all": np.ones_like(y_true, dtype=bool),
        "gap0": gap == 0,
        "gap_ge1": gap >= 1,
        "gap_ge2": gap >= 2,
    }
    for name, mask in groups.items():
        if mask.sum() < 20 or len(np.unique(y_true[mask])) < 2:
            continue
        for k, v in grade_metrics(y_true[mask], q[mask]).items():
            out[f"{prefix}_{name}_{k}"] = v
        out[f"{prefix}_{name}_n"] = int(mask.sum())
    return out


def save_npz(path: str | Path, items: List, q: np.ndarray, indices: np.ndarray) -> None:
    selected = [items[int(i)] for i in indices]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        q=q.astype(np.float32),
        y=np.asarray([x.label for x in selected], dtype=np.int64),
        patient=np.asarray([x.patient for x in selected]),
        side=np.asarray([x.side for x in selected]),
        fellow_y=np.asarray([x.fellow_label for x in selected], dtype=np.int64),
        gap=np.asarray([abs(x.label - x.fellow_label) for x in selected], dtype=np.int64),
        path=np.asarray([x.path for x in selected]),
        fellow_path=np.asarray([x.fellow_path for x in selected]),
    )
