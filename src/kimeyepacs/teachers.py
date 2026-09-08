from pathlib import Path

import numpy as np

from .metrics import decode, monotonic_project_np


def bce_np(q: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(q, eps, 1.0 - eps)
    return -(target * np.log(q) + (1.0 - target) * np.log(1.0 - q))


def ordinal_target_np(y: np.ndarray) -> np.ndarray:
    return (y[:, None] >= np.arange(1, 5)[None, :]).astype(np.float32)


def loss_guided_teacher(q_single: np.ndarray, q_fusion: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = ordinal_target_np(y)
    return bce_np(q_fusion, t) - bce_np(q_single, t)


def counterfactual_teacher(q_single: np.ndarray, q_fusion: np.ndarray, y: np.ndarray) -> np.ndarray:
    q_base = monotonic_project_np(q_fusion)
    grade_base = decode(q_base, monotonic=False)
    err_base = (grade_base - y) ** 2
    out = np.zeros_like(q_fusion, dtype=np.float32)
    for b in range(4):
        q_cf = q_fusion.copy()
        q_cf[:, b] = q_single[:, b]
        q_cf = monotonic_project_np(q_cf)
        grade_cf = decode(q_cf, monotonic=False)
        err_cf = (grade_cf - y) ** 2
        out[:, b] = err_base - err_cf
    return out


def one_sided_alpha(signal: np.ndarray, tau: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    positive = np.maximum(signal, 0.0)
    pos = positive[positive > 0]
    scale = np.median(pos) if len(pos) else 1.0
    return 1.0 - np.exp(-positive / (tau * scale + eps))


def confidence(signal: np.ndarray, gamma: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    mag = np.abs(signal)
    pos = mag[mag > 0]
    scale = np.median(pos) if len(pos) else 1.0
    return 1.0 - np.exp(-mag / (gamma * scale + eps))


def load_pair(single_npz: str | Path, fusion_npz: str | Path):
    s = np.load(single_npz, allow_pickle=True)
    f = np.load(fusion_npz, allow_pickle=True)
    assert np.all(s["patient"] == f["patient"])
    assert np.all(s["side"] == f["side"])
    return s, f
