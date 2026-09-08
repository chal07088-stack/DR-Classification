from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .metrics import monotonic_project_np


def gate_features(q_single: np.ndarray, q_fusion: np.ndarray, y_fellow_pred: np.ndarray | None = None) -> np.ndarray:
    pred_single = (q_single >= 0.5).sum(axis=1)
    if y_fellow_pred is None:
        y_fellow_pred = pred_single
    pred_gap = np.abs(pred_single - y_fellow_pred)
    signed = pred_single - y_fellow_pred
    rows = []
    for b in range(4):
        qs = q_single[:, b]
        qf = q_fusion[:, b]
        rows.append(np.stack([
            qs,
            qf,
            np.abs(qs - qf),
            np.abs(qs - 0.5),
            np.abs(qf - 0.5),
            1.0 - 2.0 * np.abs(qs - 0.5),
            1.0 - 2.0 * np.abs(qf - 0.5),
            pred_gap / 4.0,
            signed / 4.0,
            np.full(len(qs), b / 3.0),
        ], axis=1))
    return np.concatenate(rows, axis=0).astype(np.float32)


class BoundaryGate(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def apply_alpha(q_single: np.ndarray, q_fusion: np.ndarray, alpha_flat: np.ndarray) -> np.ndarray:
    alpha = alpha_flat.reshape(4, len(q_single)).T
    q = q_fusion + alpha * (q_single - q_fusion)
    return monotonic_project_np(q)


def save_alpha(path: str | Path, alpha_flat: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, alpha_flat.astype(np.float32))
