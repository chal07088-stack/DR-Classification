from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)


def classes(raw: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(raw), 0, 4).astype(int)


def metric_dict(y: np.ndarray, raw: np.ndarray) -> dict[str, float]:
    pred = classes(raw)
    return {
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "mae": float(mean_absolute_error(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, average="macro", labels=list(range(5)), zero_division=0)
        ),
    }


def write_diagnostics(
    root: Path, dataset: str, method: str, seed: int, split: str, y: np.ndarray, raw: np.ndarray
) -> None:
    pred = classes(raw)
    target = root / "outputs" / "metrics" / dataset / method
    target.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y, pred, labels=list(range(5)))
    np.savetxt(target / f"seed{seed}_{split}_confusion.csv", matrix, delimiter=",", fmt="%d")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(5)), zero_division=0
    )
    rows = ["grade,precision,recall,f1,support"] + [
        f"{g},{precision[g]},{recall[g]},{f1[g]},{support[g]}" for g in range(5)
    ]
    (target / f"seed{seed}_{split}_classwise.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (target / f"seed{seed}_{split}.json").write_text(
        json.dumps(metric_dict(y, raw), indent=2), encoding="utf-8"
    )
