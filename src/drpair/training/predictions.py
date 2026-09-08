from pathlib import Path

import numpy as np
import torch

from drpair.evaluation.ordinal import check_probabilities


def forward(model, batch, device, paired):
    target = batch["target"].to(device, non_blocking=True)
    if paired:
        return model(target, batch["fellow"].to(device, non_blocking=True))
    return model(target)


@torch.inference_mode()
def predict(model, loader, device, dtype, paired):
    model.eval()
    q = np.full((len(loader.dataset), 4), np.nan, dtype=np.float32)
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            logits = forward(model, batch, device, paired)
        q[batch["index"].numpy()] = logits.float().sigmoid().cpu().numpy()
    check_probabilities(q)
    return q


def save_predictions(path, rows, q, fold=None, seed=42):
    check_probabilities(q)
    if len(rows) != len(q):
        raise ValueError("Prediction/manifest length mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        q=q,
        patient=np.array([r["patient"] for r in rows]),
        side=np.array([r["side"] for r in rows]),
        y=np.array([r["label"] for r in rows]),
        fellow_y=np.array([r["fellow_label"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        fold=np.full(len(rows), -1) if fold is None else fold,
        seed=np.full(len(rows), seed),
    )


def load_pair(single_path, paired_path):
    with np.load(single_path, allow_pickle=False) as source:
        s = {k: source[k] for k in source.files}
    with np.load(paired_path, allow_pickle=False) as source:
        f = {k: source[k] for k in source.files}
    for key in ("patient", "side", "y", "fellow_y", "split", "fold", "seed"):
        if not np.array_equal(s[key], f[key]):
            raise ValueError(f"Single/paired alignment mismatch: {key}")
    check_probabilities(s["q"])
    check_probabilities(f["q"])
    return s, f
