import numpy as np
import torch
from torch.nn import functional as F

from drpair.runtime import seed_everything

from .network import RollbackGate, project_torch


def fit_gate(features, qs, qf, y, target, cfg, seed, mode="teacher", global_gate=False):
    seed_everything(seed)
    # Small cached-probability models stay on CPU; no image expert is retrained.
    model = RollbackGate(features.shape[1], cfg["hidden"], global_gate)
    x, s, f = [torch.tensor(v, dtype=torch.float32) for v in (features, qs, qf)]
    t = torch.tensor(np.asarray(y)[:, None] > np.arange(4), dtype=torch.float32)
    a = torch.tensor(target, dtype=torch.float32)
    with torch.no_grad():
        model.mean.copy_(x.mean(0))
        model.std.copy_(x.std(0, unbiased=False).clamp_min(1e-5))
        positives = t.sum(0)
        support = torch.minimum(positives, len(t) - positives)
        scale = support / (support + cfg["rare_boundary_prior"])
        model.support_scale.copy_(scale.mean()[None] if global_gate else scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    model.train()
    history = []
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), cfg["batch"]):
            idx = order[start : start + cfg["batch"]]
            alpha = model(x[idx])
            q = project_torch(f[idx] + alpha * (s[idx] - f[idx])).clamp(1e-6, 1 - 1e-6)
            if mode == "teacher":
                supervision = a[idx].mean(1, keepdim=True).expand(-1, 4) if global_gate else a[idx]
                loss = F.mse_loss(alpha, supervision)
            elif mode == "task":
                loss = F.binary_cross_entropy(q, t[idx])
            else:
                raise ValueError(mode)
            loss = loss + cfg["intervention_penalty"] * alpha.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(idx)
        history.append({"epoch": epoch + 1, "loss": total / len(x)})
    model.eval()
    return model, history


@torch.inference_mode()
def predict_gate(model, features):
    model.eval()
    return model(torch.tensor(features, dtype=torch.float32)).numpy()
