import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import grade_metrics, save_npz, subgroup_metrics
from .runtime import append_jsonl


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
        self.backup = {}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            state = model.state_dict()
            for k, v in state.items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for k, v in self.shadow.items():
            self.backup[k] = state[k].detach().clone()
            state[k].copy_(v)

    def restore(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.backup.items():
            state[k].copy_(v)
        self.backup = {}


def make_loader(dataset, batch_size: int, train: bool, workers: int, pin_memory: bool,
                persistent_workers: bool, prefetch_factor: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": workers,
        "drop_last": train,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def criterion_from_pos_weight(pos_weight: torch.Tensor | None, device: torch.device) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))


def _autocast(device: torch.device, amp_dtype):
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None))


def infer_single(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, indices = [], []
    with torch.no_grad():
        for x, _, _, idx in loader:
            with _autocast(device, amp_dtype):
                logits = model(x.to(device, non_blocking=True))
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            indices.append(idx.numpy())
    return np.concatenate(probs), np.concatenate(indices)


def infer_paired(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, indices = [], []
    with torch.no_grad():
        for x, fellow, _, _, idx in loader:
            with _autocast(device, amp_dtype):
                logits = model(x.to(device, non_blocking=True), fellow.to(device, non_blocking=True))
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            indices.append(idx.numpy())
    return np.concatenate(probs), np.concatenate(indices)


def train_ordinal(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_items,
    device: torch.device,
    amp_dtype,
    pos_weight: torch.Tensor | None,
    epochs: int,
    min_epochs: int,
    patience: int,
    min_save_epoch: int,
    lr_groups: Iterable[Dict],
    weight_decay: float,
    grad_accum: int,
    grad_clip: float,
    ema_decay: float,
    log_path: Path,
    tag: str,
    paired: bool = False,
) -> tuple[nn.Module, Dict[str, float]]:
    model.to(device)
    crit = criterion_from_pos_weight(pos_weight, device)
    opt = torch.optim.AdamW(lr_groups, weight_decay=weight_decay)
    total_updates = max(1, epochs * max(1, len(train_loader) // max(1, grad_accum)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    ema = EMA(model, ema_decay) if ema_decay and ema_decay > 0 else None

    best_score = -1e9
    best_state = None
    best_metrics = {}
    bad_epochs = 0
    update_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        if paired:
            model.encoder.eval()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        batches = 0
        for step, batch in enumerate(train_loader, start=1):
            if paired:
                x, fellow, y, _, _ = batch
                args = (x.to(device, non_blocking=True), fellow.to(device, non_blocking=True))
            else:
                x, y, _, _ = batch
                args = (x.to(device, non_blocking=True),)
            y = y.to(device, non_blocking=True)
            with _autocast(device, amp_dtype):
                logits = model(*args)
                loss = crit(logits, y) / grad_accum
            scaler.scale(loss).backward()
            running += float(loss.detach().cpu()) * grad_accum
            batches += 1
            if step % grad_accum == 0:
                scaler.unscale_(opt)
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                update_step += 1
                if ema:
                    ema.update(model)

        if ema:
            ema.apply(model)
        q_val, idx_val = infer_paired(model, val_loader, device, amp_dtype) if paired else infer_single(model, val_loader, device, amp_dtype)
        if ema:
            ema.restore(model)

        y_val = np.asarray([val_items[int(i)].label for i in idx_val], dtype=np.int64)
        gap = np.asarray([abs(val_items[int(i)].label - val_items[int(i)].fellow_label) for i in idx_val], dtype=np.int64)
        metrics = subgroup_metrics(y_val, q_val, gap, "val")
        score = metrics.get("val_all_qwk", -1e9)
        row = {
            "tag": tag,
            "epoch": epoch,
            "loss": running / max(1, batches),
            "score": score,
            "updates": update_step,
            "seconds": time.time() - t0,
            **metrics,
        }
        append_jsonl(log_path, row)
        print(f"[{tag}] epoch {epoch:03d} loss {row['loss']:.5f} val_qwk {score:.5f}", flush=True)

        if epoch >= min_save_epoch and score > best_score:
            best_score = score
            best_metrics = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        elif epoch >= min_epochs:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def dump_predictions(model: nn.Module, loaders: Dict[str, DataLoader], split_items: Dict[str, list],
                     device: torch.device, amp_dtype, out_dir: Path, tag: str, paired: bool) -> Dict[str, Dict[str, float]]:
    out = {}
    for split, loader in loaders.items():
        q, idx = infer_paired(model, loader, device, amp_dtype) if paired else infer_single(model, loader, device, amp_dtype)
        save_npz(out_dir / f"{tag}_{split}.npz", split_items[split], q, idx)
        y = np.asarray([split_items[split][int(i)].label for i in idx], dtype=np.int64)
        gap = np.asarray([abs(split_items[split][int(i)].label - split_items[split][int(i)].fellow_label) for i in idx], dtype=np.int64)
        out[split] = subgroup_metrics(y, q, gap, f"{tag}_{split}")
    return out
