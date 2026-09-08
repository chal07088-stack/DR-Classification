import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from drpair.evaluation.metrics import grade_metrics
from drpair.models.ordinal import ordinal_targets
from drpair.runtime import append_jsonl, save_json

from .ema import EMA, cpu_state
from .predictions import forward, predict


def fit(
    model,
    train_loader,
    cfg,
    device,
    dtype,
    paired,
    checkpoint,
    log,
    validation_loader=None,
    epochs=None,
):
    """Fixed-epoch full refit or inner-training validation selection.

    Outer OOF holdout and official validation never select expert checkpoints.
    """
    checkpoint = Path(checkpoint)
    model.to(device)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        return model, saved["epoch"]
    opt_cfg = cfg["training"]
    save_json(
        checkpoint.with_suffix(".json"),
        {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_trainable_parameters": sum(
                p.numel() for p in model.encoder.parameters() if p.requires_grad
            ),
            "paired": paired,
            "batch_unit": "target eyes (paired jobs also load their fellow eyes)",
        },
    )
    epochs = epochs or opt_cfg["epochs"]
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": opt_cfg["backbone_lr"]},
            {"params": head_params, "lr": opt_cfg["head_lr"]},
        ],
        weight_decay=opt_cfg["weight_decay"],
    )
    physical = train_loader.batch_size
    accum = opt_cfg["effective_batch"] // physical
    updates_per_epoch = math.ceil(len(train_loader) / accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * updates_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    ema = EMA(model, opt_cfg["ema_decay"]) if opt_cfg["ema_decay"] else None
    best_score, selected_epoch = -float("inf"), epochs
    best_state = None
    updates = 0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0.0, 0
        window_samples = 0
        for step, batch in enumerate(train_loader, 1):
            labels = batch["label"].to(device)
            n = len(labels)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                logits = forward(model, batch, device, paired)
                loss = F.binary_cross_entropy_with_logits(logits.float(), ordinal_targets(labels))
            # Accumulate sample sums, then divide by actual window size (including the tail).
            scaler.scale(loss * n).backward()
            total_loss += loss.item() * n
            seen += n
            window_samples += n
            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(window_samples)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt_cfg["grad_clip"], error_if_nonfinite=True
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    updates += 1
                    if ema:
                        ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                window_samples = 0
        record = {
            "epoch": epoch,
            "loss": total_loss / seen,
            "samples": seen,
            "updates": updates,
            "seconds": time.perf_counter() - start,
        }
        with ema.applied(model) if ema else nullcontext():
            if validation_loader is not None:
                q = predict(model, validation_loader, device, dtype, paired)
                y = np.array([r["label"] for r in validation_loader.dataset.rows])
                metrics = grade_metrics(y, q)
                score = metrics["qwk"]
                record["inner_validation_qwk"] = score
                if score is not None and score > best_score:
                    best_score, selected_epoch = score, epoch
                    best_state = cpu_state(model)
            elif epoch == epochs:
                best_state = cpu_state(model)
        append_jsonl(log, record)
        print(
            f"[{checkpoint.stem}] epoch {epoch}/{epochs} loss={record['loss']:.5f} "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("No valid evaluated checkpoint was produced")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(
        {
            "model": best_state,
            "epoch": selected_epoch,
            "weights": "ema" if ema else "raw",
            "paired": paired,
            "selection": "inner_validation_qwk" if validation_loader else "fixed_epoch_refit",
        },
        temporary,
    )
    temporary.replace(checkpoint)
    model.load_state_dict(best_state)
    return model, selected_epoch
