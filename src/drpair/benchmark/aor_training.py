"""Capped-budget training and native diffusion decoding for AOR-DR."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch import nn

from drpair.runtime import file_hash, git_commit, save_json, seed_everything

from .aor_dr import AORDR
from .quick_training import calculate_metrics, make_loader, read_rows


@torch.inference_mode()
def evaluate_aor(model, loader, device, seed: int, include_outputs: bool = False):
    model.eval()
    labels, predictions, outputs = [], [], []
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for batch in loader:
            image = batch["target"].to(device, non_blocking=True)
            boundaries = model.sample(image)
            predicted = boundaries.round().clamp(0, 1).sum(dim=1).long()
            labels.extend(batch["label"].tolist())
            predictions.extend(predicted.cpu().tolist())
            if include_outputs:
                outputs.extend(boundaries.float().cpu().tolist())
    return calculate_metrics(labels, predictions), labels, predictions, outputs


def train_aor(cfg: dict, phase_root: Path, smoke: bool = False, pretrained: bool = True):
    seed_everything(cfg["seed"], deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and device.type != "cuda":
        raise RuntimeError("Full AOR-DR benchmark training requires CUDA")
    model = AORDR(
        pretrained=pretrained,
        diffusion_batch_mul=cfg["diffusion_batch_mul"],
        sampling_steps=str(cfg["sampling_steps"]),
    ).to(device)
    train_rows, val_rows, test_rows = (
        read_rows(cfg["train_csv"]), read_rows(cfg["val_csv"]), read_rows(cfg["test_csv"])
    )
    train_loader = make_loader(train_rows, cfg, True, False, smoke)
    val_loader = make_loader(val_rows, cfg, False, False, smoke)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["scheduler_patience"]
    )
    accumulation = max(1, cfg["effective_batch_size"] // cfg["batch_size"])
    epochs = 1 if smoke else cfg["max_epochs"]
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    run_dir = phase_root / "B4"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training.jsonl"
    if not smoke and history_path.exists():
        history_path.unlink()
    best_score, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    started, peak_gib = time.perf_counter(), 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            image = batch["target"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                loss = model.training_loss(image, labels)
            scaler.scale(loss / accumulation).backward()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        validation, _, _, _ = evaluate_aor(
            model, val_loader, device, cfg["seed"] + 10_000
        )
        score = validation["qwk"] if validation["qwk"] is not None else -float("inf")
        scheduler.step(validation["mae"])
        if device.type == "cuda":
            peak_gib = max(peak_gib, torch.cuda.max_memory_reserved() / 1024**3)
        record = {
            "epoch": epoch, "train_loss": total_loss / seen, "validation": validation,
            "elapsed_seconds": time.perf_counter() - started, "peak_vram_gib": peak_gib,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"[B4] epoch {epoch}/{epochs} loss={record['train_loss']:.5f} "
            f"val_qwk={validation['qwk']} elapsed={record['elapsed_seconds']/60:.1f}m",
            flush=True,
        )
        if best_state is None or score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if not smoke and stale >= cfg["patience"]:
            break
    if smoke:
        return {"method": "B4", "smoke": "passed", "validation": validation}
    torch.save(
        {"complete": True, "method": "B4", "best_epoch": best_epoch,
         "validation_qwk": best_score, "model_state": best_state},
        run_dir / "best.pt",
    )
    model.load_state_dict(best_state)
    test_loader = make_loader(test_rows, cfg, False, False, False)
    test_metrics, labels, predictions, outputs = evaluate_aor(
        model, test_loader, device, cfg["seed"] + 20_000, True
    )
    prediction_path = run_dir / "predictions_test.csv"
    fields = [
        "patient_id", "target_eye", "fellow_eye", "target_grade", "fellow_grade",
        "predicted_grade", "dataset", "split", "method", "native_score",
        "native_logits", "native_output_type",
    ]
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, label, prediction, boundaries in zip(test_rows, labels, predictions, outputs):
            writer.writerow({
                "patient_id": row["patient_id"], "target_eye": row["target_eye"],
                "fellow_eye": "", "target_grade": label,
                "fellow_grade": row["fellow_grade"], "predicted_grade": prediction,
                "dataset": cfg.get("dataset_name", "EyePACS"), "split": "test", "method": "B4",
                "native_score": json.dumps(boundaries), "native_logits": "",
                "native_output_type": model.output_type,
            })
    summary = {
        "status": "complete", "method": "B4",
        "reproduction_status": "Protocol-adapted official implementation",
        "seed": cfg["seed"], "best_epoch": best_epoch, "epochs_completed": epoch,
        "training_seconds": time.perf_counter() - started, "peak_vram_gib": peak_gib,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "validation_qwk": best_score, "test": test_metrics,
        "prediction_sha256": file_hash(prediction_path),
        "git_commit_at_run": git_commit(), "config": cfg,
    }
    save_json(run_dir / "summary.json", summary)
    return summary
