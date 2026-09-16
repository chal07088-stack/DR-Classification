"""Leakage-safe training/evaluation engine for the Quick20 benchmark."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from drpair.runtime import file_hash, git_commit, save_json, seed_everything, seed_worker

from .clip_dr import CLIPDR, clip_dr_loss
from .quick_models import (
    ConvNeXtSingle,
    DRGCNNPaired,
    DRGCNNSingle,
    HuangPaired,
    HuangSingle,
    load_drgcnn_encoder,
    load_huang_encoder,
)

HUANG_MEAN = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
HUANG_STD = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "patient_id", "image_id", "target_eye", "fellow_eye", "target_grade",
        "fellow_grade", "gap", "split", "image_path", "fellow_path",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid Quick20 split schema in {path}")
    return rows


def _cached_path(cache_root: Path, original_path: str) -> Path:
    return cache_root / (Path(original_path).stem + ".jpg")


def build_transform(method: str, image_size: int, train: bool):
    matched_paper = method.startswith(("B1-", "B2-"))
    mean, std = (HUANG_MEAN, HUANG_STD) if matched_paper else (
        IMAGENET_MEAN, IMAGENET_STD
    )
    operations: list = []
    if train and matched_paper:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(
                image_size, scale=(0.87, 1.15), ratio=(0.7, 1.3), antialias=True
            ),
            transforms.RandomAffine(
                degrees=180, translate=(0.2, 0.2),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ])
    elif train and method == "B3":
        operations.extend([
            transforms.RandomResizedCrop(image_size, antialias=True),
            transforms.RandomHorizontalFlip(),
        ])
    elif train and method == "B4":
        operations.extend([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
        ])
    elif train:
        operations.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ])
    else:
        operations.append(transforms.Resize((image_size, image_size), antialias=True))
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)


class QuickEyeDataset(Dataset):
    def __init__(self, rows: list[dict], cache_root: Path, transform, paired: bool):
        self.rows = rows
        roots = cache_root if isinstance(cache_root, (list, tuple)) else [cache_root]
        self.cache_roots = [Path(root) for root in roots]
        self.transform, self.paired = transform, paired

    def __len__(self):
        return len(self.rows)

    def _load(self, path: str):
        candidates = [_cached_path(root, path) for root in self.cache_roots]
        cached = next((candidate for candidate in candidates if candidate.is_file()), None)
        if cached is None:
            raise FileNotFoundError(f"Preprocessed image missing: {candidates}")
        with Image.open(cached) as source:
            return self.transform(source.convert("RGB"))

    def __getitem__(self, index: int):
        row = self.rows[index]
        item = {
            "target": self._load(row["image_path"]),
            "label": int(row["target_grade"]),
            "index": index,
        }
        if self.paired:
            item["fellow"] = self._load(row["fellow_path"])
        return item


def make_loader(rows, cfg, train: bool, paired: bool, smoke: bool = False):
    selected = rows[: min(len(rows), cfg["smoke_samples"])] if smoke else rows
    dataset = QuickEyeDataset(
        selected, cfg.get("cache_roots", [cfg["cache_root"]]),
        build_transform(cfg["method"], cfg["image_size"], train), paired,
    )
    workers = 0 if smoke else cfg["workers"]
    options = {
        "batch_size": min(cfg["batch_size"], len(dataset)),
        "shuffle": train, "num_workers": workers,
        "pin_memory": torch.cuda.is_available(), "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(cfg["seed"]), "drop_last": False,
    }
    if train and cfg.get("dataset_balanced", False):
        counts = Counter(row["dataset"] for row in selected)
        weights = torch.as_tensor([1.0 / counts[row["dataset"]] for row in selected], dtype=torch.double)
        options["shuffle"] = False
        options["sampler"] = WeightedRandomSampler(weights, len(selected), replacement=True)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options)


def build_model(cfg: dict, pretrained: bool, phase_root: Path):
    method = cfg["method"]
    if method == "B0":
        return ConvNeXtSingle(pretrained), False
    if method == "B1-S":
        return HuangSingle(pretrained), False
    if method == "B1-P":
        checkpoint_path = phase_root / "B1-S" / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError("B1-P requires completed B1-S checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not checkpoint.get("complete"):
            raise RuntimeError("B1-S checkpoint is not marked complete")
        return HuangPaired(load_huang_encoder(checkpoint)), True
    if method == "B2-S":
        return DRGCNNSingle(pretrained), False
    if method == "B2-P":
        checkpoint_path = phase_root / "B2-S" / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError("B2-P requires completed B2-S checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not checkpoint.get("complete"):
            raise RuntimeError("B2-S checkpoint is not marked complete")
        return DRGCNNPaired(load_drgcnn_encoder(checkpoint)), True
    if method == "B3":
        return CLIPDR(pretrained), False
    raise ValueError(f"Unsupported Phase 2 method: {method}")


def _decode(method: str, output: torch.Tensor) -> torch.Tensor:
    return (
        output.argmax(dim=1)
        if method in {"B0", "B3"}
        else output.round().clamp(0, 4).long()
    )


def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, labels=list(range(5)), weights="quadratic")
    if not np.isfinite(qwk):
        qwk = None
    return {
        "n_eyes": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(5)), average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": None if qwk is None else float(qwk),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=list(range(5)), output_dict=True, zero_division=0
        ),
    }


@torch.inference_mode()
def evaluate(model, loader, cfg, device, include_outputs: bool = False):
    model.eval()
    labels, predictions, outputs = [], [], []
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    for batch in loader:
        target = batch["target"].to(device, non_blocking=True)
        fellow = batch.get("fellow")
        if fellow is not None:
            fellow = fellow.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            raw = model(target, fellow)
        predictions.extend(_decode(cfg["method"], raw).cpu().tolist())
        labels.extend(batch["label"].tolist())
        if include_outputs:
            outputs.extend(raw.float().cpu().tolist())
    return calculate_metrics(labels, predictions), labels, predictions, outputs


def _optimizer(model: nn.Module, cfg: dict):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(
            parameters, lr=cfg["learning_rate"], momentum=cfg["momentum"],
            nesterov=True, weight_decay=cfg["weight_decay"],
        )
    if cfg["optimizer"] == "radam":
        return torch.optim.RAdam(
            parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
    return torch.optim.AdamW(
        parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )


def train_method(cfg: dict, phase_root: Path, smoke: bool = False, pretrained: bool = True):
    seed_everything(cfg["seed"], deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and device.type != "cuda":
        raise RuntimeError("Full benchmark training requires CUDA")
    model, paired = build_model(cfg, pretrained, phase_root)
    model.to(device)
    train_rows, val_rows, test_rows = (
        read_rows(cfg["train_csv"]), read_rows(cfg["val_csv"]), read_rows(cfg["test_csv"])
    )
    train_loader = make_loader(train_rows, cfg, True, paired, smoke)
    val_loader = make_loader(val_rows, cfg, False, paired, smoke)
    optimizer = _optimizer(model, cfg)
    accumulation = max(1, cfg["effective_batch_size"] // cfg["batch_size"])
    total_epochs = 1 if smoke else cfg["max_epochs"]
    updates = max(1, total_epochs * ((len(train_loader) + accumulation - 1) // accumulation))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates)
        if cfg["schedule"] == "cosine" else None
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported()
    )
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    run_dir = phase_root / cfg["method"]
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training.jsonl"
    if not smoke and history_path.exists():
        history_path.unlink()
    best_qwk, best_epoch, stale, best_state = -float("inf"), 0, 0, None
    started, peak_gib = time.perf_counter(), 0.0
    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            target = batch["target"].to(device, non_blocking=True)
            fellow = batch.get("fellow")
            if fellow is not None:
                fellow = fellow.to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                output = model(target, fellow)
                if cfg["method"] == "B3":
                    loss, _ = clip_dr_loss(output.float(), label)
                elif cfg["method"] == "B0":
                    loss = nn.functional.cross_entropy(output.float(), label)
                else:
                    loss = nn.functional.mse_loss(output.float(), label.float())
            scaler.scale(loss / accumulation).backward()
            running_loss += float(loss.detach()) * len(label)
            seen += len(label)
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        validation, _, _, _ = evaluate(model, val_loader, cfg, device)
        score = validation["qwk"] if validation["qwk"] is not None else -float("inf")
        elapsed = time.perf_counter() - started
        if device.type == "cuda":
            peak_gib = max(peak_gib, torch.cuda.max_memory_reserved() / 1024**3)
        record = {
            "epoch": epoch, "train_loss": running_loss / seen,
            "validation": validation, "elapsed_seconds": elapsed,
            "peak_vram_gib": peak_gib, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"[{cfg['method']}] epoch {epoch}/{total_epochs} "
            f"loss={record['train_loss']:.5f} val_qwk={validation['qwk']} "
            f"elapsed={elapsed/60:.1f}m peak={peak_gib:.2f}GiB", flush=True,
        )
        # Always retain epoch 1 even if its tiny smoke-validation QWK is undefined.
        if best_state is None or score > best_qwk + 1e-12:
            best_qwk, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if not smoke and stale >= cfg["patience"]:
            print(f"[{cfg['method']}] early stop after {stale} stale epochs", flush=True)
            break
    if best_state is None:
        raise RuntimeError(f"{cfg['method']} produced no valid validation checkpoint")
    if smoke:
        return {"method": cfg["method"], "smoke": "passed", "validation": validation}
    checkpoint = {
        "complete": True, "method": cfg["method"], "best_epoch": best_epoch,
        "validation_qwk": best_qwk, "model_state": best_state,
    }
    torch.save(checkpoint, run_dir / "best.pt")
    model.load_state_dict(best_state)
    # The test split is first loaded only after validation checkpoint selection.
    test_loader = make_loader(test_rows, cfg, False, paired, False)
    test_metrics, labels, predictions, outputs = evaluate(
        model, test_loader, cfg, device, True
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
        for row, label, prediction, output in zip(test_rows, labels, predictions, outputs):
            writer.writerow({
                "patient_id": row["patient_id"], "target_eye": row["target_eye"],
                "fellow_eye": row["fellow_eye"] if paired else "",
                "target_grade": label, "fellow_grade": row["fellow_grade"],
                "predicted_grade": prediction, "dataset": cfg.get("dataset_name", "EyePACS"), "split": "test",
                "method": cfg["method"],
                "native_score": output if cfg["method"] not in {"B0", "B3"} else "",
                "native_logits": json.dumps(output) if cfg["method"] in {"B0", "B3"} else "",
                "native_output_type": model.output_type,
            })
    summary = {
        "status": "complete", "method": cfg["method"],
        "reproduction_status": (
            "Protocol-adapted official implementation"
            if cfg["method"].startswith(("B1-", "B2-")) or cfg["method"] == "B3"
            else "Reimplemented from paper"
        ),
        "seed": cfg["seed"], "best_epoch": best_epoch, "epochs_completed": epoch,
        "training_seconds": time.perf_counter() - started, "peak_vram_gib": peak_gib,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "validation_qwk": best_qwk, "test": test_metrics,
        "prediction_sha256": file_hash(prediction_path),
        "git_commit_at_run": git_commit(), "config": cfg,
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def load_toml_config(path: str | Path) -> dict:
    import tomllib

    path = Path(path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    cfg = dict(config["benchmark"])
    root = path.parents[2]
    for key in ("train_csv", "val_csv", "test_csv", "cache_root"):
        value = Path(cfg[key])
        cfg[key] = str(value if value.is_absolute() else root / value)
    return cfg


def alignment_key(row: dict) -> tuple[str, str]:
    return row["patient_id"], row["target_eye"]


def bilateral_analysis(
    phase_root: Path,
    single_method: str,
    paired_method: str,
    comparison_name: str,
    file_prefix: str,
    report_name: str,
    report_title: str,
):
    def load_predictions(method: str):
        path = phase_root / method / "predictions_test.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return {alignment_key(row): row for row in rows}

    single, paired = load_predictions(single_method), load_predictions(paired_method)
    if single.keys() != paired.keys():
        raise RuntimeError(
            f"{single_method}/{paired_method} target-eye prediction alignment failed"
        )
    groups = {"All": [], "gap=0": [], "gap=1": [], "gap>=2": []}
    samples = []
    for key in sorted(single):
        s, p = single[key], paired[key]
        true = int(s["target_grade"])
        gap = abs(true - int(s["fellow_grade"]))
        delta = (
            abs(true - int(s["predicted_grade"]))
            - abs(true - int(p["predicted_grade"]))
        )
        effect = "Benefit" if delta > 0 else "Harm" if delta < 0 else "Neutral"
        record = {
            "patient_id": key[0], "target_eye": key[1], "target_grade": true,
            "fellow_grade": int(s["fellow_grade"]), "gap": gap,
            "single_prediction": int(s["predicted_grade"]),
            "paired_prediction": int(p["predicted_grade"]), "delta_error": delta,
            "effect": effect,
        }
        samples.append(record)
        groups["All"].append(record)
        groups["gap=0" if gap == 0 else "gap=1" if gap == 1 else "gap>=2"].append(record)
    table = []
    for group, rows in groups.items():
        counts, n = Counter(row["effect"] for row in rows), len(rows)
        table.append({
            "Method Pair": comparison_name, "Group": group, "N": n,
            "Benefit count": counts["Benefit"], "Benefit %": 100 * counts["Benefit"] / n,
            "Neutral count": counts["Neutral"], "Neutral %": 100 * counts["Neutral"] / n,
            "Harm count": counts["Harm"], "Harm %": 100 * counts["Harm"] / n,
            "Mean Delta Error": float(np.mean([row["delta_error"] for row in rows])),
            "Median Delta Error": float(np.median([row["delta_error"] for row in rows])),
        })
    results_root = phase_root.parents[1]
    analysis_dir, tables_dir = results_root / "analysis", results_root / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (analysis_dir / f"{file_prefix}_bilateral_benefit_harm_samples.csv", samples),
        (tables_dir / f"{file_prefix}_benefit_harm.csv", table),
    ):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    s_summary = json.loads(
        (phase_root / single_method / "summary.json").read_text(encoding="utf-8")
    )
    p_summary = json.loads(
        (phase_root / paired_method / "summary.json").read_text(encoding="utf-8")
    )
    delta_qwk = p_summary["test"]["qwk"] - s_summary["test"]["qwk"]
    overall = table[0]
    report = (
        f"# {report_title}\n\n"
        "Prediction alignment: **PASS** (same patient/target-eye keys).\n\n"
        f"- {single_method} test QWK: {s_summary['test']['qwk']:.6f}\n"
        f"- {paired_method} test QWK: {p_summary['test']['qwk']:.6f}\n"
        f"- Delta QWK ({paired_method} − {single_method}): {delta_qwk:+.6f}\n"
        f"- Benefit: {overall['Benefit %']:.2f}%\n"
        f"- Neutral: {overall['Neutral %']:.2f}%\n"
        f"- Harm: {overall['Harm %']:.2f}%\n\n"
        "These are one-seed capped-budget pilot estimates, not definitive superiority claims.\n"
    )
    (results_root / report_name).write_text(report, encoding="utf-8")
    return {"delta_qwk": delta_qwk, "benefit_harm": table}


def phase2_analysis(phase_root: Path):
    return bilateral_analysis(
        phase_root, "B1-S", "B1-P", "Huang PFF vs Huang Single",
        "phase2_huang", "PHASE2_HUANG.md", "PHASE 2 — Huang controlled pair",
    )


def phase3_analysis(phase_root: Path):
    return bilateral_analysis(
        phase_root, "B2-S", "B2-P", "DRGCNN BFFN vs DRGCNN Single",
        "phase3_drgcnn", "PHASE3_DRGCNN.md", "PHASE 3 — DRGCNN controlled pair",
    )
