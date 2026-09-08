import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import (
    PairedEyeDataset,
    SingleEyeDataset,
    boundary_pos_weight,
    class_counts,
    limit_patients,
    scan_all,
)
from .engine import dump_predictions, make_loader, train_ordinal
from .models import PairedOrdinal, SingleOrdinal, total_param_count, trainable_param_count
from .runtime import configure_torch, device_and_amp, ensure_dir, load_config, save_json, seed_everything


def _items_for_run(cfg, seed: int, max_train_patients, max_val_patients, max_test_patients):
    items = scan_all(cfg["data_root"])
    items["train"] = limit_patients(items["train"], max_train_patients, seed)
    items["val"] = limit_patients(items["val"], max_val_patients, seed + 1)
    items["test"] = limit_patients(items["test"], max_test_patients, seed + 2)
    return items


def _loader_cfg(cfg):
    r = cfg["runtime"]
    return {
        "workers": int(r.get("num_workers", 4)),
        "pin_memory": bool(r.get("pin_memory", True)),
        "persistent_workers": bool(r.get("persistent_workers", True)),
        "prefetch_factor": int(r.get("prefetch_factor", 2)),
    }


def _save_summary(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--paired-epochs", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--max-train-patients", type=int, default=None)
    ap.add_argument("--max-val-patients", type=int, default=None)
    ap.add_argument("--max-test-patients", type=int, default=None)
    ap.add_argument("--pretrained", choices=["true", "false"], default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.image_size:
        cfg["model"]["image_size"] = args.image_size
    if args.epochs:
        cfg["optimization"]["single_epochs"] = args.epochs
    if args.paired_epochs:
        cfg["optimization"]["paired_epochs"] = args.paired_epochs
    if args.pretrained is not None:
        cfg["model"]["pretrained"] = args.pretrained == "true"
    if args.num_workers is not None:
        cfg["runtime"]["num_workers"] = args.num_workers
        cfg["runtime"]["persistent_workers"] = args.num_workers > 0

    configure_torch(cfg)
    seed_everything(args.seed)
    device, amp_dtype = device_and_amp(cfg["runtime"].get("precision", "bf16"))

    run_name = args.run_name or f"seed{args.seed}_experts"
    run_dir = ensure_dir(Path(cfg["out_root"]) / run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    pred_dir = ensure_dir(run_dir / "predictions")
    metric_dir = ensure_dir(run_dir / "metrics")
    log_dir = ensure_dir(run_dir / "logs")
    save_json(run_dir / "resolved_config.json", cfg)
    shutil.copy2(args.config, run_dir / "source_config.yaml")

    items = _items_for_run(cfg, args.seed, args.max_train_patients, args.max_val_patients, args.max_test_patients)
    split_counts = {
        split: {"eyes": len(v), "patients": len(v) // 2, "class_counts": class_counts(v)}
        for split, v in items.items()
    }
    save_json(run_dir / "dataset_summary.json", split_counts)
    print(f"run_dir={run_dir}", flush=True)
    print(f"device={device} amp_dtype={amp_dtype}", flush=True)
    print(f"split_counts={split_counts}", flush=True)

    model_cfg = cfg["model"]
    opt = cfg["optimization"]
    rt = _loader_cfg(cfg)
    image_size = int(model_cfg["image_size"])
    color_mode = model_cfg.get("color_mode", "grayscale_reference")
    hflip = bool(cfg.get("augmentation", {}).get("horizontal_flip", True))

    single_sets = {
        s: SingleEyeDataset(items[s], image_size, s == "train", hflip, color_mode)
        for s in ("train", "val", "test")
    }
    single_loaders = {
        s: make_loader(single_sets[s], int(opt["single_batch"]), s == "train", **rt)
        for s in ("train", "val", "test")
    }
    pos_weight = None
    if opt.get("boundary_pos_weight", True):
        pos_weight = boundary_pos_weight(items["train"], float(opt.get("max_pos_weight", 20.0)))

    single = SingleOrdinal(model_cfg["arch"], bool(model_cfg.get("pretrained", True)))
    print(
        f"single params total={total_param_count(single):,} trainable={trainable_param_count(single):,}",
        flush=True,
    )
    single_groups = [
        {"params": single.encoder.parameters(), "lr": float(opt["backbone_lr"])},
        {"params": single.head.parameters(), "lr": float(opt["head_lr"])},
    ]
    single, single_best = train_ordinal(
        single,
        single_loaders["train"],
        single_loaders["val"],
        items["val"],
        device,
        amp_dtype,
        pos_weight,
        int(opt["single_epochs"]),
        int(opt["min_epochs"]),
        int(opt["patience"]),
        int(opt["min_save_epoch"]),
        single_groups,
        float(opt["weight_decay"]),
        int(opt["single_grad_accum"]),
        float(cfg["runtime"].get("grad_clip", 1.0)),
        float(opt.get("ema_decay", 0.0)),
        log_dir / "single_train.jsonl",
        "single",
        paired=False,
    )
    torch.save(single.state_dict(), ckpt_dir / "single_best.pt")
    single_metrics = dump_predictions(single, single_loaders, items, device, amp_dtype, pred_dir, "single", paired=False)

    paired_sets = {
        s: PairedEyeDataset(items[s], image_size, s == "train", hflip, color_mode)
        for s in ("train", "val", "test")
    }
    paired_loaders = {
        s: make_loader(paired_sets[s], int(opt["paired_batch"]), s == "train", **rt)
        for s in ("train", "val", "test")
    }
    paired = PairedOrdinal(single.encoder, single.num_features)
    print(
        f"paired params total={total_param_count(paired):,} trainable={trainable_param_count(paired):,}",
        flush=True,
    )
    paired_groups = [{"params": paired.head.parameters(), "lr": float(opt["paired_head_lr"])}]
    paired, paired_best = train_ordinal(
        paired,
        paired_loaders["train"],
        paired_loaders["val"],
        items["val"],
        device,
        amp_dtype,
        pos_weight,
        int(opt["paired_epochs"]),
        int(opt["min_epochs"]),
        int(opt["patience"]),
        int(opt["min_save_epoch"]),
        paired_groups,
        float(opt["weight_decay"]),
        int(opt["paired_grad_accum"]),
        float(cfg["runtime"].get("grad_clip", 1.0)),
        float(opt.get("ema_decay", 0.0)),
        log_dir / "paired_train.jsonl",
        "paired",
        paired=True,
    )
    torch.save(paired.state_dict(), ckpt_dir / "paired_best.pt")
    paired_metrics = dump_predictions(paired, paired_loaders, items, device, amp_dtype, pred_dir, "paired", paired=True)

    rows = []
    rows.append({"arm": "A_single", "best_val_qwk": single_best.get("score", ""), **single_metrics["test"]})
    rows.append({"arm": "B_paired", "best_val_qwk": paired_best.get("score", ""), **paired_metrics["test"]})
    _save_summary(metric_dir / "expert_summary.csv", rows)
    print(f"summary={metric_dir / 'expert_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
