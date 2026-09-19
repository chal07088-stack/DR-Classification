from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from drpair.full_baselines.metrics import classes, metric_dict, write_diagnostics
from drpair.full_baselines.runtime import (
    METHODS,
    DirectionalDataset,
    atomic_json,
    batch_for,
    build_model,
    epochs_for,
    forward_loss,
    image_root,
    infer_batch,
    method_transform,
    optimizer_for,
    rows_for_run,
    schedule_for,
    seed_everything,
    sha256,
    worker_seed,
)

EYE_MEAN = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
EYE_STD = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
FULL_NUM_WORKERS = int(os.environ.get("DR_NUM_WORKERS", "24"))
FULL_PREFETCH_FACTOR = int(os.environ.get("DR_PREFETCH_FACTOR", "4"))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["eyepacs", "messidor2"])
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def stats(repo: Path, dataset: str, method: str):
    if method not in {"huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn"}:
        return IMAGENET_MEAN, IMAGENET_STD
    if dataset == "eyepacs":
        if method == "drgcnn_bffn":
            return [0.4844570755958557, 0.2249917984008789, 0.0770205482840538], [
                0.29392191767692566,
                0.14738348126411438,
                0.06418044865131378,
            ]
        return EYE_MEAN, EYE_STD
    path = repo / "outputs/data_audit/messidor_rgb_stats.json"
    if not path.is_file():
        # Dataset statistics are computed in audit_full_data.py on the outer training pool.
        raise RuntimeError(f"Messidor RGB statistics missing: {path}")
    value = json.loads(path.read_text())
    return value["mean"], value["std"]


def dependency_checkpoint(repo: Path, dataset: str, method: str, seed: int):
    parent = {"huang_pff": "huang_single", "drgcnn_bffn": "drgcnn_single"}.get(method)
    if not parent:
        return None
    path = repo / "outputs/checkpoints" / dataset / parent / f"seed{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"same-seed dependency missing: {path}")
    return path


def evaluate(repo, dataset, method, seed, model, rows, split, transform, device, batch_size):
    selected = [r for r in rows if r["run_split"] == split]
    loader = DataLoader(
        DirectionalDataset(selected, image_root(repo, dataset), transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=FULL_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=FULL_PREFETCH_FACTOR,
    )
    raw, probabilities = [], []
    model.eval()
    for target, fellow, _, _ in loader:
        target, fellow = target.to(device, non_blocking=True), fellow.to(device, non_blocking=True)
        score, prob = infer_batch(model, method, target, fellow)
        if prob.ndim != 2 or prob.shape[1] != 5 or len(score) != len(prob):
            raise RuntimeError(
                f"invalid probability shape: score={tuple(score.shape)} prob={tuple(prob.shape)}"
            )
        if not torch.isfinite(score).all() or not torch.isfinite(prob).all():
            raise FloatingPointError("non-finite inference output")
        if not torch.allclose(prob.sum(1), torch.ones(len(prob), device=prob.device), atol=1e-4):
            raise RuntimeError("probabilities do not sum to one")
        raw.extend(score.float().cpu().tolist())
        probabilities.extend(prob.float().cpu().tolist())
    raw_array = np.asarray(raw)
    y = np.asarray([int(r["y_true"]) for r in selected])
    metrics = metric_dict(y, raw_array)
    if split in {"val", "inner_val", "test"}:
        write_diagnostics(repo, dataset, method, seed, split, y, raw_array)
    if split == "test":
        out = repo / "outputs/predictions" / dataset / method / f"seed{seed}_test.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "dataset",
            "model",
            "seed",
            "split",
            "sample_id",
            "patient_id",
            "target_eye",
            "fellow_eye",
            "target_image_path",
            "fellow_image_path",
            "y_true",
            "fellow_y_true",
            "y_pred",
            "raw_score",
            "prob_0",
            "prob_1",
            "prob_2",
            "prob_3",
            "prob_4",
            "inter_eye_gap",
        ]
        with out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row, score, pred, prob in zip(
                selected, raw_array, classes(raw_array), probabilities
            ):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "model": method,
                        "seed": seed,
                        "split": "test",
                        "sample_id": row["sample_id"],
                        "patient_id": row["patient_id"],
                        "target_eye": row["target_eye"],
                        "fellow_eye": row["fellow_eye"],
                        "target_image_path": row["target_image_path"],
                        "fellow_image_path": row["fellow_image_path"],
                        "y_true": row["y_true"],
                        "fellow_y_true": row["fellow_y_true"],
                        "y_pred": pred,
                        "raw_score": score,
                        **{f"prob_{g}": prob[g] for g in range(5)},
                        "inter_eye_gap": row["inter_eye_gap"],
                    }
                )
    return metrics


def main():
    args = parse_args()
    repo = args.repo.resolve()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    rows = rows_for_run(repo, args.dataset)
    mean, std = stats(repo, args.dataset, args.method)
    train_t = method_transform(args.method, True, mean, std)
    eval_t = method_transform(args.method, False, mean, std)
    val_name = "val" if args.dataset == "eyepacs" else "inner_val"
    train_rows = [r for r in rows if r["run_split"] == "train"]
    if args.smoke:
        smoke_train = 32 if args.method == "d_odr" else max(2, min(batch_for(args.method), 8))
        train_rows = train_rows[:smoke_train]
        rows = train_rows + [r for r in rows if r["run_split"] == val_name][:4]
    parent = dependency_checkpoint(repo, args.dataset, args.method, args.seed)
    dependency_hash = sha256(parent) if parent else None
    model = build_model(repo, args.method, pretrained=True, parent_checkpoint=parent).to(device)
    optimizer = optimizer_for(model, args.method)
    scheduler = schedule_for(optimizer, args.method)
    generator = torch.Generator().manual_seed(args.seed)
    full_loader_options = (
        {}
        if args.smoke
        else {
            "persistent_workers": True,
            "prefetch_factor": FULL_PREFETCH_FACTOR,
        }
    )
    loader = DataLoader(
        DirectionalDataset(train_rows, image_root(repo, args.dataset), train_t),
        batch_size=min(batch_for(args.method), len(train_rows))
        if args.smoke
        else batch_for(args.method),
        shuffle=True,
        drop_last=not args.smoke,
        num_workers=0 if args.smoke else FULL_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_seed,
        generator=generator,
        **full_loader_options,
    )
    print(
        f"DATALOADER workers={loader.num_workers} "
        f"prefetch={FULL_PREFETCH_FACTOR if loader.num_workers else 0} "
        f"persistent={bool(loader.num_workers)}",
        flush=True,
    )
    epochs = 1 if args.smoke else epochs_for(args.method)
    run_dir = repo / "outputs/checkpoints" / args.dataset / args.method / f"seed{args.seed}"
    if args.smoke:
        run_dir = repo / "outputs/smoke" / args.dataset / args.method
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = (
        repo
        / {
            "huang_single": "configs/huang/main.json",
            "huang_pff": "configs/huang/main.json",
            "drgcnn_single": "configs/drgcnn/main.json",
            "drgcnn_bffn": "configs/drgcnn/main.json",
            "d_odr": "configs/d_odr/main.json",
            "aor_dr": "configs/aor_dr/main.json",
        }[args.method]
    )
    protocol_hash = sha256(repo / "configs/common/protocol.json")
    config_hash = sha256(config_path)
    scaler = torch.amp.GradScaler("cuda", enabled=args.method in {"d_odr", "aor_dr"})
    best = (-float("inf"), -float("inf"), -float("inf"))
    best_epoch = 0
    start_epoch = 1
    last_path = run_dir / "last.pt"
    if last_path.is_file() and not args.smoke:
        resume = torch.load(last_path, map_location=device, weights_only=False)
        if (
            resume.get("protocol_sha256") != protocol_hash
            or resume.get("method_config_sha256") != config_hash
            or resume.get("dependency_checkpoint_sha256") != dependency_hash
        ):
            print("IGNORE stale last.pt because protocol/config checksum changed", flush=True)
            resume = None
        if resume is not None:
            model.load_state_dict(resume["model_state"])
            optimizer.load_state_dict(resume["optimizer_state"])
            scaler.load_state_dict(resume["scaler_state"])
            if scheduler and resume["scheduler_state"] is not None:
                scheduler.load_state_dict(resume["scheduler_state"])
            best = tuple(resume["best"])
            best_epoch = int(resume["best_epoch"])
            start_epoch = int(resume["epoch"]) + 1
            random.setstate(resume["python_rng"])
            np.random.set_state(resume["numpy_rng"])
            torch.set_rng_state(resume["torch_rng"])
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
            generator.set_state(resume["loader_rng"])
            print(f"RESUME epoch={start_epoch} from {last_path}", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses = []
        started = time.time()
        for target, fellow, labels, _ in loader:
            target, fellow, labels = (
                target.to(device, non_blocking=True),
                fellow.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=args.method in {"d_odr", "aor_dr"}
            ):
                loss, _ = forward_loss(model, args.method, target, fellow, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {loss}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.method in {"huang_pff", "drgcnn_bffn", "d_odr"}:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val = evaluate(
            repo,
            args.dataset,
            args.method,
            args.seed,
            model,
            rows,
            val_name,
            eval_t,
            device,
            batch_for(args.method),
        )
        score = (
            (((val["qwk"] + 1.0) / 2.0 + val["macro_f1"]), val["qwk"], -val["mae"])
            if args.method == "d_odr"
            else (val["qwk"], val["macro_f1"], -val["mae"])
        )
        print(
            f"EPOCH {epoch}/{epochs} loss={np.mean(losses):.6f} val_qwk={val['qwk']:.6f} val_f1={val['macro_f1']:.6f} seconds={time.time() - started:.1f}",
            flush=True,
        )
        if score > best:
            best, best_epoch = score, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "dataset": args.dataset,
                    "method": args.method,
                    "validation": val,
                },
                run_dir / "best.pt",
            )
        if scheduler:
            scheduler.step(val["qwk"]) if args.method == "aor_dr" else scheduler.step()
        if not args.smoke:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "scaler_state": scaler.state_dict(),
                    "best": best,
                    "best_epoch": best_epoch,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "loader_rng": generator.get_state(),
                    "protocol_sha256": protocol_hash,
                    "method_config_sha256": config_hash,
                    "dependency_checkpoint_sha256": dependency_hash,
                },
                last_path,
            )
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    # save/load and probability shape are part of the smoke assertion.
    test_split = val_name if args.smoke else "test"
    final_metrics = evaluate(
        repo,
        args.dataset,
        args.method,
        args.seed,
        model,
        rows,
        test_split,
        eval_t,
        device,
        batch_for(args.method),
    )
    if args.smoke:
        print(f"SMOKE PASSED {args.dataset}/{args.method}", flush=True)
        return
    checksum = sha256(run_dir / "best.pt")
    status = {
        "status": "complete",
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint": str(run_dir / "best.pt"),
        "checkpoint_sha256": checksum,
        "dependency_checkpoint_sha256": dependency_hash,
        "protocol_sha256": protocol_hash,
        "method_config_sha256": config_hash,
        "test_metrics": final_metrics,
    }
    atomic_json(
        repo / "outputs/status" / f"{args.dataset}_{args.method}_seed{args.seed}.done", status
    )
    (repo / "outputs/status" / f"{args.dataset}_{args.method}_seed{args.seed}.failed").unlink(
        missing_ok=True
    )
    print(
        f"FINISHED {args.dataset}/{args.method}/seed{args.seed} checkpoint_sha256={checksum}",
        flush=True,
    )


if __name__ == "__main__":
    main()
