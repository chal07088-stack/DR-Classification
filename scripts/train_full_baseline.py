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
    PAIRED,
    DirectionalDataset,
    atomic_json,
    batch_for,
    physical_batch_for,
    accumulation_steps_for,
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
FULL_NUM_WORKERS = int(os.environ.get("DR_NUM_WORKERS", "8"))
FULL_PREFETCH_FACTOR = int(os.environ.get("DR_PREFETCH_FACTOR", "4"))
EVAL_NUM_WORKERS = int(os.environ.get("DR_EVAL_NUM_WORKERS", "8"))
EVAL_PREFETCH_FACTOR = int(os.environ.get("DR_EVAL_PREFETCH_FACTOR", "2"))
TEST_BATCH_SIZE = int(os.environ.get("DR_TEST_BATCH_SIZE", "128"))
# The bilateral 512px BFFN loader transfers two large tensors per sample.
# Keep eight workers but use a smaller prefetch queue for bounded IPC pressure.
BFFN_NUM_WORKERS = int(os.environ.get("DR_BFFN_NUM_WORKERS", "8"))
BFFN_PREFETCH_FACTOR = int(os.environ.get("DR_BFFN_PREFETCH_FACTOR", "2"))
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
    path = repo / "outputs/data_audit" / "drgcnn_messidor2_rgb_stats.json"
    if not path.is_file():
        # Dataset statistics are computed in audit_full_data.py on the outer training pool.
        raise RuntimeError(f"Messidor RGB statistics missing: {path}")
    value = json.loads(path.read_text())
    return value["mean"], value["std"]


def dependency_checkpoint(repo: Path, dataset: str, method: str, seed: int, smoke: bool = False):
    parent = {"huang_pff": "huang_single", "drgcnn_bffn": "drgcnn_single"}.get(method)
    if not parent:
        return None
    path = (
        repo / "outputs/smoke" / dataset / parent / "best.pt"
        if smoke
        else repo / "outputs/checkpoints" / dataset / parent / f"seed{seed}" / "best.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"same-seed dependency missing: {path}")
    return path


def evaluate(repo, dataset, method, seed, model, rows, split, transform, device, batch_size):
    selected = [r for r in rows if r["run_split"] == split]
    # BFFN's bilateral 512px batches require a modest, validated worker count.
    eval_workers = BFFN_NUM_WORKERS if method == "drgcnn_bffn" else EVAL_NUM_WORKERS
    eval_loader_options = (
        {}
        if eval_workers == 0
        else {
            "persistent_workers": False,
            "prefetch_factor": EVAL_PREFETCH_FACTOR,
        }
    )
    loader = DataLoader(
        DirectionalDataset(
            selected,
            image_root(repo, dataset, method),
            transform,
            load_fellow=method in PAIRED,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=True,
        **eval_loader_options,
    )
    print(
        f"EVAL split={split} batch_size={batch_size} workers={eval_workers} "
        f"prefetch={EVAL_PREFETCH_FACTOR if eval_workers else 0}",
        flush=True,
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
    # Linux uses file_descriptor with the server's ample /dev/shm.  Windows
    # only exposes file_system, so select from the runtime-supported set.
    sharing = (
        "file_descriptor"
        if "file_descriptor" in torch.multiprocessing.get_all_sharing_strategies()
        else "file_system"
    )
    torch.multiprocessing.set_sharing_strategy(sharing)
    print(f"MULTIPROCESSING sharing_strategy={sharing}", flush=True)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    # This executable is deliberately train/validation-only.  Keeping test
    # rows out of memory makes pre-freeze label access impossible by design;
    # the separate frozen-checkpoint evaluator is the only test entry point.
    rows = rows_for_run(repo, args.dataset, include_test=False)
    mean, std = stats(repo, args.dataset, args.method)
    train_t = method_transform(args.method, True, mean, std)
    eval_t = method_transform(args.method, False, mean, std)
    val_name = "val" if args.dataset == "eyepacs" else "inner_val"
    train_rows = [r for r in rows if r["run_split"] == "train"]
    if args.smoke:
        # Exercise one complete paper-native physical batch for D-ODR.  Its
        # released recipe uses batch 128 with no gradient accumulation, so a
        # smaller smoke subset would not validate the actual update path.
        # Several full 128-sample updates let the same dynamic GradScaler used
        # by the released FP16 Trainer settle if its conservative initial scale
        # overflows on the first real retinal batch.
        smoke_train = 1024 if args.method == "d_odr" else max(2, min(physical_batch_for(args.method), 8))
        train_rows = train_rows[:smoke_train]
        rows = train_rows + [r for r in rows if r["run_split"] == val_name][:4]
    parent = dependency_checkpoint(repo, args.dataset, args.method, args.seed, args.smoke)
    dependency_hash = sha256(parent) if parent else None
    model = build_model(repo, args.method, pretrained=True, parent_checkpoint=parent).to(device)
    optimizer = optimizer_for(model, args.method)
    generator = torch.Generator().manual_seed(args.seed)
    train_workers = BFFN_NUM_WORKERS if args.method == "drgcnn_bffn" else FULL_NUM_WORKERS
    train_prefetch = (
        2 if args.smoke else BFFN_PREFETCH_FACTOR if args.method == "drgcnn_bffn" else FULL_PREFETCH_FACTOR
    )
    full_loader_options = (
        {}
        if train_workers == 0
        else {
            "persistent_workers": False,
            "prefetch_factor": train_prefetch,
        }
    )
    loader = DataLoader(
        DirectionalDataset(
            train_rows,
            image_root(repo, args.dataset, args.method),
            train_t,
            load_fellow=args.method in PAIRED,
        ),
        batch_size=min(physical_batch_for(args.method), len(train_rows))
        if args.smoke
        else physical_batch_for(args.method),
        shuffle=True,
        drop_last=not args.smoke and args.method != "d_odr",
        num_workers=train_workers,
        pin_memory=True,
        worker_init_fn=worker_seed,
        generator=generator,
        **full_loader_options,
    )
    print(
        f"DATALOADER workers={loader.num_workers} "
        f"prefetch={train_prefetch if loader.num_workers else 0} "
        f"persistent={loader.persistent_workers}",
        flush=True,
    )
    accumulation_steps = accumulation_steps_for(args.method)
    print(
        f"BATCH physical={physical_batch_for(args.method)} "
        f"accumulation={accumulation_steps} effective={batch_for(args.method)}",
        flush=True,
    )
    scheduler = schedule_for(optimizer, args.method, steps_per_epoch=len(loader))
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
    # The released AOR-DR trainer performs its diffusion loss/backward pass in
    # FP32. Running that loss under FP16 autocast can overflow to NaN, so AMP
    # remains enabled only for D-ODR here.
    amp_enabled = args.method == "d_odr"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"AMP enabled={amp_enabled}", flush=True)
    best = (-float("inf"), -float("inf"), -float("inf"))
    best_epoch = 0
    optimizer_steps = 0
    start_epoch = 1
    last_path = run_dir / "last.pt"
    if last_path.is_file() and not args.smoke:
        # Keep RNG tensors on CPU; torch.set_rng_state and DataLoader generators
        # reject CUDA ByteTensors. Model/optimizer load_state_dict moves their
        # tensors to the already-created CUDA parameters as needed.
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
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
            if scaler.is_enabled():
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
        optimizer.zero_grad(set_to_none=True)
        # Full training uses only complete accumulation windows.  For
        # DR-GCNN Single this consumes floor(N/16)*16 samples, matching the
        # original physical-batch-16 drop_last behavior exactly.  Smoke mode
        # deliberately permits a shorter final window so it still exercises
        # a real optimizer update on its tiny subset.
        usable_batches = (
            len(loader)
            if args.smoke
            else len(loader) - (len(loader) % accumulation_steps)
        )
        for batch_index, (target, fellow, labels, _) in enumerate(loader):
            if batch_index >= usable_batches:
                break
            target, fellow, labels = (
                target.to(device, non_blocking=True),
                fellow.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            window_start = (batch_index // accumulation_steps) * accumulation_steps
            window_size = min(accumulation_steps, usable_batches - window_start)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                loss, _ = forward_loss(model, args.method, target, fellow, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {loss}")
            scaler.scale(loss / window_size).backward()
            end_of_window = (batch_index + 1 == usable_batches) or (
                (batch_index + 1) % accumulation_steps == 0
            )
            if end_of_window:
                scaler.unscale_(optimizer)
                if args.method in {"huang_pff", "drgcnn_bffn", "d_odr"}:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler skips optimizer.step() when it finds an overflow.
                # A per-update scheduler must be advanced only after a real
                # parameter update, otherwise PyTorch correctly warns that the
                # first learning-rate value was skipped.
                optimizer_step_was_skipped = (
                    scaler.is_enabled() and scaler.get_scale() < scale_before_step
                )
                if not optimizer_step_was_skipped:
                    optimizer_steps += 1
                elif args.smoke:
                    print(
                        f"AMP optimizer step skipped after overflow; "
                        f"new_scale={scaler.get_scale():.1f}",
                        flush=True,
                    )
                optimizer.zero_grad(set_to_none=True)
                if scheduler and args.method == "d_odr" and not optimizer_step_was_skipped:
                    scheduler.step()
            losses.append(float(loss.detach()))
        if args.smoke and optimizer_steps == 0:
            raise RuntimeError(
                "smoke epoch completed without a real optimizer update; "
                "AMP overflow caused every GradScaler step to be skipped"
            )
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
            physical_batch_for(args.method),
        )
        # Tiny smoke splits can contain a single class, for which sklearn
        # correctly reports QWK as NaN.  Treat undefined metrics as the worst
        # possible selection values so the smoke run still writes a loadable
        # checkpoint and exercises the complete save/load path.
        selection_qwk = val["qwk"] if np.isfinite(val["qwk"]) else -1.0
        selection_f1 = val["macro_f1"] if np.isfinite(val["macro_f1"]) else -1.0
        selection_mae = val["mae"] if np.isfinite(val["mae"]) else float("inf")
        score = (
            (((selection_qwk + 1.0) / 2.0 + selection_f1), selection_qwk, -selection_mae)
            if args.method == "d_odr"
            else (selection_qwk, selection_f1, -selection_mae)
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
        if scheduler and args.method != "d_odr":
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
    # Save/load and probability shape are part of the smoke assertion.  Smoke
    # evaluates validation a second time; it never loads official test rows.
    if args.smoke:
        evaluate(
            repo,
            args.dataset,
            args.method,
            args.seed,
            model,
            rows,
            val_name,
            eval_t,
            device,
            physical_batch_for(args.method),
        )
        print(f"SMOKE PASSED {args.dataset}/{args.method}", flush=True)
        return
    checksum = sha256(run_dir / "best.pt")
    status = {
        "status": "trained_awaiting_global_freeze",
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint": str(run_dir / "best.pt"),
        "checkpoint_sha256": checksum,
        "dependency_checkpoint_sha256": dependency_hash,
        "physical_batch_size": physical_batch_for(args.method),
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_for(args.method),
        "protocol_sha256": protocol_hash,
        "method_config_sha256": config_hash,
        "official_test_manifest_loaded": False,
        "official_test_image_forward": 0,
        "official_test_metric": 0,
    }
    atomic_json(
        repo / "outputs/status" / f"{args.dataset}_{args.method}_seed{args.seed}.trained", status
    )
    print(
        f"TRAINING FROZEN CANDIDATE {args.dataset}/{args.method}/seed{args.seed} "
        f"checkpoint_sha256={checksum} official_test_accessed=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
