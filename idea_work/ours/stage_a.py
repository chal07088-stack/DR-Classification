from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .core import (
    BOUNDARY_NAMES,
    DirectionalImageDataset,
    StageAModel,
    atomic_json,
    atomic_torch_save,
    boundaries,
    image_root,
    image_transform,
    load_rows,
    metric_key,
    metrics,
    read_csv,
    reconstruct_logits,
    restore_rng_state,
    rng_state,
    seed_everything,
    seed_worker,
    sha256,
    write_csv,
)


def _loader(
    config: dict,
    dataset: str,
    rows: list[dict[str, str]],
    train: bool,
    paired=False,
    generator: torch.Generator | None = None,
):
    workers = int(config["num_workers"] if train else config["eval_num_workers"])
    options = {}
    if workers:
        options = {
            "persistent_workers": train,
            "prefetch_factor": int(config["prefetch_factor"]),
        }
    return DataLoader(
        DirectionalImageDataset(
            rows,
            image_root(config, dataset),
            image_transform(train, int(config["input_size"])),
            include_fellow=paired,
        ),
        batch_size=int(config["stage_a_batch"]),
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator or torch.Generator().manual_seed(int(config["seed"])),
        **options,
    )


def _accumulation_steps(config: dict) -> int:
    steps = int(config.get("stage_a_accumulation_steps", 1))
    if steps < 1:
        raise ValueError("stage_a_accumulation_steps must be >= 1")
    expected = int(config["stage_a_batch"]) * steps
    if expected != int(config["stage_a_effective_batch"]):
        raise ValueError(
            f"Stage-A batch contract mismatch: physical={config['stage_a_batch']} "
            f"accumulation={steps} effective={config['stage_a_effective_batch']}"
        )
    return steps


def _usable_microbatches(loader: DataLoader, accumulation_steps: int) -> int:
    """Match a physical effective-batch loader's drop_last sample count."""

    return len(loader) - (len(loader) % accumulation_steps)


@torch.inference_mode()
def evaluate_model(model, loader, device) -> tuple[dict[str, float], list[int]]:
    model.eval()
    labels, predictions = [], []
    for batch in loader:
        y = batch["label"].numpy().tolist()
        _, logits = model(batch["target"].to(device, non_blocking=True))
        labels.extend(y)
        predictions.extend(logits.argmax(1).cpu().tolist())
    return metrics(np.asarray(labels), np.asarray(predictions)), predictions


def _checkpoint_payload(
    model, optimizer, scheduler, scaler, epoch, best, best_epoch, history, loader_generator
):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_key": best,
        "best_epoch": best_epoch,
        "history": history,
        "rng": rng_state(),
        "loader_rng": loader_generator.get_state(),
    }


def train_main(config: dict, dataset: str, root: Path) -> None:
    seed_everything(int(config["seed"]))
    rows = load_rows(config, dataset)
    val_name = "val" if dataset == "eyepacs" else "inner_val"
    train_rows = [row for row in rows if row["run_split"] == "train"]
    val_rows = [row for row in rows if row["run_split"] == val_name]
    device = torch.device("cuda:0")
    model = StageAModel(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["stage_a_lr"]), weight_decay=float(config["stage_a_wd"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["stage_a_epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["stage_a_amp"]))
    run_dir = root / "outputs/checkpoints" / dataset / "stage_a" / "seed42"
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    start_epoch, best, best_epoch, history = 1, (-float("inf"),) * 3, 0, []
    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        best = tuple(saved["best_key"])
        best_epoch = int(saved["best_epoch"])
        history = list(saved["history"])
        restore_rng_state(saved["rng"])
        if "loader_rng" in saved:
            loader_generator.set_state(saved["loader_rng"])
        print(f"RESUME main Stage-A {dataset} at epoch {start_epoch}", flush=True)
    train_loader = _loader(config, dataset, train_rows, True, generator=loader_generator)
    val_loader = _loader(config, dataset, val_rows, False)
    accumulation_steps = _accumulation_steps(config)
    print(
        f"STAGE_A_BATCH physical={config['stage_a_batch']} "
        f"accumulation={accumulation_steps} effective={config['stage_a_effective_batch']}",
        flush=True,
    )
    for epoch in range(start_epoch, int(config["stage_a_epochs"]) + 1):
        started = time.time()
        model.train()
        losses = []
        usable_batches = _usable_microbatches(train_loader, accumulation_steps)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= usable_batches:
                break
            images = batch["target"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=bool(config["stage_a_amp"])):
                _, logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        scheduler.step()
        val_metrics, _ = evaluate_model(model, val_loader, device)
        key = metric_key(val_metrics)
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "seconds": time.time() - started,
            "lr": scheduler.get_last_lr()[0],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(entry)
        if key > best:
            best, best_epoch = key, epoch
            atomic_torch_save(
                best_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": val_metrics,
                    "architecture": "torchvision.convnext_small",
                },
            )
        atomic_torch_save(
            last_path,
            _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best,
                best_epoch,
                history,
                loader_generator,
            ),
        )
        write_csv(run_dir / "history.csv", history)
        print(
            f"STAGE_A {dataset} epoch={epoch}/25 loss={entry['loss']:.6f} "
            f"val_qwk={val_metrics['qwk']:.6f} val_f1={val_metrics['macro_f1']:.6f} "
            f"seconds={entry['seconds']:.1f}",
            flush=True,
        )
    if not best_path.is_file():
        raise RuntimeError(f"Missing best checkpoint: {best_path}")
    atomic_json(run_dir / "summary.json", {"best_epoch": best_epoch, "best_key": best})


def _prediction_fields(include_features: bool = False) -> list[str]:
    fields = [
        "dataset", "seed", "variant", "split", "sample_id", "patient_id", "target_eye",
        "fellow_eye", "y_true", "fellow_y_true", "pred_grade",
    ]
    fields += [f"logit_{grade}" for grade in range(5)]
    fields += [f"prob_{grade}" for grade in range(5)]
    if include_features:
        fields += [f"fellow_logit_{grade}" for grade in range(5)]
        fields += [f"a_target_{name}" for name in BOUNDARY_NAMES]
        fields += [f"a_fellow_{name}" for name in BOUNDARY_NAMES]
    return fields


@torch.inference_mode()
def predict_rows(config: dict, dataset: str, model, rows: list[dict[str, str]], device):
    loader = _loader(config, dataset, rows, False, paired=True)
    records: list[dict[str, object]] = []
    model.eval()
    offset = 0
    for batch in loader:
        target = batch["target"].to(device, non_blocking=True)
        fellow = batch["fellow"].to(device, non_blocking=True)
        combined = torch.cat([target, fellow], dim=0)
        _, logits = model(combined)
        target_logits, fellow_logits = logits.chunk(2)
        probs = target_logits.softmax(1)
        at, af = boundaries(target_logits), boundaries(fellow_logits)
        count = len(target)
        for local in range(count):
            row = rows[offset + local]
            record: dict[str, object] = {
                "dataset": dataset,
                "seed": config["seed"],
                "variant": "target_only",
                "split": row["run_split"],
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "target_eye": row["target_eye"],
                "fellow_eye": row["fellow_eye"],
                "y_true": row["y_true"],
                "fellow_y_true": row["fellow_y_true"],
                "pred_grade": int(target_logits[local].argmax()),
            }
            record.update({f"logit_{g}": float(target_logits[local, g]) for g in range(5)})
            record.update({f"prob_{g}": float(probs[local, g]) for g in range(5)})
            record.update({f"fellow_logit_{g}": float(fellow_logits[local, g]) for g in range(5)})
            record.update({f"a_target_{n}": float(at[local, k]) for k, n in enumerate(BOUNDARY_NAMES)})
            record.update({f"a_fellow_{n}": float(af[local, k]) for k, n in enumerate(BOUNDARY_NAMES)})
            records.append(record)
        offset += count
    return records


def target_only_test(config: dict, dataset: str, root: Path) -> None:
    rows = load_rows(config, dataset)
    test_rows = [row for row in rows if row["run_split"] == "test"]
    checkpoint = root / "outputs/checkpoints" / dataset / "stage_a/seed42/best.pt"
    model = StageAModel(pretrained=False).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    records = predict_rows(config, dataset, model, test_rows, torch.device("cuda:0"))
    path = root / "outputs/predictions" / dataset / "target_only_seed42_test.csv"
    write_csv(path, records, _prediction_fields(True))
    value = metrics(
        np.asarray([int(row["y_true"]) for row in records]),
        np.asarray([int(row["pred_grade"]) for row in records]),
    )
    atomic_json(root / "outputs/metrics" / dataset / "target_only_seed42_test.json", value)
    print(f"TARGET_ONLY {dataset} {json.dumps(value, sort_keys=True)}", flush=True)


def create_oof_mapping(config: dict, dataset: str, root: Path) -> None:
    rows = [row for row in load_rows(config, dataset) if row["run_split"] == "train"]
    patients: dict[str, int] = {}
    for row in rows:
        patient = row["patient_id"]
        patients[patient] = max(patients.get(patient, -1), int(row["y_true"]), int(row["fellow_y_true"]))
    patient_ids = np.asarray(sorted(patients))
    severity = np.asarray([patients[patient] for patient in patient_ids])
    folds = np.full(len(patient_ids), -1, dtype=int)
    splitter = StratifiedKFold(
        n_splits=int(config["oof_folds"]), shuffle=True, random_state=int(config["oof_random_state"])
    )
    for fold, (_, held_out) in enumerate(splitter.split(patient_ids, severity)):
        folds[held_out] = fold
    if np.any(folds < 0):
        raise AssertionError("Incomplete OOF fold assignment")
    records = [
        {"patient_id": patient, "patient_severity": int(sev), "fold_id": int(fold)}
        for patient, sev, fold in zip(patient_ids, severity, folds)
    ]
    path = root / "outputs/tau" / f"{dataset}_seed42_oof_patient_folds.csv"
    if path.is_file():
        existing = read_csv(path)
        normalized = [{key: str(value) for key, value in row.items()} for row in records]
        if existing != normalized:
            raise RuntimeError(f"Existing OOF mapping differs: {path}")
    else:
        write_csv(path, records, ["patient_id", "patient_severity", "fold_id"])
    digest = sha256(path)
    (path.with_suffix(path.suffix + ".sha256")).write_text(f"{digest}  {path.name}\n", encoding="ascii")
    print(f"OOF_MAPPING {dataset} patients={len(records)} sha256={digest}", flush=True)


def train_oof_fold(config: dict, dataset: str, fold: int, root: Path) -> None:
    seed_everything(int(config["seed"]))
    all_rows = load_rows(config, dataset)
    mapping_path = root / "outputs/tau" / f"{dataset}_seed42_oof_patient_folds.csv"
    mapping = {row["patient_id"]: int(row["fold_id"]) for row in read_csv(mapping_path)}
    train_rows = [
        row for row in all_rows if row["run_split"] == "train" and mapping[row["patient_id"]] != fold
    ]
    held_rows = [
        row for row in all_rows if row["run_split"] == "train" and mapping[row["patient_id"]] == fold
    ]
    if {row["patient_id"] for row in train_rows} & {row["patient_id"] for row in held_rows}:
        raise AssertionError("OOF patient leakage")
    device = torch.device("cuda:0")
    model = StageAModel(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["stage_a_lr"]), weight_decay=float(config["stage_a_wd"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["stage_a_epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["stage_a_amp"]))
    run_dir = root / "outputs/checkpoints" / dataset / "oof" / f"fold{fold}"
    last_path, final_path = run_dir / "last.pt", run_dir / "final.pt"
    start_epoch, history = 1, []
    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        history = list(saved["history"])
        restore_rng_state(saved["rng"])
        if "loader_rng" in saved:
            loader_generator.set_state(saved["loader_rng"])
    loader = _loader(config, dataset, train_rows, True, generator=loader_generator)
    accumulation_steps = _accumulation_steps(config)
    print(
        f"OOF_BATCH fold={fold} physical={config['stage_a_batch']} "
        f"accumulation={accumulation_steps} effective={config['stage_a_effective_batch']}",
        flush=True,
    )
    for epoch in range(start_epoch, int(config["stage_a_epochs"]) + 1):
        started = time.time()
        model.train()
        losses = []
        usable_batches = _usable_microbatches(loader, accumulation_steps)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            if batch_index >= usable_batches:
                break
            images = batch["target"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=bool(config["stage_a_amp"])):
                _, logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.time() - started}
        )
        atomic_torch_save(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "rng": rng_state(),
                "loader_rng": loader_generator.get_state(),
            },
        )
        write_csv(run_dir / "history.csv", history)
        print(
            f"OOF {dataset} fold={fold} epoch={epoch}/25 loss={history[-1]['loss']:.6f} "
            f"seconds={history[-1]['seconds']:.1f}",
            flush=True,
        )
    atomic_torch_save(final_path, {"model": model.state_dict(), "epoch": int(config["stage_a_epochs"])})
    records = predict_rows(config, dataset, model, held_rows, device)
    path = root / "outputs/tau" / f"{dataset}_seed42_oof_fold{fold}_predictions.csv"
    write_csv(path, records, _prediction_fields(True))


def generate_tau(config: dict, dataset: str, root: Path) -> None:
    predictions = []
    for fold in range(int(config["oof_folds"])):
        predictions.extend(
            read_csv(root / "outputs/tau" / f"{dataset}_seed42_oof_fold{fold}_predictions.csv")
        )
    expected = [row for row in load_rows(config, dataset) if row["run_split"] == "train"]
    if {row["sample_id"] for row in predictions} != {row["sample_id"] for row in expected}:
        raise AssertionError("OOF predictions do not cover train sample_ids exactly")
    epsilon = float(config["cf_epsilon"])
    output = []
    for row in predictions:
        z_target = torch.tensor([[float(row[f"logit_{g}"]) for g in range(5)]])
        z_fellow = torch.tensor([[float(row[f"fellow_logit_{g}"]) for g in range(5)]])
        y = torch.tensor([int(row["y_true"])])
        at, af = boundaries(z_target), boundaries(z_fellow)
        target_loss = float(F.cross_entropy(z_target, y, reduction="none")[0])
        record: dict[str, object] = {
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "L_T": target_loss,
        }
        for index, name in enumerate(BOUNDARY_NAMES):
            counterfactual = at.clone()
            counterfactual[:, index] = af[:, index]
            loss = float(F.cross_entropy(reconstruct_logits(counterfactual), y, reduction="none")[0])
            delta = target_loss - loss
            valid = abs(delta) > epsilon
            record[f"tau_{name}"] = int(delta > epsilon) if valid else ""
            record[f"valid_tau_{name}"] = int(valid)
            record[f"L_CF_{name}"] = loss
            record[f"DeltaL_{name}"] = delta
        output.append(record)
    fields = ["sample_id", "patient_id"]
    fields += [f"tau_{name}" for name in BOUNDARY_NAMES]
    fields += [f"valid_tau_{name}" for name in BOUNDARY_NAMES]
    fields += ["L_T"]
    fields += [f"L_CF_{name}" for name in BOUNDARY_NAMES]
    fields += [f"DeltaL_{name}" for name in BOUNDARY_NAMES]
    write_csv(root / "outputs/tau" / f"{dataset}_seed42_tau.csv", output, fields)
    print(f"TAU {dataset} rows={len(output)}", flush=True)


@torch.inference_mode()
def cache_features(config: dict, dataset: str, root: Path) -> None:
    rows = load_rows(config, dataset)
    checkpoint = root / "outputs/checkpoints" / dataset / "stage_a/seed42/best.pt"
    model = StageAModel(pretrained=False).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()
    loader = _loader(config, dataset, rows, False, paired=True)
    values: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("h_target", "h_fellow", "z_target", "z_fellow", "p_target", "p_fellow", "a_target", "a_fellow")
    }
    for batch in loader:
        combined = torch.cat(
            [batch["target"].cuda(non_blocking=True), batch["fellow"].cuda(non_blocking=True)], dim=0
        )
        features, logits = model(combined)
        h_target, h_fellow = features.chunk(2)
        z_target, z_fellow = logits.chunk(2)
        batch_values = {
            "h_target": h_target,
            "h_fellow": h_fellow,
            "z_target": z_target,
            "z_fellow": z_fellow,
            "p_target": z_target.softmax(1),
            "p_fellow": z_fellow.softmax(1),
            "a_target": boundaries(z_target),
            "a_fellow": boundaries(z_fellow),
        }
        for key, tensor in batch_values.items():
            values[key].append(tensor.float().cpu())
    payload: dict[str, object] = {
        "metadata": [
            {
                key: row[key]
                for key in (
                    "dataset", "run_split", "sample_id", "patient_id", "target_eye", "fellow_eye",
                    "y_true", "fellow_y_true", "target_image_path", "fellow_image_path",
                )
            }
            for row in rows
        ]
    }
    payload.update({key: torch.cat(tensors) for key, tensors in values.items()})
    if any(len(payload[key]) != len(rows) for key in values):
        raise AssertionError("Feature cache length mismatch")
    path = root / "outputs/features" / f"{dataset}_seed42_main_cache.pt"
    atomic_torch_save(path, payload)
    print(f"CACHE {dataset} rows={len(rows)} path={path}", flush=True)
