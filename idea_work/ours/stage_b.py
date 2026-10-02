from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .core import (
    BOUNDARY_NAMES,
    StageBGate,
    atomic_json,
    atomic_torch_save,
    metric_key,
    metrics,
    read_csv,
    reconstruct_logits,
    restore_rng_state,
    rng_state,
    seed_everything,
    write_csv,
)

VARIANTS = {
    "grade_only": {"mode": "bounded", "cf": False, "fallback": False},
    "full": {"mode": "bounded", "cf": True, "fallback": True},
    "global": {"mode": "global", "cf": False, "fallback": False},
    "unbounded": {"mode": "unbounded", "cf": False, "fallback": False},
    "grade_cf": {"mode": "bounded", "cf": True, "fallback": False},
    "grade_fallback": {"mode": "bounded", "cf": False, "fallback": True},
}


def load_cache(root: Path, dataset: str) -> dict:
    return torch.load(
        root / "outputs/features" / f"{dataset}_seed42_main_cache.pt",
        map_location="cpu",
        weights_only=False,
    )


def split_indices(cache: dict, split: str) -> list[int]:
    return [index for index, row in enumerate(cache["metadata"]) if row["run_split"] == split]


def _tensor(cache: dict, key: str, indices: torch.Tensor, device) -> torch.Tensor:
    return cache[key][indices].to(device, non_blocking=True)


def tau_tensors(root: Path, dataset: str, cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    source = {row["sample_id"]: row for row in read_csv(root / "outputs/tau" / f"{dataset}_seed42_tau.csv")}
    tau = torch.zeros((len(cache["metadata"]), 4), dtype=torch.float32)
    valid = torch.zeros((len(cache["metadata"]), 4), dtype=torch.bool)
    train_ids = set()
    for index, row in enumerate(cache["metadata"]):
        if row["run_split"] != "train":
            continue
        sample_id = row["sample_id"]
        if sample_id not in source:
            raise AssertionError(f"Missing tau for {sample_id}")
        train_ids.add(sample_id)
        for boundary, name in enumerate(BOUNDARY_NAMES):
            valid[index, boundary] = bool(int(source[sample_id][f"valid_tau_{name}"]))
            if valid[index, boundary]:
                tau[index, boundary] = float(source[sample_id][f"tau_{name}"])
    if train_ids != set(source):
        raise AssertionError("Tau/cache sample_id join is not 1:1")
    return tau, valid


def random_fellow_mapping(cache: dict, indices: list[int], seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    metadata = cache["metadata"]
    by_laterality: dict[str, list[int]] = {}
    for global_index in indices:
        by_laterality.setdefault(metadata[global_index]["fellow_eye"], []).append(global_index)
    mapping = []
    for global_index in indices:
        row = metadata[global_index]
        preferred = [
            candidate
            for candidate in by_laterality.get(row["fellow_eye"], [])
            if metadata[candidate]["patient_id"] != row["patient_id"]
        ]
        candidates = preferred or [
            candidate for candidate in indices if metadata[candidate]["patient_id"] != row["patient_id"]
        ]
        if not candidates:
            raise AssertionError("No different-patient random fellow candidate")
        chosen = int(candidates[int(generator.integers(0, len(candidates)))])
        if metadata[chosen]["patient_id"] == row["patient_id"]:
            raise AssertionError("Random fellow patient collision")
        mapping.append(chosen)
    return torch.tensor(mapping, dtype=torch.long)


def _forward(model: StageBGate, cache: dict, indices: torch.Tensor, device, fellow_indices=None):
    if fellow_indices is None:
        fellow_indices = indices
    return model(
        _tensor(cache, "h_target", indices, device),
        _tensor(cache, "h_fellow", fellow_indices, device),
        _tensor(cache, "p_target", indices, device),
        _tensor(cache, "p_fellow", fellow_indices, device),
        _tensor(cache, "a_target", indices, device),
        _tensor(cache, "a_fellow", fellow_indices, device),
    )


@torch.inference_mode()
def evaluate(model: StageBGate, cache: dict, indices: list[int], device) -> dict[str, float]:
    model.eval()
    predictions, labels = [], []
    loader = DataLoader(TensorDataset(torch.tensor(indices)), batch_size=1024, shuffle=False)
    for (batch_indices,) in loader:
        logits, _ = _forward(model, cache, batch_indices, device)
        predictions.extend(logits.argmax(1).cpu().tolist())
        labels.extend(int(cache["metadata"][index]["y_true"]) for index in batch_indices.tolist())
    return metrics(np.asarray(labels), np.asarray(predictions))


def train_variant(config: dict, dataset: str, variant: str, root: Path) -> None:
    if variant not in VARIANTS:
        raise ValueError(variant)
    settings = VARIANTS[variant]
    seed_everything(int(config["seed"]))
    cache = load_cache(root, dataset)
    train_indices = split_indices(cache, "train")
    val_indices = split_indices(cache, "val" if dataset == "eyepacs" else "inner_val")
    tau, valid_tau = tau_tensors(root, dataset, cache)
    device = torch.device("cuda:0")
    model = StageBGate(
        settings["mode"], int(config["adapter_dim"]), int(config["gate_hidden"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["stage_b_lr"]), weight_decay=float(config["stage_b_wd"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["stage_b_epochs"])
    )
    run_dir = root / "outputs/checkpoints" / dataset / "stage_b" / variant
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    start_epoch, best, best_epoch, history = 1, (-float("inf"),) * 3, 0, []
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"]) + 1
        best = tuple(saved["best_key"])
        best_epoch = int(saved["best_epoch"])
        history = list(saved["history"])
        restore_rng_state(saved["rng"])
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    for epoch in range(start_epoch, int(config["stage_b_epochs"]) + 1):
        started = time.time()
        model.train()
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        loader = DataLoader(
            TensorDataset(torch.arange(len(train_indices))),
            batch_size=int(config["stage_b_batch"]),
            shuffle=True,
            generator=generator,
        )
        random_map = random_fellow_mapping(
            cache, train_indices, int(config["random_fellow_seed"]) + epoch
        )
        totals = []
        grade_values, cf_values, fallback_values = [], [], []
        for (local_indices,) in loader:
            global_indices = train_index_tensor[local_indices]
            labels = torch.tensor(
                [int(cache["metadata"][index]["y_true"]) for index in global_indices.tolist()],
                dtype=torch.long,
                device=device,
            )
            logits, coefficient = _forward(model, cache, global_indices, device)
            grade_loss = F.cross_entropy(logits, labels)
            cf_loss = torch.zeros((), device=device)
            if settings["cf"]:
                mask = valid_tau[global_indices].to(device)
                if mask.any():
                    cf_loss = F.binary_cross_entropy(
                        coefficient[mask], tau[global_indices].to(device)[mask]
                    )
            fallback_loss = torch.zeros((), device=device)
            if settings["fallback"]:
                random_indices = random_map[local_indices]
                random_logits, _ = _forward(
                    model, cache, global_indices, device, fellow_indices=random_indices
                )
                random_prob = random_logits.softmax(1)
                target_prob = _tensor(cache, "p_target", global_indices, device).detach()
                fallback_loss = F.kl_div(
                    torch.log(random_prob.clamp_min(1e-8)),
                    target_prob.clamp_min(1e-8),
                    reduction="batchmean",
                )
            loss = (
                grade_loss
                + float(config["lambda_cf"]) * cf_loss
                + float(config["lambda_fallback"]) * fallback_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("Non-finite Stage-B gradient")
            optimizer.step()
            totals.append(float(loss.detach()))
            grade_values.append(float(grade_loss.detach()))
            cf_values.append(float(cf_loss.detach()))
            fallback_values.append(float(fallback_loss.detach()))
        scheduler.step()
        val_metrics = evaluate(model, cache, val_indices, device)
        key = metric_key(val_metrics)
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "grade_loss": float(np.mean(grade_values)),
            "cf_loss": float(np.mean(cf_values)),
            "fallback_loss": float(np.mean(fallback_values)),
            "seconds": time.time() - started,
            "lr": scheduler.get_last_lr()[0],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(entry)
        if key > best:
            best, best_epoch = key, epoch
            atomic_torch_save(
                best_path,
                {
                    "model": model.state_dict(),
                    "mode": settings["mode"],
                    "epoch": epoch,
                    "val": val_metrics,
                },
            )
        atomic_torch_save(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_key": best,
                "best_epoch": best_epoch,
                "history": history,
                "rng": rng_state(),
            },
        )
        write_csv(run_dir / "history.csv", history)
        print(
            f"STAGE_B {dataset} {variant} epoch={epoch}/50 loss={entry['loss']:.6f} "
            f"val_qwk={val_metrics['qwk']:.6f} seconds={entry['seconds']:.1f}",
            flush=True,
        )
    atomic_json(run_dir / "summary.json", {"best_epoch": best_epoch, "best_key": best})
    predict_variant(config, dataset, variant, root)


def prediction_fields(mode: str = "bounded") -> list[str]:
    fields = [
        "dataset", "seed", "variant", "sample_id", "patient_id", "target_eye", "fellow_eye",
        "y_true", "fellow_y_true", "pred_grade", "target_only_pred", "final_pred",
    ]
    fields += [f"logit_{grade}" for grade in range(5)]
    fields += [f"prob_{grade}" for grade in range(5)]
    fields += [f"alpha_{name}" for name in BOUNDARY_NAMES]
    if mode == "unbounded":
        fields += [f"beta_{name}" for name in BOUNDARY_NAMES]
    return fields


@torch.inference_mode()
def make_predictions(
    config: dict,
    dataset: str,
    variant: str,
    cache: dict,
    indices: list[int],
    model: StageBGate | None,
    device,
    fellow_indices: torch.Tensor | None = None,
) -> list[dict[str, object]]:
    records = []
    if model is not None:
        model.eval()
    index_tensor = torch.tensor(indices, dtype=torch.long)
    for start in range(0, len(indices), 1024):
        batch = index_tensor[start : start + 1024]
        random_batch = fellow_indices[start : start + len(batch)] if fellow_indices is not None else batch
        if variant == "fixed50":
            coefficient = torch.full((len(batch), 4), 0.5, device=device)
            corrected = _tensor(cache, "a_target", batch, device) + coefficient * (
                _tensor(cache, "a_fellow", random_batch, device)
                - _tensor(cache, "a_target", batch, device)
            )
            logits = reconstruct_logits(corrected)
        else:
            if model is None:
                raise AssertionError("Model required")
            logits, coefficient = _forward(model, cache, batch, device, random_batch)
        probabilities = logits.softmax(1)
        target_logits = cache["z_target"][batch]
        for local, global_index in enumerate(batch.tolist()):
            row = cache["metadata"][global_index]
            final_pred = int(logits[local].argmax())
            record: dict[str, object] = {
                "dataset": dataset,
                "seed": config["seed"],
                "variant": variant,
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "target_eye": row["target_eye"],
                "fellow_eye": row["fellow_eye"],
                "y_true": row["y_true"],
                "fellow_y_true": row["fellow_y_true"],
                "pred_grade": final_pred,
                "target_only_pred": int(target_logits[local].argmax()),
                "final_pred": final_pred,
            }
            record.update({f"logit_{grade}": float(logits[local, grade]) for grade in range(5)})
            record.update({f"prob_{grade}": float(probabilities[local, grade]) for grade in range(5)})
            record.update(
                {f"alpha_{name}": float(coefficient[local, k]) for k, name in enumerate(BOUNDARY_NAMES)}
            )
            if VARIANTS.get(variant, {}).get("mode") == "unbounded":
                record.update(
                    {f"beta_{name}": float(coefficient[local, k]) for k, name in enumerate(BOUNDARY_NAMES)}
                )
            records.append(record)
    return records


def predict_variant(config: dict, dataset: str, variant: str, root: Path) -> None:
    cache = load_cache(root, dataset)
    test_indices = split_indices(cache, "test")
    device = torch.device("cuda:0")
    model = None
    mode = "bounded"
    if variant != "fixed50":
        settings = VARIANTS[variant]
        mode = settings["mode"]
        model = StageBGate(mode, int(config["adapter_dim"]), int(config["gate_hidden"])).to(device)
        checkpoint = root / "outputs/checkpoints" / dataset / "stage_b" / variant / "best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    records = make_predictions(config, dataset, variant, cache, test_indices, model, device)
    path = root / "outputs/predictions" / dataset / f"{variant}_seed42_test.csv"
    write_csv(path, records, prediction_fields(mode))
    value = metrics(
        np.asarray([int(row["y_true"]) for row in records]),
        np.asarray([int(row["pred_grade"]) for row in records]),
    )
    atomic_json(root / "outputs/metrics" / dataset / f"{variant}_seed42_test.json", value)
    print(f"PREDICT {dataset} {variant} {json.dumps(value, sort_keys=True)}", flush=True)


def fixed50(config: dict, dataset: str, root: Path) -> None:
    predict_variant(config, dataset, "fixed50", root)
