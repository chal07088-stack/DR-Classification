from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .core import BOUNDARY_NAMES, StageBGate, atomic_json, atomic_torch_save, boundaries, metric_key, read_csv, reconstruct_logits, seed_everything, write_csv
from .soft_tau_experiment import EPS_NUM, verify_stage_a_frozen
from .stage_b import evaluate, load_cache, make_predictions, prediction_fields, random_fellow_mapping, split_indices

EXPERIMENTS = (
    {"name": "EXP_RPS_SOFT_TAU_WMSE_CF1", "supervision": "rps", "lambda_cf": 1.0, "relation_dim": 522},
    {"name": "EXP_UTILITY_WMSE_CF02", "supervision": "utility", "lambda_cf": 0.2, "relation_dim": 522},
    {"name": "EXP_DISAGREE_GATE_UTILITY_CF02", "supervision": "utility", "lambda_cf": 0.2, "relation_dim": 538},
)


class DisagreementStageBGate(nn.Module):
    def __init__(self, adapter_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.mode = "bounded"
        self.adapter = nn.Sequential(nn.Linear(768, adapter_dim), nn.GELU(), nn.LayerNorm(adapter_dim))
        self.gate = nn.Sequential(nn.Linear(adapter_dim * 2 + 10 + 16, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def coefficients(self, h_target, h_fellow, p_target, p_fellow, a_target, a_fellow):
        target = self.adapter(h_target)
        fellow = self.adapter(h_fellow)
        delta = a_fellow - a_target
        relation = torch.cat((target - fellow, target * fellow, p_target, p_fellow, a_target, a_fellow, delta, delta.abs()), dim=1)
        if relation.shape[1] != 538:
            raise AssertionError(f"EXP-C relation dimension {relation.shape[1]} != 538")
        return torch.sigmoid(self.gate(relation))

    def forward(self, h_target, h_fellow, p_target, p_fellow, a_target, a_fellow):
        alpha = self.coefficients(h_target, h_fellow, p_target, p_fellow, a_target, a_fellow)
        return reconstruct_logits(a_target + alpha * (a_fellow - a_target)), alpha


def model_for(config, relation_dim: int):
    if relation_dim == 522:
        model = StageBGate("bounded", int(config["adapter_dim"]), int(config["gate_hidden"]))
        if model.gate[0].in_features != 522:
            raise AssertionError("EXP-A/B relation dimension mismatch")
        return model
    if relation_dim == 538:
        model = DisagreementStageBGate(int(config["adapter_dim"]), int(config["gate_hidden"]))
        if model.gate[0].in_features != 538:
            raise AssertionError("EXP-C relation dimension mismatch")
        return model
    raise ValueError(relation_dim)


def forward_model(model, cache, indices, device, fellow_indices=None):
    if fellow_indices is None:
        fellow_indices = indices
    keys = ("h_target", "h_fellow", "p_target", "p_fellow", "a_target", "a_fellow")
    values = []
    for key in keys:
        source_indices = fellow_indices if key.endswith("fellow") else indices
        values.append(cache[key][source_indices].to(device, non_blocking=True))
    return model(*values)


@torch.inference_mode()
def evaluate_model(model, cache, indices, device):
    model.eval(); predictions, labels = [], []
    for (batch,) in DataLoader(TensorDataset(torch.tensor(indices)), batch_size=1024, shuffle=False):
        logits, _ = forward_model(model, cache, batch, device)
        predictions.extend(logits.argmax(1).cpu().tolist())
        labels.extend(int(cache["metadata"][index]["y_true"]) for index in batch.tolist())
    from .core import metrics
    return metrics(np.asarray(labels), np.asarray(predictions))


def generate_utility_cache(root: Path) -> Path:
    path = root / "results/THREE_STAGE_ASSETS/eyepacs_seed42_endpoint_utility.csv"
    if path.is_file():
        return path
    predictions = []
    for fold in range(3):
        predictions.extend(read_csv(root / f"outputs/tau/eyepacs_seed42_oof_fold{fold}_predictions.csv"))
    cache = load_cache(root, "eyepacs")
    expected = {row["sample_id"] for row in cache["metadata"] if row["run_split"] == "train"}
    if len(predictions) != len(expected) or {row["sample_id"] for row in predictions} != expected:
        raise AssertionError("Utility OOF/cache mismatch")
    rows = []
    for source in predictions:
        zt = torch.tensor([[float(source[f"logit_{grade}"]) for grade in range(5)]], dtype=torch.float64)
        zf = torch.tensor([[float(source[f"fellow_logit_{grade}"]) for grade in range(5)]], dtype=torch.float64)
        at, af = boundaries(zt), boundaries(zf)
        label = torch.tensor([int(source["y_true"])])
        row = {"sample_id": source["sample_id"], "patient_id": source["patient_id"], "y_true": int(source["y_true"]), "fellow_y_true": int(source["fellow_y_true"])}
        for boundary, name in enumerate(BOUNDARY_NAMES):
            endpoint = at.clone(); endpoint[:, boundary] = af[:, boundary]
            loss0 = float(F.cross_entropy(reconstruct_logits(at), label))
            loss1 = float(F.cross_entropy(reconstruct_logits(endpoint), label))
            best, worst = min(loss0, loss1), max(loss0, loss1)
            row.update({f"tau_utility_{name}": int(loss1 < loss0), f"weight_utility_{name}": min(1.0, max(0.0, (worst - best) / (worst + EPS_NUM))), f"loss_t0_{name}": loss0, f"loss_t1_{name}": loss1})
        rows.append(row)
    write_csv(path, rows)
    atomic_json(path.with_suffix(".provenance.json"), {"rows": len(rows), "target_gt_only": True, "semantic_threshold": None, "tie_break": "L1 < L0 else target-anchored zero", "eps_num": EPS_NUM})
    return path


def supervision_tensors(root: Path, cache, kind: str):
    if kind == "rps":
        path = root / "results/RPS_SOFT_TAU_DIAGNOSTIC/rps_soft_tau_cache.csv"
        tau_fields = [f"tau_rps_{name}" for name in BOUNDARY_NAMES]
        weight_fields = [f"weight_rps_{name}" for name in BOUNDARY_NAMES]
    elif kind == "utility":
        path = generate_utility_cache(root)
        tau_fields = [f"tau_utility_{name}" for name in BOUNDARY_NAMES]
        weight_fields = [f"weight_utility_{name}" for name in BOUNDARY_NAMES]
    else:
        raise ValueError(kind)
    source = {row["sample_id"]: row for row in read_csv(path)}
    tau = torch.zeros((len(cache["metadata"]), 4), dtype=torch.float32)
    weight = torch.zeros_like(tau)
    joined = set()
    for index, metadata in enumerate(cache["metadata"]):
        if metadata["run_split"] != "train":
            continue
        row = source[metadata["sample_id"]]; joined.add(metadata["sample_id"])
        for boundary in range(4):
            tau[index, boundary] = float(row[tau_fields[boundary]])
            weight[index, boundary] = float(row[weight_fields[boundary]])
    if joined != set(source):
        raise AssertionError(f"{kind} supervision/cache join mismatch")
    return tau, weight, path


def smoke_test(config, root: Path, spec: dict):
    output = root / "results" / spec["name"]
    cache = load_cache(root, "eyepacs")
    tau, weight, source = supervision_tensors(root, cache, spec["supervision"])
    indices = torch.tensor(split_indices(cache, "train")[:512], dtype=torch.long)
    if not torch.isfinite(tau[indices]).all() or not ((tau[indices] >= 0) & (tau[indices] <= 1)).all():
        raise AssertionError("Invalid supervision tau")
    if not torch.isfinite(weight[indices]).all() or not ((weight[indices] >= 0) & (weight[indices] <= 1)).all():
        raise AssertionError("Invalid supervision weight")
    device = torch.device("cuda:0")
    model = model_for(config, int(spec["relation_dim"])).to(device)
    logits, alpha = forward_model(model, cache, indices, device)
    weights = weight[indices].to(device)
    cf_loss = (weights * (alpha - tau[indices].to(device)).square()).sum() / (weights.sum() + EPS_NUM)
    labels = torch.tensor([int(cache["metadata"][i]["y_true"]) for i in indices.tolist()], device=device)
    loss = F.cross_entropy(logits, labels) + float(spec["lambda_cf"]) * cf_loss
    loss.backward()
    if not torch.isfinite(loss) or not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise AssertionError("Smoke gradient failure")
    frozen = verify_stage_a_frozen(root)
    result = {"passed": True, "experiment": spec["name"], "supervision": spec["supervision"], "supervision_path": str(source), "lambda_cf": spec["lambda_cf"], "lambda_fallback": config["lambda_fallback"], "relation_dim": model.gate[0].in_features, "stage_a": frozen, "weighted_loss_formula": "sum(w*(alpha-tau)^2)/(sum(w)+1e-8)", "loss": float(loss.detach()), "cf_loss": float(cf_loss.detach())}
    atomic_json(output / "smoke/smoke_test.json", result)
    print(f"SMOKE {spec['name']} PASSED {json.dumps(result, sort_keys=True)}", flush=True)


def train_experiment(config, root: Path, spec: dict) -> Path:
    seed_everything(int(config["seed"]))
    output = root / "results" / spec["name"]
    best_path = output / "checkpoints/best.pt"
    if best_path.exists():
        raise FileExistsError(f"Refusing to overwrite {best_path}")
    cache = load_cache(root, "eyepacs")
    train, val = split_indices(cache, "train"), split_indices(cache, "val")
    tau, confidence, source = supervision_tensors(root, cache, spec["supervision"])
    device = torch.device("cuda:0")
    model = model_for(config, int(spec["relation_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["stage_b_lr"]), weight_decay=float(config["stage_b_wd"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["stage_b_epochs"]))
    train_tensor = torch.tensor(train, dtype=torch.long)
    best_key, best_epoch, history = (-float("inf"),) * 3, 0, []
    for epoch in range(1, int(config["stage_b_epochs"]) + 1):
        started = time.time(); model.train()
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        loader = DataLoader(TensorDataset(torch.arange(len(train))), batch_size=int(config["stage_b_batch"]), shuffle=True, generator=generator)
        random_map = random_fellow_mapping(cache, train, int(config["random_fellow_seed"]) + epoch)
        total_values, grade_values, cf_values, fallback_values = [], [], [], []
        for (local,) in loader:
            indices = train_tensor[local]
            labels = torch.tensor([int(cache["metadata"][i]["y_true"]) for i in indices.tolist()], dtype=torch.long, device=device)
            logits, alpha = forward_model(model, cache, indices, device)
            grade_loss = F.cross_entropy(logits, labels)
            weights = confidence[indices].to(device)
            cf_loss = (weights * (alpha - tau[indices].to(device)).square()).sum() / (weights.sum() + EPS_NUM)
            random_logits, _ = forward_model(model, cache, indices, device, random_map[local])
            target_prob = cache["p_target"][indices].to(device).detach()
            fallback_loss = F.kl_div(torch.log(random_logits.softmax(1).clamp_min(EPS_NUM)), target_prob.clamp_min(EPS_NUM), reduction="batchmean")
            loss = grade_loss + float(spec["lambda_cf"]) * cf_loss + float(config["lambda_fallback"]) * fallback_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError("Non-finite Stage-B gradient")
            optimizer.step()
            total_values.append(float(loss.detach())); grade_values.append(float(grade_loss.detach())); cf_values.append(float(cf_loss.detach())); fallback_values.append(float(fallback_loss.detach()))
        scheduler.step()
        val_metrics = evaluate_model(model, cache, val, device); key = metric_key(val_metrics)
        entry = {"epoch": epoch, "loss": float(np.mean(total_values)), "grade_loss": float(np.mean(grade_values)), "cf_loss": float(np.mean(cf_values)), "fallback_loss": float(np.mean(fallback_values)), "seconds": time.time() - started, "lr": scheduler.get_last_lr()[0], **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(entry)
        if key > best_key:
            best_key, best_epoch = key, epoch
            atomic_torch_save(best_path, {"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "experiment": spec["name"], "fresh_initialization": True, "relation_dim": spec["relation_dim"], "lambda_cf": spec["lambda_cf"], "supervision_path": str(source)})
        write_csv(output / "checkpoints/history.csv", history)
        print(f"{spec['name']} epoch={epoch}/{config['stage_b_epochs']} loss={entry['loss']:.6f} val_qwk={val_metrics['qwk']:.6f} seconds={entry['seconds']:.1f}", flush=True)
    atomic_json(output / "checkpoints/summary.json", {"best_epoch": best_epoch, "best_key": best_key, "lambda_cf": spec["lambda_cf"], "relation_dim": spec["relation_dim"]})
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    records = make_predictions(config, "eyepacs", "full", cache, split_indices(cache, "test"), model, device)
    for row in records:
        row["variant"] = spec["name"]
    path = output / "predictions/eyepacs_seed42_test.csv"
    write_csv(path, records, prediction_fields("bounded"))
    return path


@torch.inference_mode()
def coefficient_matrix(config, root: Path, checkpoint: Path, relation_dim: int, indices):
    cache = load_cache(root, "eyepacs"); device = torch.device("cuda:0")
    model = model_for(config, relation_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]); model.eval()
    output = []
    index_tensor = torch.tensor(indices, dtype=torch.long)
    for start in range(0, len(indices), 1024):
        batch = index_tensor[start:start + 1024]
        _, alpha = forward_model(model, cache, batch, device)
        output.append(alpha.cpu())
    return torch.cat(output)
