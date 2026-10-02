from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .core import (
    BOUNDARY_NAMES,
    atomic_json,
    atomic_torch_save,
    metric_key,
    reconstruct_logits,
    seed_everything,
    write_csv,
)
from .soft_tau_experiment import EPS_NUM, verify_stage_a_frozen
from .stage_b import load_cache, random_fellow_mapping, split_indices
from .three_stage_experiments import supervision_tensors


EXPERIMENT = "EXP_D2_UTILITY_STRENGTH"
LAMBDA_UTILITY = 0.2


class TwoHeadStageBGate(nn.Module):
    """Separate utility selection from correction amount."""

    def __init__(self, adapter_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(768, adapter_dim), nn.GELU(), nn.LayerNorm(adapter_dim)
        )
        relation_dim = adapter_dim * 2 + 10
        self.trunk = nn.Sequential(nn.Linear(relation_dim, hidden), nn.GELU())
        self.utility_head = nn.Linear(hidden, 4)
        self.amount_head = nn.Linear(hidden, 4)

    def components(self, h_target, h_fellow, p_target, p_fellow):
        target = self.adapter(h_target)
        fellow = self.adapter(h_fellow)
        relation = torch.cat(
            (target - fellow, target * fellow, p_target, p_fellow), dim=1
        )
        if relation.shape[1] != 522:
            raise AssertionError(f"D2 relation dimension {relation.shape[1]} != 522")
        shared = self.trunk(relation)
        utility = torch.sigmoid(self.utility_head(shared))
        amount = torch.sigmoid(self.amount_head(shared))
        alpha = utility * amount
        return utility, amount, alpha

    def forward(self, h_target, h_fellow, p_target, p_fellow, a_target, a_fellow):
        utility, amount, alpha = self.components(
            h_target, h_fellow, p_target, p_fellow
        )
        logits = reconstruct_logits(a_target + alpha * (a_fellow - a_target))
        return logits, alpha, utility, amount


def forward_model(model, cache, indices, device, fellow_indices=None):
    if fellow_indices is None:
        fellow_indices = indices
    values = []
    for key in (
        "h_target",
        "h_fellow",
        "p_target",
        "p_fellow",
        "a_target",
        "a_fellow",
    ):
        source = fellow_indices if key.endswith("fellow") else indices
        values.append(cache[key][source].to(device, non_blocking=True))
    return model(*values)


@torch.inference_mode()
def evaluate_model(model, cache, indices, device):
    from .core import metrics

    model.eval()
    predictions, labels = [], []
    tensor = torch.tensor(indices, dtype=torch.long)
    for (batch,) in DataLoader(TensorDataset(tensor), batch_size=1024):
        logits, _, _, _ = forward_model(model, cache, batch, device)
        predictions.extend(logits.argmax(1).cpu().tolist())
        labels.extend(int(cache["metadata"][index]["y_true"]) for index in batch.tolist())
    return metrics(np.asarray(labels), np.asarray(predictions))


def utility_loss(utility, tau, weight):
    return (weight * (utility - tau).square()).sum() / (weight.sum() + EPS_NUM)


def fallback_loss(model, cache, indices, random_indices, device):
    logits, _, _, _ = forward_model(model, cache, indices, device, random_indices)
    target_prob = cache["p_target"][indices].to(device).detach()
    return F.kl_div(
        torch.log(logits.softmax(1).clamp_min(EPS_NUM)),
        target_prob.clamp_min(EPS_NUM),
        reduction="batchmean",
    )


def smoke_test(config, root: Path, cache, tau, confidence, source):
    output = root / "results" / EXPERIMENT
    device = torch.device("cuda:0")
    indices = torch.tensor(split_indices(cache, "train")[:512], dtype=torch.long)
    model = TwoHeadStageBGate(
        int(config["adapter_dim"]), int(config["gate_hidden"])
    ).to(device)
    labels = torch.tensor(
        [int(cache["metadata"][i]["y_true"]) for i in indices.tolist()],
        dtype=torch.long,
        device=device,
    )
    logits, alpha, utility, amount = forward_model(model, cache, indices, device)
    weight = confidence[indices].to(device)
    target_tau = tau[indices].to(device)
    grade = F.cross_entropy(logits, labels)
    utility_value = utility_loss(utility, target_tau, weight)
    mapping = random_fellow_mapping(
        cache, split_indices(cache, "train"), int(config["random_fellow_seed"])
    )[: len(indices)]
    fallback = fallback_loss(model, cache, indices, mapping, device)
    total = grade + LAMBDA_UTILITY * utility_value + float(config["lambda_fallback"]) * fallback
    total.backward()
    frozen = verify_stage_a_frozen(root)
    assertions = {
        "passed": True,
        "experiment": EXPERIMENT,
        "stage_a": frozen,
        "stage_a_optimizer_excluded": True,
        "relation_dim": model.trunk[0].in_features,
        "shared_hidden_dim": model.trunk[0].out_features,
        "two_heads": True,
        "utility_head_outputs": 4,
        "amount_head_outputs": 4,
        "alpha_formula": "sigmoid(utility_head) * sigmoid(amount_head)",
        "utility_supervision_target": "binary endpoint utility",
        "utility_loss_formula": "sum(w*(utility-tau)^2)/(sum(w)+1e-8)",
        "amount_direct_supervision": False,
        "utility_loss_applied_to": "u_not_alpha",
        "fellow_gt_used_for_training": False,
        "lambda_utility": LAMBDA_UTILITY,
        "lambda_fallback": float(config["lambda_fallback"]),
        "harm_loss": False,
        "disagreement_features": False,
        "supervision_path": str(source),
        "finite": bool(
            torch.isfinite(total)
            and torch.isfinite(alpha).all()
            and torch.isfinite(utility).all()
            and torch.isfinite(amount).all()
        ),
        "alpha_identity_max_error": float((alpha - utility * amount).abs().max()),
        "loss_scale": {
            "grade": float(grade.detach()),
            "utility": float(utility_value.detach()),
            "fallback": float(fallback.detach()),
            "total": float(total.detach()),
        },
        "means": {
            "u": float(utility.mean().detach()),
            "s": float(amount.mean().detach()),
            "alpha": float(alpha.mean().detach()),
            "abs_delta_a": float((cache["a_fellow"][indices] - cache["a_target"][indices]).abs().mean()),
            "abs_correction": float((alpha.detach().cpu() * (cache["a_fellow"][indices] - cache["a_target"][indices])).abs().mean()),
        },
        "boundary_means": {
            name: {
                "u": float(utility[:, boundary].mean().detach()),
                "s": float(amount[:, boundary].mean().detach()),
                "alpha": float(alpha[:, boundary].mean().detach()),
                "abs_delta_a": float((cache["a_fellow"][indices, boundary] - cache["a_target"][indices, boundary]).abs().mean()),
                "abs_correction": float((alpha[:, boundary].detach().cpu() * (cache["a_fellow"][indices, boundary] - cache["a_target"][indices, boundary])).abs().mean()),
            }
            for boundary, name in enumerate(BOUNDARY_NAMES)
        },
    }
    if not (
        assertions["finite"]
        and assertions["alpha_identity_max_error"] == 0.0
        and frozen["all_requires_grad_false"]
        and model.trunk[0].in_features == 522
    ):
        raise AssertionError(f"D2 smoke assertion failed: {assertions}")
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("D2 smoke non-finite gradient")
    atomic_json(output / "smoke/smoke_test.json", assertions)
    print(f"D2 SMOKE PASSED {json.dumps(assertions, sort_keys=True)}", flush=True)


def train(config, root: Path, cache, tau, confidence, source):
    seed_everything(int(config["seed"]))
    output = root / "results" / EXPERIMENT
    best_path = output / "checkpoints/best.pt"
    if best_path.exists():
        raise FileExistsError(f"Refusing to overwrite {best_path}")
    train_indices = split_indices(cache, "train")
    val_indices = split_indices(cache, "val")
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    device = torch.device("cuda:0")
    model = TwoHeadStageBGate(
        int(config["adapter_dim"]), int(config["gate_hidden"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["stage_b_lr"]),
        weight_decay=float(config["stage_b_wd"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["stage_b_epochs"])
    )
    best_key, best_epoch, history = (-float("inf"),) * 3, 0, []
    for epoch in range(1, int(config["stage_b_epochs"]) + 1):
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
        values = {key: [] for key in ("loss", "grade", "utility", "fallback", "alpha", "amount")}
        for (local,) in loader:
            indices = train_tensor[local]
            labels = torch.tensor(
                [int(cache["metadata"][i]["y_true"]) for i in indices.tolist()],
                dtype=torch.long,
                device=device,
            )
            logits, alpha, utility, amount = forward_model(model, cache, indices, device)
            grade = F.cross_entropy(logits, labels)
            utility_value = utility_loss(
                utility, tau[indices].to(device), confidence[indices].to(device)
            )
            fallback = fallback_loss(model, cache, indices, random_map[local], device)
            loss = grade + LAMBDA_UTILITY * utility_value + float(config["lambda_fallback"]) * fallback
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or not all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise FloatingPointError("D2 non-finite loss/gradient")
            optimizer.step()
            for key, value in (
                ("loss", loss),
                ("grade", grade),
                ("utility", utility_value),
                ("fallback", fallback),
                ("alpha", alpha.mean()),
                ("amount", amount.mean()),
            ):
                values[key].append(float(value.detach()))
        scheduler.step()
        val = evaluate_model(model, cache, val_indices, device)
        key = metric_key(val)
        entry = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{name}": float(np.mean(series)) for name, series in values.items()},
            **{f"val_{name}": value for name, value in val.items()},
        }
        history.append(entry)
        if key > best_key:
            best_key, best_epoch = key, epoch
            atomic_torch_save(
                best_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "experiment": EXPERIMENT,
                    "fresh_initialization": True,
                    "relation_dim": 522,
                    "lambda_utility": LAMBDA_UTILITY,
                    "lambda_fallback": config["lambda_fallback"],
                    "alpha_formula": "utility * amount",
                    "supervision_path": str(source),
                },
            )
        write_csv(output / "checkpoints/history.csv", history)
        print(
            f"{EXPERIMENT} epoch={epoch}/{config['stage_b_epochs']} "
            f"loss={entry['train_loss']:.6f} grade={entry['train_grade']:.6f} "
            f"utility={entry['train_utility']:.6f} fallback={entry['train_fallback']:.6f} "
            f"alpha={entry['train_alpha']:.6f} amount={entry['train_amount']:.6f} "
            f"val_qwk={val['qwk']:.6f} seconds={entry['seconds']:.1f}",
            flush=True,
        )
    atomic_json(
        output / "checkpoints/summary.json",
        {
            "best_epoch": best_epoch,
            "best_key": list(best_key),
            "selection_data": "validation_only",
            "test_used_for_selection": False,
        },
    )
    return best_path


@torch.inference_mode()
def predict(config, root: Path, cache, checkpoint: Path):
    output = root / "results" / EXPERIMENT
    device = torch.device("cuda:0")
    model = TwoHeadStageBGate(
        int(config["adapter_dim"]), int(config["gate_hidden"])
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    rows = []
    indices = torch.tensor(split_indices(cache, "test"), dtype=torch.long)
    for (batch,) in DataLoader(TensorDataset(indices), batch_size=1024):
        logits, alpha, utility, amount = forward_model(model, cache, batch, device)
        probabilities = logits.softmax(1)
        predictions = logits.argmax(1)
        for local, index in enumerate(batch.tolist()):
            metadata = cache["metadata"][index]
            row = {
                "dataset": "eyepacs",
                "seed": config["seed"],
                "sample_id": metadata["sample_id"],
                "patient_id": metadata["patient_id"],
                "target_eye": metadata["target_eye"],
                "fellow_eye": metadata["fellow_eye"],
                "y_true": int(metadata["y_true"]),
                "fellow_y_true": int(metadata["fellow_y_true"]),
                "pred_grade": int(predictions[local]),
                "target_only_pred": int(cache["z_target"][index].argmax()),
                "final_pred": int(predictions[local]),
                "variant": EXPERIMENT,
            }
            row.update({f"logit_{i}": float(logits[local, i]) for i in range(5)})
            row.update({f"prob_{i}": float(probabilities[local, i]) for i in range(5)})
            for boundary, name in enumerate(BOUNDARY_NAMES):
                row[f"alpha_{name}"] = float(alpha[local, boundary])
                row[f"utility_{name}"] = float(utility[local, boundary])
                row[f"amount_{name}"] = float(amount[local, boundary])
            rows.append(row)
    fields = list(rows[0])
    path = output / "predictions/eyepacs_seed42_test.csv"
    write_csv(path, rows, fields)
    atomic_json(
        output / "complete.json",
        {
            "experiment": EXPERIMENT,
            "checkpoint": str(checkpoint),
            "prediction": str(path),
            "best_epoch": state["epoch"],
            "validation": state["val"],
            "test_rows": len(rows),
        },
    )
    return path


def run(config, root: Path):
    output = root / "results" / EXPERIMENT
    if output.exists():
        raise FileExistsError(f"Refusing to overlap D2 output: {output}")
    cache = load_cache(root, "eyepacs")
    tau, confidence, source = supervision_tensors(root, cache, "utility")
    smoke_test(config, root, cache, tau, confidence, source)
    checkpoint = train(config, root, cache, tau, confidence, source)
    return predict(config, root, cache, checkpoint)
