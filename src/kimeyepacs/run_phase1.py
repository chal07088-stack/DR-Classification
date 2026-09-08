import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from .data import (
    PairedEyeDataset,
    SingleEyeDataset,
    boundary_pos_weight,
    class_counts,
    limit_patients,
    ordinal_target,
    scan_all,
)
from .engine import dump_predictions, infer_paired, infer_single, make_loader, train_ordinal
from .gates import BoundaryGate, apply_alpha, gate_features
from .metrics import grade_metrics, save_npz, subgroup_metrics
from .models import PairedOrdinal, SingleOrdinal, total_param_count, trainable_param_count
from .runtime import append_jsonl, configure_torch, device_and_amp, ensure_dir, load_config, save_json, seed_everything
from .teachers import confidence, counterfactual_teacher, loss_guided_teacher, one_sided_alpha, ordinal_target_np


def _loader_cfg(cfg):
    r = cfg["runtime"]
    return {
        "workers": int(r.get("num_workers", 4)),
        "pin_memory": bool(r.get("pin_memory", True)),
        "persistent_workers": bool(r.get("persistent_workers", True)),
        "prefetch_factor": int(r.get("prefetch_factor", 2)),
    }


def _split_items(cfg, seed, max_train_patients, max_val_patients, max_test_patients):
    items = scan_all(cfg["data_root"])
    items["train"] = limit_patients(items["train"], max_train_patients, seed)
    items["val"] = limit_patients(items["val"], max_val_patients, seed + 1)
    items["test"] = limit_patients(items["test"], max_test_patients, seed + 2)
    return items


def _datasets(cfg, items, paired=False):
    cls = PairedEyeDataset if paired else SingleEyeDataset
    m = cfg["model"]
    hflip = bool(cfg.get("augmentation", {}).get("horizontal_flip", True))
    return {
        split: cls(items[split], int(m["image_size"]), split == "train", hflip, m.get("color_mode", "grayscale_reference"))
        for split in ("train", "val", "test")
    }


def _single_loaders(cfg, items):
    opt = cfg["optimization"]
    ds = _datasets(cfg, items, paired=False)
    rt = _loader_cfg(cfg)
    return {
        split: make_loader(ds[split], int(opt["single_batch"]), split == "train", **rt)
        for split in ("train", "val", "test")
    }


def _paired_loaders(cfg, items):
    opt = cfg["optimization"]
    ds = _datasets(cfg, items, paired=True)
    rt = _loader_cfg(cfg)
    return {
        split: make_loader(ds[split], int(opt["paired_batch"]), split == "train", **rt)
        for split in ("train", "val", "test")
    }


def _fit_single(cfg, items, loaders, device, amp_dtype, pos_weight, out_dir, tag):
    opt = cfg["optimization"]
    model = SingleOrdinal(cfg["model"]["arch"], bool(cfg["model"].get("pretrained", True)))
    print(f"{tag}: total={total_param_count(model):,} trainable={trainable_param_count(model):,}", flush=True)
    groups = [
        {"params": model.encoder.parameters(), "lr": float(opt["backbone_lr"])},
        {"params": model.head.parameters(), "lr": float(opt["head_lr"])},
    ]
    model, best = train_ordinal(
        model,
        loaders["train"],
        loaders["val"],
        items["val"],
        device,
        amp_dtype,
        pos_weight,
        int(opt["single_epochs"]),
        int(opt["min_epochs"]),
        int(opt["patience"]),
        int(opt["min_save_epoch"]),
        groups,
        float(opt["weight_decay"]),
        int(opt["single_grad_accum"]),
        float(cfg["runtime"].get("grad_clip", 1.0)),
        float(opt.get("ema_decay", 0.0)),
        out_dir / "logs" / f"{tag}.jsonl",
        tag,
        paired=False,
    )
    return model, best


def _fit_paired(cfg, single, items, loaders, device, amp_dtype, pos_weight, out_dir, tag):
    opt = cfg["optimization"]
    model = PairedOrdinal(single.encoder, single.num_features)
    print(f"{tag}: total={total_param_count(model):,} trainable={trainable_param_count(model):,}", flush=True)
    groups = [{"params": model.head.parameters(), "lr": float(opt["paired_head_lr"])}]
    model, best = train_ordinal(
        model,
        loaders["train"],
        loaders["val"],
        items["val"],
        device,
        amp_dtype,
        pos_weight,
        int(opt["paired_epochs"]),
        int(opt["min_epochs"]),
        int(opt["patience"]),
        int(opt["min_save_epoch"]),
        groups,
        float(opt["weight_decay"]),
        int(opt["paired_grad_accum"]),
        float(cfg["runtime"].get("grad_clip", 1.0)),
        float(opt.get("ema_decay", 0.0)),
        out_dir / "logs" / f"{tag}.jsonl",
        tag,
        paired=True,
    )
    return model, best


def _save_summary_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _patient_fold_labels(train_items):
    by_patient = {}
    for item in train_items:
        by_patient.setdefault(item.patient, []).append(item.label)
    patients = np.asarray(sorted(by_patient, key=int))
    labels = np.asarray([max(by_patient[p]) for p in patients])
    return patients, labels


def _items_by_patients(items, patients):
    keep = set(patients)
    return [x for x in items if x.patient in keep]


def _oof_arrays(items):
    n = len(items)
    return np.full((n, 4), np.nan, dtype=np.float32), np.full(n, -1, dtype=np.int64)


def _write_oof(path, items, q):
    idx = np.arange(len(items))
    save_npz(path, items, q, idx)


def _make_eval_loader(dataset, batch_size, cfg):
    rt = _loader_cfg(cfg)
    return make_loader(dataset, batch_size, False, **rt)


def run_oof(cfg, items, seed, folds, device, amp_dtype, run_dir):
    patients, fold_labels = _patient_fold_labels(items["train"])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    q_single_oof, fold_id = _oof_arrays(items["train"])
    q_paired_oof, _ = _oof_arrays(items["train"])
    pos_weight_full = boundary_pos_weight(items["train"], float(cfg["optimization"].get("max_pos_weight", 20.0)))

    index_by_key = {(x.patient, x.side): i for i, x in enumerate(items["train"])}
    fold_rows = []

    for fold, (tr_idx, ho_idx) in enumerate(splitter.split(patients, fold_labels), start=1):
        fold_seed = seed * 100 + fold
        seed_everything(fold_seed)
        train_patients = patients[tr_idx]
        hold_patients = patients[ho_idx]
        fold_items = {
            "train": _items_by_patients(items["train"], train_patients),
            "val": items["val"],
            "test": _items_by_patients(items["train"], hold_patients),
        }
        fold_dir = ensure_dir(run_dir / "oof" / f"fold{fold}")
        save_json(fold_dir / "fold_summary.json", {
            "fold": fold,
            "seed": fold_seed,
            "train_patients": int(len(train_patients)),
            "holdout_patients": int(len(hold_patients)),
            "train_class_counts": class_counts(fold_items["train"]),
            "holdout_class_counts": class_counts(fold_items["test"]),
        })
        single_loaders = _single_loaders(cfg, fold_items)
        single, single_best = _fit_single(cfg, fold_items, single_loaders, device, amp_dtype, pos_weight_full, fold_dir, f"oof{fold}_single")
        torch.save(single.state_dict(), fold_dir / "single_best.pt")

        hold_single_ds = SingleEyeDataset(
            fold_items["test"],
            int(cfg["model"]["image_size"]),
            False,
            False,
            cfg["model"].get("color_mode", "grayscale_reference"),
        )
        hold_single_loader = _make_eval_loader(hold_single_ds, int(cfg["optimization"]["single_batch"]), cfg)
        q_s, idx_s = infer_single(single, hold_single_loader, device, amp_dtype)
        for row_q, local_i in zip(q_s, idx_s):
            item = fold_items["test"][int(local_i)]
            global_i = index_by_key[(item.patient, item.side)]
            q_single_oof[global_i] = row_q
            fold_id[global_i] = fold

        paired_loaders = _paired_loaders(cfg, fold_items)
        paired, paired_best = _fit_paired(cfg, single, fold_items, paired_loaders, device, amp_dtype, pos_weight_full, fold_dir, f"oof{fold}_paired")
        torch.save(paired.state_dict(), fold_dir / "paired_best.pt")

        hold_pair_ds = PairedEyeDataset(
            fold_items["test"],
            int(cfg["model"]["image_size"]),
            False,
            False,
            cfg["model"].get("color_mode", "grayscale_reference"),
        )
        hold_pair_loader = _make_eval_loader(hold_pair_ds, int(cfg["optimization"]["paired_batch"]), cfg)
        q_f, idx_f = infer_paired(paired, hold_pair_loader, device, amp_dtype)
        for row_q, local_i in zip(q_f, idx_f):
            item = fold_items["test"][int(local_i)]
            global_i = index_by_key[(item.patient, item.side)]
            q_paired_oof[global_i] = row_q
        fold_rows.append({
            "fold": fold,
            "single_best_val_qwk": single_best.get("score", ""),
            "paired_best_val_qwk": paired_best.get("score", ""),
        })
        torch.cuda.empty_cache()

    if np.isnan(q_single_oof).any() or np.isnan(q_paired_oof).any():
        raise RuntimeError("OOF predictions contain missing rows")
    oof_dir = ensure_dir(run_dir / "oof")
    _write_oof(oof_dir / "single_train_oof.npz", items["train"], q_single_oof)
    _write_oof(oof_dir / "paired_train_oof.npz", items["train"], q_paired_oof)
    np.save(oof_dir / "fold_id.npy", fold_id)
    _save_summary_csv(oof_dir / "fold_summary.csv", fold_rows)
    return q_single_oof, q_paired_oof


def _load_pred_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["q"], d["y"], d["gap"]


def _fellow_pred_from_npz(path):
    d = np.load(path, allow_pickle=True)
    q = d["q"]
    pred = (q >= 0.5).sum(axis=1)
    key = [(str(p), str(s)) for p, s in zip(d["patient"], d["side"])]
    by = {k: v for k, v in zip(key, pred)}
    other = []
    for p, s in key:
        fs = "right" if s == "left" else "left"
        other.append(by.get((p, fs), pred[len(other)]))
    return np.asarray(other)


def _fellow_pred_from_items(items, pred):
    by = {(x.patient, x.side): int(p) for x, p in zip(items, pred)}
    out = []
    for item in items:
        fellow_side = "right" if item.side == "left" else "left"
        out.append(by.get((item.patient, fellow_side), int(pred[len(out)])))
    return np.asarray(out)


def _train_gate_supervised(features, target, weights, epochs, batch_size, seed, out_path, tag):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryGate(features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    n = len(x)
    for ep in range(1, epochs + 1):
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb, wb = x[idx].to(device), y[idx].to(device), w[idx].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = ((pred - yb) ** 2 * wb).mean()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
        print(f"[{tag}] gate epoch {ep:03d} loss {total / n:.6f}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return model


def _train_gate_e2e(features, q_single, q_fusion, y, epochs, batch_size, seed, out_path):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryGate(features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.tensor(features, dtype=torch.float32)
    qs = torch.tensor(q_single, dtype=torch.float32)
    qf = torch.tensor(q_fusion, dtype=torch.float32)
    target = torch.tensor(ordinal_target_np(y), dtype=torch.float32)
    crit = torch.nn.BCELoss()
    n = len(y)
    for ep in range(1, epochs + 1):
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx_eye = order[start:start + batch_size]
            idx_feat = torch.cat([idx_eye + b * n for b in range(4)])
            alpha = model(x[idx_feat].to(device)).reshape(4, len(idx_eye)).T
            q = qf[idx_eye].to(device) + alpha * (qs[idx_eye].to(device) - qf[idx_eye].to(device))
            q = torch.clamp(q, 0.0, 1.0)
            q = torch.cummin(q, dim=1).values
            loss = crit(q, target[idx_eye].to(device)) + 0.01 * alpha.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx_eye)
        print(f"[D_e2e] gate epoch {ep:03d} loss {total / n:.6f}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return model


def _predict_gate(model, features, batch_size):
    device = next(model.parameters()).device
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def run_gates(cfg, run_dir, items, q_single_oof, q_paired_oof, seed):
    y_train = np.asarray([x.label for x in items["train"]], dtype=np.int64)
    gap_train = np.asarray([abs(x.label - x.fellow_label) for x in items["train"]], dtype=np.int64)
    pred_train = (q_single_oof >= 0.5).sum(axis=1)
    fellow_train = _fellow_pred_from_items(items["train"], pred_train)
    features = gate_features(q_single_oof, q_paired_oof, fellow_train)

    teacher_dir = ensure_dir(run_dir / "teachers")
    gate_dir = ensure_dir(run_dir / "gates")
    metric_dir = ensure_dir(run_dir / "metrics")
    signal_a = loss_guided_teacher(q_single_oof, q_paired_oof, y_train)
    signal_b = counterfactual_teacher(q_single_oof, q_paired_oof, y_train)
    alpha_a = one_sided_alpha(signal_a, float(cfg["gate"].get("tau", 1.0)))
    alpha_b = one_sided_alpha(signal_b, float(cfg["gate"].get("tau", 1.0)))
    conf_b = confidence(signal_b, float(cfg["gate"].get("gamma", 1.0)))
    np.savez_compressed(
        teacher_dir / "teacher_targets_train_oof.npz",
        loss_guided_signal=signal_a,
        counterfactual_signal=signal_b,
        alpha_loss_guided=alpha_a,
        alpha_counterfactual=alpha_b,
        confidence_counterfactual=conf_b,
        y=y_train,
        gap=gap_train,
    )

    corr_rows = []
    for b in range(4):
        a = signal_a[:, b]
        c = signal_b[:, b]
        if np.std(a) > 0 and np.std(c) > 0:
            pearson = float(np.corrcoef(a, c)[0, 1])
        else:
            pearson = float("nan")
        disagree = float(np.mean((a > 0) != (c > 0)))
        corr_rows.append({"boundary": b, "pearson": pearson, "rollback_disagreement_rate": disagree})
    _save_summary_csv(metric_dir / "teacher_a_vs_b.csv", corr_rows)

    gate_epochs = int(cfg["optimization"].get("gate_epochs", 40))
    gate_batch = int(cfg["optimization"].get("gate_batch", 512))
    ones = np.ones(features.shape[0], dtype=np.float32)
    global_alpha = alpha_b.mean(axis=1)
    model_c = _train_gate_supervised(features, np.tile(global_alpha, 4), ones, gate_epochs, gate_batch, seed + 10, gate_dir / "C_global_cf_gate.pt", "C_global")
    model_d = _train_gate_e2e(features, q_single_oof, q_paired_oof, y_train, gate_epochs, gate_batch, seed + 20, gate_dir / "D_boundary_e2e.pt")
    model_e = _train_gate_supervised(features, alpha_a.T.reshape(-1), ones, gate_epochs, gate_batch, seed + 30, gate_dir / "E_loss_guided.pt", "E_loss_guided")
    model_f = _train_gate_supervised(features, alpha_b.T.reshape(-1), ones, gate_epochs, gate_batch, seed + 40, gate_dir / "F_counterfactual.pt", "F_counterfactual")
    model_g = _train_gate_supervised(features, alpha_b.T.reshape(-1), conf_b.T.reshape(-1), gate_epochs, gate_batch, seed + 50, gate_dir / "G_counterfactual_conf.pt", "G_counterfactual_conf")

    rows = []
    rows.extend(_eval_saved_predictions(run_dir, "single", "A_single"))
    rows.extend(_eval_saved_predictions(run_dir, "paired", "B_paired"))
    for model, arm in [
        (model_c, "C_global_gate"),
        (model_d, "D_boundary_e2e"),
        (model_e, "E_loss_guided_gate"),
        (model_f, "F_counterfactual_gate"),
        (model_g, "G_counterfactual_conf_gate"),
    ]:
        rows.extend(_eval_gate_arm(cfg, run_dir, model, arm))
    _save_summary_csv(metric_dir / "phase1_ablation_summary.csv", rows)


def _eval_saved_predictions(run_dir, tag, arm):
    rows = []
    for split in ("val", "test"):
        q, y, gap = _load_pred_npz(run_dir / "predictions" / f"{tag}_{split}.npz")
        row = {"arm": arm, "split": split, **subgroup_metrics(y, q, gap, arm)}
        rows.append(row)
    return rows


def _eval_gate_arm(cfg, run_dir, model, arm):
    rows = []
    gate_batch = int(cfg["optimization"].get("gate_batch", 512))
    for split in ("val", "test"):
        s = np.load(run_dir / "predictions" / f"single_{split}.npz", allow_pickle=True)
        f = np.load(run_dir / "predictions" / f"paired_{split}.npz", allow_pickle=True)
        fellow = _fellow_pred_from_npz(run_dir / "predictions" / f"single_{split}.npz")
        feats = gate_features(s["q"], f["q"], fellow)
        alpha = _predict_gate(model, feats, gate_batch)
        q = apply_alpha(s["q"], f["q"], alpha)
        np.savez_compressed(
            run_dir / "predictions" / f"{arm}_{split}.npz",
            q=q.astype(np.float32),
            alpha=alpha.astype(np.float32),
            y=s["y"],
            patient=s["patient"],
            side=s["side"],
            fellow_y=s["fellow_y"],
            gap=s["gap"],
        )
        rows.append({"arm": arm, "split": split, **subgroup_metrics(s["y"], q, s["gap"], arm)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default="phase1_seed42_convnext_s")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--paired-epochs", type=int, default=None)
    ap.add_argument("--gate-epochs", type=int, default=None)
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
    if args.gate_epochs:
        cfg["optimization"]["gate_epochs"] = args.gate_epochs
    if args.pretrained is not None:
        cfg["model"]["pretrained"] = args.pretrained == "true"
    if args.num_workers is not None:
        cfg["runtime"]["num_workers"] = args.num_workers
        cfg["runtime"]["persistent_workers"] = args.num_workers > 0

    configure_torch(cfg)
    seed_everything(args.seed)
    device, amp_dtype = device_and_amp(cfg["runtime"].get("precision", "bf16"))
    run_dir = ensure_dir(Path(cfg["out_root"]) / args.run_name)
    for d in ("checkpoints", "logs", "metrics", "predictions", "teachers", "gates", "oof"):
        ensure_dir(run_dir / d)
    shutil.copy2(args.config, run_dir / "source_config.yaml")
    save_json(run_dir / "resolved_config.json", cfg)

    items = _split_items(cfg, args.seed, args.max_train_patients, args.max_val_patients, args.max_test_patients)
    save_json(run_dir / "dataset_summary.json", {
        split: {"eyes": len(v), "patients": len(v) // 2, "class_counts": class_counts(v)}
        for split, v in items.items()
    })
    print(f"PHASE1 run_dir={run_dir}", flush=True)
    print(f"device={device} amp_dtype={amp_dtype}", flush=True)

    pos_weight = None
    if cfg["optimization"].get("boundary_pos_weight", True):
        pos_weight = boundary_pos_weight(items["train"], float(cfg["optimization"].get("max_pos_weight", 20.0)))
        print(f"boundary_pos_weight={pos_weight.tolist()}", flush=True)

    single_loaders = _single_loaders(cfg, items)
    single, single_best = _fit_single(cfg, items, single_loaders, device, amp_dtype, pos_weight, run_dir, "final_single")
    torch.save(single.state_dict(), run_dir / "checkpoints" / "final_single_best.pt")
    dump_predictions(single, single_loaders, items, device, amp_dtype, run_dir / "predictions", "single", paired=False)

    paired_loaders = _paired_loaders(cfg, items)
    paired, paired_best = _fit_paired(cfg, single, items, paired_loaders, device, amp_dtype, pos_weight, run_dir, "final_paired")
    torch.save(paired.state_dict(), run_dir / "checkpoints" / "final_paired_best.pt")
    dump_predictions(paired, paired_loaders, items, device, amp_dtype, run_dir / "predictions", "paired", paired=True)

    q_single_oof, q_paired_oof = run_oof(cfg, items, args.seed, args.folds, device, amp_dtype, run_dir)
    run_gates(cfg, run_dir, items, q_single_oof, q_paired_oof, args.seed)
    append_jsonl(run_dir / "logs" / "phase1_status.jsonl", {"status": "complete", "time": time.time(), "run_dir": str(run_dir)})
    print(f"PHASE1 complete: {run_dir / 'metrics' / 'phase1_ablation_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
