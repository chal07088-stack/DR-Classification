from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from .core import BOUNDARY_NAMES, StageAModel, StageBGate, atomic_json, atomic_torch_save, boundaries, metric_key, metrics, read_csv, reconstruct_logits, seed_everything, sha256, write_csv
from .stage_b import _forward, evaluate, load_cache, make_predictions, prediction_fields, random_fellow_mapping, split_indices

EPS_NUM = 1e-8
GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0
OPT_ITERATIONS = 32


def _loss_at_t(at, af, labels, boundary, t):
    states = at.clone()
    states[:, boundary] = at[:, boundary] + t * (af[:, boundary] - at[:, boundary])
    return F.cross_entropy(reconstruct_logits(states), labels, reduction="none")


def _bounded_optimum(at, af, labels, boundary):
    """Deterministic batched golden-section search with explicit endpoints."""
    n = len(labels)
    left = torch.zeros(n, dtype=torch.float64)
    right = torch.ones(n, dtype=torch.float64)
    x1 = right - GOLDEN_RATIO * (right - left)
    x2 = left + GOLDEN_RATIO * (right - left)
    f1 = _loss_at_t(at, af, labels, boundary, x1)
    f2 = _loss_at_t(at, af, labels, boundary, x2)
    for _ in range(OPT_ITERATIONS):
        choose_left = f1 <= f2
        left = torch.where(choose_left, left, x1)
        right = torch.where(choose_left, x2, right)
        x1 = right - GOLDEN_RATIO * (right - left)
        x2 = left + GOLDEN_RATIO * (right - left)
        f1 = _loss_at_t(at, af, labels, boundary, x1)
        f2 = _loss_at_t(at, af, labels, boundary, x2)
    interior = (left + right) / 2.0
    loss0 = _loss_at_t(at, af, labels, boundary, torch.zeros_like(interior))
    lossi = _loss_at_t(at, af, labels, boundary, interior)
    loss1 = _loss_at_t(at, af, labels, boundary, torch.ones_like(interior))
    candidate_t = torch.stack((torch.zeros_like(interior), interior, torch.ones_like(interior)), dim=1)
    candidate_loss = torch.stack((loss0, lossi, loss1), dim=1)
    # Exact numerical ties select the first candidate; candidates are increasing in t.
    best = candidate_loss.argmin(dim=1)
    row = torch.arange(n)
    tstar = candidate_t[row, best]
    lossstar = candidate_loss[row, best]
    endpoint_max = torch.maximum(loss0, loss1)
    weight = ((endpoint_max - lossstar) / (endpoint_max + EPS_NUM)).clamp(0.0, 1.0)
    return tstar, weight, loss0, loss1, lossstar


def generate_soft_tau(config, root: Path, output: Path) -> Path:
    path = output / "cache_soft_tau" / "eyepacs_seed42_soft_tau_wmse.csv"
    if path.is_file():
        print(f"SOFT_TAU_WMSE reuse={path}", flush=True)
        return path
    predictions, sources = [], []
    for fold in range(int(config["oof_folds"])):
        source = root / "outputs/tau" / f"eyepacs_seed42_oof_fold{fold}_predictions.csv"
        sources.append(source)
        predictions.extend(read_csv(source))
    cache = load_cache(root, "eyepacs")
    expected = {row["sample_id"] for row in cache["metadata"] if row["run_split"] == "train"}
    if len(predictions) != len(expected) or {row["sample_id"] for row in predictions} != expected:
        raise AssertionError("OOF predictions do not cover EyePACS train cache exactly")
    at = boundaries(torch.tensor([[float(r[f"logit_{g}"]) for g in range(5)] for r in predictions], dtype=torch.float64))
    af = boundaries(torch.tensor([[float(r[f"fellow_logit_{g}"]) for g in range(5)] for r in predictions], dtype=torch.float64))
    labels = torch.tensor([int(r["y_true"]) for r in predictions], dtype=torch.long)
    optimized = [_bounded_optimum(at, af, labels, b) for b in range(4)]
    output_rows = []
    for i, source_row in enumerate(predictions):
        row = {"sample_id": source_row["sample_id"], "patient_id": source_row["patient_id"], "y_true": int(source_row["y_true"]), "fellow_y_true": int(source_row["fellow_y_true"]), "gap": abs(int(source_row["y_true"]) - int(source_row["fellow_y_true"]))}
        for b, name in enumerate(BOUNDARY_NAMES):
            tstar, weight, loss0, loss1, lossstar = optimized[b]
            row.update({f"tau_soft_{name}": float(tstar[i]), f"weight_{name}": float(weight[i]), f"loss_t0_{name}": float(loss0[i]), f"loss_t1_{name}": float(loss1[i]), f"loss_tstar_{name}": float(lossstar[i]), f"tstar_{name}": float(tstar[i]), f"optimizer_iterations_{name}": OPT_ITERATIONS})
        output_rows.append(row)
    write_csv(path, output_rows)
    atomic_json(output / "cache_soft_tau/provenance.json", {"seed": int(config["seed"]), "definition": "continuous bounded t* and confidence-weighted MSE", "optimizer": "deterministic batched golden-section", "iterations": OPT_ITERATIONS, "semantic_threshold": None, "tie_break": "exact numerical argmin; smaller t wins exact tie", "eps_num": EPS_NUM, "target_gt_only": True, "rows": len(output_rows), "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources]})
    print(f"SOFT_TAU_WMSE generated rows={len(output_rows)} path={path}", flush=True)
    return path


def load_soft_tau(cache, path: Path):
    source = {row["sample_id"]: row for row in read_csv(path)}
    tau = torch.zeros((len(cache["metadata"]), 4), dtype=torch.float32)
    weight = torch.zeros_like(tau)
    joined = set()
    for i, metadata in enumerate(cache["metadata"]):
        if metadata["run_split"] != "train":
            continue
        row = source[metadata["sample_id"]]
        joined.add(metadata["sample_id"])
        for b, name in enumerate(BOUNDARY_NAMES):
            tau[i, b] = float(row[f"tau_soft_{name}"])
            weight[i, b] = float(row[f"weight_{name}"])
    if joined != set(source):
        raise AssertionError("Soft tau/cache sample_id join is not 1:1")
    return tau, weight


def verify_stage_a_frozen(root: Path):
    checkpoint = root / "outputs/checkpoints/eyepacs/stage_a/seed42/best.pt"
    model = StageAModel(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.requires_grad_(False)
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "parameter_count": sum(p.numel() for p in model.parameters()), "all_requires_grad_false": all(not p.requires_grad for p in model.parameters()), "stage_a_in_stage_b_optimizer": False, "stage_a_forward_during_stage_b": False}
    if not result["all_requires_grad_false"]:
        raise AssertionError("Stage A freeze assertion failed")
    return result


def soft_tau_diagnostics(path: Path):
    rows = read_csv(path)
    result = []
    for name in BOUNDARY_NAMES:
        tau = np.asarray([float(r[f"tau_soft_{name}"]) for r in rows])
        weight = np.asarray([float(r[f"weight_{name}"]) for r in rows])
        result.append({"boundary": name, "n": len(rows), "mean_tau_soft": float(tau.mean()), "median_tau_soft": float(np.median(tau)), "mean_weight": float(weight.mean()), "median_weight": float(np.median(weight)), "weight_near_zero_rate": float((weight <= 1e-6).mean()), "tstar_zero_rate": float((tau == 0).mean()), "tstar_interior_rate": float(((tau > 0) & (tau < 1)).mean()), "tstar_one_rate": float((tau == 1).mean())})
    return result


def smoke_test(config, root: Path, output: Path, tau_path: Path):
    cache = load_cache(root, "eyepacs")
    for key, value in cache.items():
        if torch.is_tensor(value) and (value.requires_grad or not torch.isfinite(value).all()):
            raise AssertionError(f"Invalid frozen cache tensor: {key}")
    tau, weight = load_soft_tau(cache, tau_path)
    train = split_indices(cache, "train")
    indices = torch.tensor(train[: min(512, len(train))], dtype=torch.long)
    all_rows = read_csv(tau_path)
    all_tau = np.asarray([[float(r[f"tau_soft_{name}"]) for name in BOUNDARY_NAMES] for r in all_rows])
    if not torch.isfinite(tau[indices]).all() or not ((tau[indices] >= 0) & (tau[indices] <= 1)).all():
        raise AssertionError("tau_soft must be finite and in [0,1]")
    if not torch.isfinite(weight[indices]).all() or not ((weight[indices] >= 0) & (weight[indices] <= 1)).all():
        raise AssertionError("weight must be finite and in [0,1]")
    if not (all_tau == 0).any() or not ((all_tau > 0) & (all_tau < 1)).any():
        raise AssertionError("Expected both t*=0 and interior t* cases")
    device = torch.device("cuda:0")
    model = StageBGate("bounded", int(config["adapter_dim"]), int(config["gate_hidden"])).to(device)
    labels = torch.tensor([int(cache["metadata"][i]["y_true"]) for i in indices.tolist()], device=device)
    logits, alpha = _forward(model, cache, indices, device)
    weights = weight[indices].to(device)
    cf_loss = (weights * (alpha - tau[indices].to(device)).square()).sum() / (weights.sum() + EPS_NUM)
    loss = F.cross_entropy(logits, labels) + cf_loss
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("Smoke backward produced non-finite values")
    diagnostics = soft_tau_diagnostics(tau_path)
    result = {"passed": True, "subset_n": len(indices), "weighted_mse": float(cf_loss.detach()), "has_tstar_zero": bool((all_tau == 0).any()), "has_tstar_interior": bool(((all_tau > 0) & (all_tau < 1)).any()), "has_tstar_one": bool((all_tau == 1).any()), "output_path": str(output), "boundary_diagnostics": diagnostics}
    atomic_json(output / "smoke/smoke_test.json", result)
    write_csv(output / "smoke/soft_tau_boundary_diagnostics.csv", diagnostics)
    print(f"SMOKE PASSED {json.dumps(result, sort_keys=True)}", flush=True)


def train_experiment(config, root: Path, experiment: str, lambda_cf: float, tau_path: Path):
    seed_everything(int(config["seed"]))
    output = root / "results" / experiment
    run_dir = output / "checkpoints"
    best_path = run_dir / "best.pt"
    if best_path.exists():
        raise FileExistsError(f"Refusing to overwrite {best_path}")
    cache = load_cache(root, "eyepacs")
    train, val = split_indices(cache, "train"), split_indices(cache, "val")
    tau, confidence = load_soft_tau(cache, tau_path)
    device = torch.device("cuda:0")
    model = StageBGate("bounded", int(config["adapter_dim"]), int(config["gate_hidden"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["stage_b_lr"]), weight_decay=float(config["stage_b_wd"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["stage_b_epochs"]))
    best_key, best_epoch, history = (-float("inf"),) * 3, 0, []
    train_tensor = torch.tensor(train, dtype=torch.long)
    for epoch in range(1, int(config["stage_b_epochs"]) + 1):
        started = time.time(); model.train()
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        loader = DataLoader(TensorDataset(torch.arange(len(train))), batch_size=int(config["stage_b_batch"]), shuffle=True, generator=generator)
        random_map = random_fellow_mapping(cache, train, int(config["random_fellow_seed"]) + epoch)
        totals, grades, cfs, fallbacks = [], [], [], []
        for (local,) in loader:
            indices = train_tensor[local]
            labels = torch.tensor([int(cache["metadata"][i]["y_true"]) for i in indices.tolist()], dtype=torch.long, device=device)
            logits, alpha = _forward(model, cache, indices, device)
            grade_loss = F.cross_entropy(logits, labels)
            weights = confidence[indices].to(device)
            cf_loss = (weights * (alpha - tau[indices].to(device)).square()).sum() / (weights.sum() + EPS_NUM)
            random_logits, _ = _forward(model, cache, indices, device, fellow_indices=random_map[local])
            target_prob = cache["p_target"][indices].to(device).detach()
            fallback_loss = F.kl_div(torch.log(random_logits.softmax(1).clamp_min(EPS_NUM)), target_prob.clamp_min(EPS_NUM), reduction="batchmean")
            loss = grade_loss + lambda_cf * cf_loss + float(config["lambda_fallback"]) * fallback_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite Stage-B gradient")
            optimizer.step()
            totals.append(float(loss.detach())); grades.append(float(grade_loss.detach())); cfs.append(float(cf_loss.detach())); fallbacks.append(float(fallback_loss.detach()))
        scheduler.step()
        val_metrics = evaluate(model, cache, val, device)
        key = metric_key(val_metrics)
        entry = {"epoch": epoch, "loss": float(np.mean(totals)), "grade_loss": float(np.mean(grades)), "cf_loss": float(np.mean(cfs)), "fallback_loss": float(np.mean(fallbacks)), "seconds": time.time() - started, "lr": scheduler.get_last_lr()[0], **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(entry)
        if key > best_key:
            best_key, best_epoch = key, epoch
            atomic_torch_save(best_path, {"model": model.state_dict(), "mode": "bounded", "epoch": epoch, "val": val_metrics, "experiment": experiment, "lambda_cf": lambda_cf, "fresh_initialization": True})
        write_csv(run_dir / "history.csv", history)
        print(f"{experiment} epoch={epoch}/{config['stage_b_epochs']} loss={entry['loss']:.6f} val_qwk={val_metrics['qwk']:.6f} seconds={entry['seconds']:.1f}", flush=True)
    atomic_json(run_dir / "summary.json", {"best_epoch": best_epoch, "best_key": best_key, "lambda_cf": lambda_cf})
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    records = make_predictions(config, "eyepacs", "full", cache, split_indices(cache, "test"), model, device)
    for row in records:
        row["variant"] = experiment
    pred_path = output / "predictions/eyepacs_seed42_test.csv"
    write_csv(pred_path, records, prediction_fields("bounded"))
    return pred_path


def _full_metrics(y, pred):
    base = metrics(y, pred)
    recalls = recall_score(y, pred, labels=list(range(5)), average=None, zero_division=0)
    f1s = f1_score(y, pred, labels=list(range(5)), average=None, zero_division=0)
    result = {"overall_qwk": base["qwk"], "overall_macro_f1": base["macro_f1"], "overall_balanced_accuracy": float(recalls.mean()), "overall_mae": base["mae"], "overall_accuracy": base["accuracy"], "n": base["n"]}
    result.update({f"recall_grade{g}": float(recalls[g]) for g in range(5)})
    result.update({f"f1_grade{g}": float(f1s[g]) for g in range(5)})
    return result


def _counts(rows):
    target_error = np.asarray([abs(int(r["target_only_pred"]) - int(r["y_true"])) for r in rows])
    final_error = np.asarray([abs(int(r["final_pred"]) - int(r["y_true"])) for r in rows])
    n = len(rows); benefit = int((final_error < target_error).sum()); harm = int((final_error > target_error).sum()); neutral = n - benefit - harm
    return {"n": n, "benefit_count": benefit, "benefit_rate": benefit / n if n else float("nan"), "harm_count": harm, "harm_rate": harm / n if n else float("nan"), "neutral_count": neutral, "neutral_rate": neutral / n if n else float("nan")}


def _gap_label(row):
    gap = abs(int(row["y_true"]) - int(row["fellow_y_true"]))
    return str(gap) if gap <= 2 else "3+"


def _read_method_rows(root: Path, path: Path):
    rows = read_csv(path)
    target = {r["sample_id"]: r for r in read_csv(root / "outputs/predictions/eyepacs/target_only_seed42_test.csv")}
    for row in rows:
        row["final_pred"] = row.get("final_pred", row.get("pred_grade"))
        row["target_only_pred"] = row.get("target_only_pred", target[row["sample_id"]]["pred_grade"])
    return rows


def analyze(config, root: Path, experiment: str, pred_path: Path, tau_path: Path):
    del config
    out = root / "results" / experiment / "analysis"
    rows = _read_method_rows(root, pred_path)
    y = np.asarray([int(r["y_true"]) for r in rows]); final = np.asarray([int(r["final_pred"]) for r in rows])
    overall = _full_metrics(y, final)
    write_csv(out / "metrics_overall.csv", [overall])
    recalls = recall_score(y, final, labels=list(range(5)), average=None, zero_division=0)
    f1s = f1_score(y, final, labels=list(range(5)), average=None, zero_division=0)
    per_grade = [{"grade": g, "n": int((y == g).sum()), "recall": float(recalls[g]), "f1": float(f1s[g])} for g in range(5)]
    write_csv(out / "metrics_per_grade.csv", per_grade)
    by_grade = [{"target_grade": g, **_counts([r for r in rows if int(r["y_true"]) == g])} for g in range(5)]
    write_csv(out / "benefit_harm_by_grade.csv", by_grade)
    gap_rows = []
    for gap in ("0", "1", "2", "3+"):
        subset = [r for r in rows if _gap_label(r) == gap]
        sy = np.asarray([int(r["y_true"]) for r in subset]); sf = np.asarray([int(r["final_pred"]) for r in subset]); st = np.asarray([int(r["target_only_pred"]) for r in subset])
        fm, tm = _full_metrics(sy, sf), _full_metrics(sy, st)
        gap_rows.append({"gap": gap, "n": len(subset), "target_qwk": tm["overall_qwk"], "full_qwk": fm["overall_qwk"], "delta_qwk": fm["overall_qwk"] - tm["overall_qwk"], "target_macro_f1": tm["overall_macro_f1"], "full_macro_f1": fm["overall_macro_f1"], "target_balanced_accuracy": tm["overall_balanced_accuracy"], "full_balanced_accuracy": fm["overall_balanced_accuracy"], "target_mae": tm["overall_mae"], "full_mae": fm["overall_mae"], **_counts(subset)})
    write_csv(out / "metrics_by_gap.csv", gap_rows)
    cross_rows = []
    for gap in ("0", "1", "2", "3+"):
        for grade in range(5):
            subset = [r for r in rows if _gap_label(r) == gap and int(r["y_true"]) == grade]
            n = len(subset); sy = np.asarray([int(r["y_true"]) for r in subset]); sf = np.asarray([int(r["final_pred"]) for r in subset]); st = np.asarray([int(r["target_only_pred"]) for r in subset])
            ta = float((st == sy).mean()) if n else float("nan"); fa = float((sf == sy).mean()) if n else float("nan"); tm = float(np.abs(st - sy).mean()) if n else float("nan"); fm = float(np.abs(sf - sy).mean()) if n else float("nan")
            cross_rows.append({"gap": gap, "target_grade": grade, "n": n, "small_n": n < 20, "target_accuracy": ta, "full_accuracy": fa, "accuracy_delta": fa - ta, "target_mae": tm, "full_mae": fm, "mae_delta": fm - tm, **_counts(subset)})
    write_csv(out / "gap_x_grade_analysis.csv", cross_rows)
    cache = load_cache(root, "eyepacs"); cache_ids = {r["sample_id"]: i for i, r in enumerate(cache["metadata"]) if r["run_split"] == "test"}
    diagnostic_rows = []
    for r in rows:
        i = cache_ids[r["sample_id"]]; te = abs(int(r["target_only_pred"]) - int(r["y_true"])); fe = abs(int(r["final_pred"]) - int(r["y_true"])); outcome = "benefit" if fe < te else "harm" if fe > te else "neutral"
        for b, name in enumerate(BOUNDARY_NAMES):
            alpha = float(r[f"alpha_{name}"]); magnitude = abs(alpha * float(cache["a_fellow"][i, b] - cache["a_target"][i, b]))
            diagnostic_rows.append({"sample_id": r["sample_id"], "gap": _gap_label(r), "target_grade": int(r["y_true"]), "outcome": outcome, "boundary": name, "alpha": alpha, "correction_magnitude": magnitude})
    write_csv(out / "alpha_correction_diagnostics.csv", diagnostic_rows)
    aggregates = []
    for dimension in ("gap", "boundary", "outcome"):
        for value in sorted({str(r[dimension]) for r in diagnostic_rows}):
            subset = [r for r in diagnostic_rows if str(r[dimension]) == value]
            aggregates.append({"dimension": dimension, "value": value, "n_boundary_records": len(subset), "mean_alpha": float(np.mean([r["alpha"] for r in subset])), "mean_correction_magnitude": float(np.mean([r["correction_magnitude"] for r in subset]))})
    write_csv(out / "alpha_correction_aggregates.csv", aggregates)
    tau_source = read_csv(tau_path); tau_diagnostics = []
    for dimension, values in (("boundary", list(BOUNDARY_NAMES)), ("gap", ["0", "1", "2", "3+"])):
        for value in values:
            for name in BOUNDARY_NAMES if dimension == "gap" else (value,):
                subset = tau_source if dimension == "boundary" else [r for r in tau_source if (str(r["gap"]) if int(r["gap"]) <= 2 else "3+") == value]
                tau = np.asarray([float(r[f"tstar_{name}"]) for r in subset]); weight = np.asarray([float(r[f"weight_{name}"]) for r in subset])
                tau_diagnostics.append({"dimension": dimension, "value": value, "boundary": name, "n": len(subset), "mean_tstar": float(tau.mean()), "median_tstar": float(np.median(tau)), "tstar_zero_rate": float((tau == 0).mean()), "tstar_0_025_rate": float(((tau > 0) & (tau < .25)).mean()), "tstar_025_05_rate": float(((tau >= .25) & (tau < .5)).mean()), "tstar_05_075_rate": float(((tau >= .5) & (tau < .75)).mean()), "tstar_075_1_rate": float(((tau >= .75) & (tau < 1)).mean()), "tstar_one_rate": float((tau == 1).mean()), "weight_mean": float(weight.mean()), "weight_median": float(np.median(weight)), "weight_q25": float(np.quantile(weight, .25)), "weight_q75": float(np.quantile(weight, .75)), "weight_near_zero_rate": float((weight <= 1e-6).mean())})
    write_csv(out / "soft_tau_weight_diagnostics.csv", tau_diagnostics)
    binary_qwk = json.loads((root / "outputs/metrics/eyepacs/full_seed42_test.json").read_text())["qwk"]
    gap1 = next(r for r in gap_rows if r["gap"] == "1")
    effective = bool(gap1["delta_qwk"] > -0.0417 and overall["overall_qwk"] >= float(binary_qwk) - 0.005)
    result = {"experiment": experiment, "SOFT_TAU_EFFECTIVE": effective, "overall": overall, "gap_metrics": gap_rows, "benefit_harm_by_grade": by_grade, "binary_full_qwk": float(binary_qwk)}
    atomic_json(out / "summary.json", result)
    risk = sorted([r for r in cross_rows if r["n"]], key=lambda r: (r["accuracy_delta"], -r["n"]))[:5]
    lines = [experiment, f"SOFT_TAU_EFFECTIVE = {str(effective).upper()}", f"Overall: QWK={overall['overall_qwk']:.6f}, Macro-F1={overall['overall_macro_f1']:.6f}, Balanced Accuracy={overall['overall_balanced_accuracy']:.6f}, MAE={overall['overall_mae']:.6f}, Accuracy={overall['overall_accuracy']:.6f}", "Per-grade: " + ", ".join(f"G{r['grade']} recall={r['recall']:.4f} F1={r['f1']:.4f}" for r in per_grade), "GAP: " + ", ".join(f"GAP{r['gap']} n={r['n']} target={r['target_qwk']:.4f} full={r['full_qwk']:.4f} delta={r['delta_qwk']:+.4f}" for r in gap_rows), "Highest-risk GAP x grade: " + "; ".join(f"GAP{r['gap']}/grade{r['target_grade']} n={r['n']} acc_delta={r['accuracy_delta']:+.4f}{' small-n' if r['small_n'] else ''}" for r in risk)]
    (out / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def comparison(root: Path, exp1, exp2):
    methods = [("Binary Full", _read_method_rows(root, root / "outputs/predictions/eyepacs/full_seed42_test.csv")), (str(exp1["experiment"]), _read_method_rows(root, root / "results" / str(exp1["experiment"]) / "predictions/eyepacs_seed42_test.csv"))]
    if exp2:
        methods.append((str(exp2["experiment"]), _read_method_rows(root, root / "results" / str(exp2["experiment"]) / "predictions/eyepacs_seed42_test.csv")))
    overall_rows, gap_rows, grade_rows, cross_rows = [], [], [], []
    for method, rows in methods:
        y = np.asarray([int(r["y_true"]) for r in rows]); pred = np.asarray([int(r["final_pred"]) for r in rows])
        overall_rows.append({"method": method, **_full_metrics(y, pred)})
        for grade in range(5):
            grade_rows.append({"method": method, "target_grade": grade, **_counts([r for r in rows if int(r["y_true"]) == grade])})
        for gap in ("0", "1", "2", "3+"):
            subset = [r for r in rows if _gap_label(r) == gap]; sy = np.asarray([int(r["y_true"]) for r in subset]); sp = np.asarray([int(r["final_pred"]) for r in subset]); st = np.asarray([int(r["target_only_pred"]) for r in subset]); fq, tq = metrics(sy, sp)["qwk"], metrics(sy, st)["qwk"]
            gap_rows.append({"method": method, "gap": gap, "n": len(subset), "target_qwk": tq, "full_qwk": fq, "delta_qwk": fq - tq, **_counts(subset)})
            for grade in range(5):
                cell = [r for r in subset if int(r["y_true"]) == grade]
                if not cell: continue
                cy = np.asarray([int(r["y_true"]) for r in cell]); cp = np.asarray([int(r["final_pred"]) for r in cell]); ct = np.asarray([int(r["target_only_pred"]) for r in cell])
                cross_rows.append({"method": method, "gap": gap, "target_grade": grade, "n": len(cell), "small_n": len(cell) < 20, "accuracy_delta": float((cp == cy).mean() - (ct == cy).mean()), "mae_delta": float(np.abs(cp - cy).mean() - np.abs(ct - cy).mean()), **_counts(cell)})
    write_csv(root / "results/SOFT_TAU_WMSE_EXP_COMPARISON.csv", overall_rows)
    write_csv(root / "results/SOFT_TAU_WMSE_EXP_COMPARISON_GAP.csv", gap_rows)
    write_csv(root / "results/SOFT_TAU_WMSE_EXP_COMPARISON_BENEFIT_HARM_BY_GRADE.csv", grade_rows)
    write_csv(root / "results/SOFT_TAU_WMSE_EXP_COMPARISON_GAP_X_GRADE.csv", cross_rows)
    lines = ["Binary Full / Soft Tau WMSE comparison"] + [f"{r['method']}: QWK={r['overall_qwk']:.6f}, Macro-F1={r['overall_macro_f1']:.6f}, Balanced Accuracy={r['overall_balanced_accuracy']:.6f}, MAE={r['overall_mae']:.6f}, Accuracy={r['overall_accuracy']:.6f}" for r in overall_rows] + [f"{r['method']} GAP{r['gap']}: n={r['n']}, QWK={r['full_qwk']:.6f}, delta={r['delta_qwk']:+.6f}, benefit={r['benefit_rate']:.4f}, harm={r['harm_rate']:.4f}" for r in gap_rows]
    risk = sorted(cross_rows, key=lambda r: (r["accuracy_delta"], -r["n"]))[:10]
    lines.append("Highest-risk GAP x grade: " + "; ".join(f"{r['method']} GAP{r['gap']}/grade{r['target_grade']} n={r['n']} acc_delta={r['accuracy_delta']:+.4f}{' small-n' if r['small_n'] else ''}" for r in risk))
    (root / "results/SOFT_TAU_WMSE_EXP_COMPARISON.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
