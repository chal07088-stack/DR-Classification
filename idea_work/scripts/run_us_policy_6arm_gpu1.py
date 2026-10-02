from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/minjae/DR-Classification-idea")
BASELINE_ROOT = Path("/home/minjae/DR-Classification")
BASE_OUT = ROOT / "results/US_POLICY_6ARM_GPU1_20260930"
CACHE_PATH = ROOT / "outputs/features/eyepacs_seed42_main_cache.pt"
STAGE_A_PATH = ROOT / "outputs/checkpoints/eyepacs/stage_a/seed42/best.pt"
D2_PATH = ROOT / "results/EXP_D2_UTILITY_STRENGTH/checkpoints/best.pt"
SPLIT_PATH = ROOT / "results/D11_CLEAN_FULL/patient_split_manifest.csv"
D14_SOURCE = ROOT / "scripts/run_d14_conditional_expected_utility.py"
B0_SOURCE = ROOT / "scripts/run_entry_representation_feasibility.py"

EXPECTED_CACHE_SHA = "37370e9ee02a7a2ccb6cd47048a203c20ef4f4fcfbbd4ddbdce0b0b089db516e"
EXPECTED_STAGE_A_SHA = "624fed930762496233e6c6998566b6457b9ed652f58c452240caa809b5426116"
EXPECTED_D2_SHA = "df8be42fcbdf1660609408e866757c9538387715e28aa5a8af147342c95553ed"
BOUNDARIES = ("01", "12", "23", "34")
ARMS = ("Target", "USFull", "SoftUtilityGate", "D14-style-US", "B0-style-US", "Hybrid-style-US")
EPS = 1e-12
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def choose_output() -> Path:
    if not BASE_OUT.exists():
        return BASE_OUT
    suffix = 1
    while True:
        candidate = Path(f"{BASE_OUT}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def logits_from_boundaries(boundaries: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((len(boundaries), 1)), np.cumsum(boundaries, axis=1)], axis=1)


def ce_per_row(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    probability = stable_softmax(logits)
    return -np.log(np.clip(probability[np.arange(len(y)), y], EPS, 1.0))


def gap_group(gap: np.ndarray) -> np.ndarray:
    return np.where(gap >= 3, "GAP3+", np.char.add("GAP", gap.astype(str)))


def grading(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "qwk": float(cohen_kappa_score(y, pred, labels=np.arange(5), weights="quadratic")) if len(y) else np.nan,
        "mae": float(np.mean(np.abs(pred - y))) if len(y) else np.nan,
        "macro_f1": float(f1_score(y, pred, labels=np.arange(5), average="macro", zero_division=0)) if len(y) else np.nan,
        "balanced_accuracy": float(recall_score(y, pred, labels=np.arange(5), average="macro", zero_division=0)) if len(y) else np.nan,
        "accuracy": float(np.mean(pred == y)) if len(y) else np.nan,
    }


def transition(y: np.ndarray, target: np.ndarray, pred: np.ndarray) -> dict:
    target_correct = target == y
    method_correct = pred == y
    benefit = (~target_correct) & method_correct
    harm = target_correct & (~method_correct)
    neutral = ~(benefit | harm)
    target_error = np.abs(target - y)
    method_error = np.abs(pred - y)
    reduced = method_error < target_error
    increased = method_error > target_error
    unchanged = ~(reduced | increased)
    return {
        "benefit": int(benefit.sum()),
        "harm": int(harm.sum()),
        "neutral": int(neutral.sum()),
        "benefit_rate": float(benefit.mean()),
        "harm_rate": float(harm.mean()),
        "neutral_rate": float(neutral.mean()),
        "benefit_minus_harm": int(benefit.sum() - harm.sum()),
        "ordinal_reduced": int(reduced.sum()),
        "ordinal_increased": int(increased.sum()),
        "ordinal_unchanged": int(unchanged.sum()),
        "ordinal_error_target_sum": int(target_error.sum()),
        "ordinal_error_method_sum": int(method_error.sum()),
        "ordinal_error_mass_delta": int(target_error.sum() - method_error.sum()),
        "target_correct_broken": int(harm.sum()),
        "target_wrong_corrected": int(benefit.sum()),
    }


def entropy(probability: np.ndarray) -> np.ndarray:
    return -np.sum(probability * np.log(np.clip(probability, EPS, 1.0)), axis=1)


def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return 1.0 - np.sum(x * y, axis=1) / np.clip(denominator, EPS, None)


def compact81(values: dict) -> tuple[np.ndarray, list[str]]:
    pt, pf = values["p_t"], values["p_f"]
    delta, abs_delta = values["delta"], np.abs(values["delta"])
    abs_p = np.abs(pt - pf)
    midpoint = 0.5 * (pt + pf)
    js = 0.5 * np.sum(pt * np.log(np.clip(pt / np.clip(midpoint, EPS, None), EPS, None)), axis=1)
    js += 0.5 * np.sum(pf * np.log(np.clip(pf / np.clip(midpoint, EPS, None), EPS, None)), axis=1)
    symmetric_kl = 0.5 * (
        np.sum(pt * np.log(np.clip(pt / np.clip(pf, EPS, None), EPS, None)), axis=1)
        + np.sum(pf * np.log(np.clip(pf / np.clip(pt, EPS, None), EPS, None)), axis=1)
    )
    mu_t, mu_f = pt @ np.arange(5), pf @ np.arange(5)
    sorted_t, sorted_f = np.sort(pt, axis=1), np.sort(pf, axis=1)
    feature: dict[str, np.ndarray] = {}
    for boundary, name in enumerate(BOUNDARIES):
        feature[f"delta_{name}"] = delta[:, boundary]
        feature[f"abs_delta_{name}"] = abs_delta[:, boundary]
    feature.update(
        mean_abs_delta=abs_delta.mean(1), max_abs_delta=abs_delta.max(1),
        l2_delta=np.linalg.norm(delta, axis=1), sum_abs_delta=abs_delta.sum(1),
    )
    for grade in range(5):
        feature[f"abs_prob_diff_{grade}"] = abs_p[:, grade]
    feature.update(
        prob_l1=abs_p.sum(1), prob_l2=np.linalg.norm(pt - pf, axis=1),
        prob_cosine_distance=cosine_distance(pt, pf), js_divergence=js,
        symmetric_kl=symmetric_kl, entropy_target=entropy(pt), entropy_fellow=entropy(pf),
        abs_entropy_diff=np.abs(entropy(pt) - entropy(pf)), maxp_target=pt.max(1),
        maxp_fellow=pf.max(1), margin_target=sorted_t[:, -1] - sorted_t[:, -2],
        margin_fellow=sorted_f[:, -1] - sorted_f[:, -2], mu_target=mu_t, mu_fellow=mu_f,
        expected_grade_disagreement=np.abs(mu_t - mu_f),
        predicted_grade_gap=np.abs(pt.argmax(1) - pf.argmax(1)),
    )
    et, ef = values["e_t"], values["e_f"]
    feature.update(
        embedding_cosine_distance=cosine_distance(et, ef), embedding_l2=np.linalg.norm(et - ef, axis=1),
        embedding_l1=np.abs(et - ef).sum(1), embedding_target_norm=np.linalg.norm(et, axis=1),
        embedding_fellow_norm=np.linalg.norm(ef, axis=1),
        embedding_norm_difference=np.abs(np.linalg.norm(et, axis=1) - np.linalg.norm(ef, axis=1)),
    )
    u, s = values["u"], values["s"]
    potential_fraction = u * (1.0 - s)
    correction_u = np.abs(u * delta)
    correction_us = np.abs(u * s * delta)
    reduction = correction_u - correction_us
    for boundary, name in enumerate(BOUNDARIES):
        feature[f"u_{name}"] = u[:, boundary]
        feature[f"s_{name}"] = s[:, boundary]
        feature[f"one_minus_s_{name}"] = 1.0 - s[:, boundary]
        feature[f"potential_fraction_{name}"] = potential_fraction[:, boundary]
        feature[f"correction_u_{name}"] = correction_u[:, boundary]
        feature[f"correction_us_{name}"] = correction_us[:, boundary]
        feature[f"potential_reduction_{name}"] = reduction[:, boundary]
    feature.update(
        mean_u=u.mean(1), min_u=u.min(1), max_u=u.max(1), mean_s=s.mean(1), min_s=s.min(1), max_s=s.max(1),
        mean_one_minus_s=(1.0 - s).mean(1), max_one_minus_s=(1.0 - s).max(1),
        mean_potential_fraction=potential_fraction.mean(1), max_potential_fraction=potential_fraction.max(1),
        max_correction_u=correction_u.max(1), mean_potential_reduction=reduction.mean(1),
        max_potential_reduction=reduction.max(1), sum_potential_reduction=reduction.sum(1),
    )
    frame = pd.DataFrame(feature)
    if frame.shape[1] != 81 or not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError(f"compact_plus_us_v1 integrity failure: {frame.shape}")
    return frame.to_numpy(np.float64), list(frame.columns)


def infer_indices(model, cache: dict, indices: list[int], device: torch.device) -> dict:
    from ours.d2_two_head_experiment import forward_model

    tensor = torch.tensor(indices, dtype=torch.long)
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in ("e_t", "e_f", "u", "s")}
    model.eval()
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(tensor), batch_size=1024, shuffle=False, num_workers=0):
            h_target = cache["h_target"][batch].to(device)
            h_fellow = cache["h_fellow"][batch].to(device)
            p_target = cache["p_target"][batch].to(device)
            p_fellow = cache["p_fellow"][batch].to(device)
            e_target = model.adapter(h_target)
            e_fellow = model.adapter(h_fellow)
            utility, amount, _ = model.components(h_target, h_fellow, p_target, p_fellow)
            chunks["e_t"].append(e_target.cpu().numpy())
            chunks["e_f"].append(e_fellow.cpu().numpy())
            chunks["u"].append(utility.cpu().numpy())
            chunks["s"].append(amount.cpu().numpy())
    result = {key: np.concatenate(value) for key, value in chunks.items()}
    for source, target in (("p_target", "p_t"), ("p_fellow", "p_f"), ("a_target", "a_t"), ("a_fellow", "a_f"), ("z_target", "z_t"), ("z_fellow", "z_f")):
        result[target] = cache[source][tensor].numpy()
    result["delta"] = result["a_f"] - result["a_t"]
    result["correction"] = result["u"] * result["s"] * result["delta"]
    metadata = [cache["metadata"][index] for index in indices]
    result["metadata"] = metadata
    result["y"] = np.asarray([int(row["y_true"]) for row in metadata])
    result["fellow_y"] = np.asarray([int(row["fellow_y_true"]) for row in metadata])
    result["patient"] = np.asarray([str(row["patient_id"]) for row in metadata])
    result["sample_id"] = np.asarray([str(row["sample_id"]) for row in metadata])
    result["target_eye"] = np.asarray([str(row["target_eye"]) for row in metadata])
    result["fellow_eye"] = np.asarray([str(row["fellow_eye"]) for row in metadata])
    return result


def final_from_gate(values: dict, gate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = values["a_t"] + gate[:, None] * values["correction"]
    logits = logits_from_boundaries(boundaries)
    probability = stable_softmax(logits)
    return logits, probability, probability.argmax(1)


def fit_soft_gate(x_train: np.ndarray, train: dict, output: Path) -> tuple[StandardScaler, nn.Linear, dict]:
    scaler = StandardScaler().fit(x_train)
    scale = scaler.scale_.copy()
    scale[scale == 0] = 1.0
    scaler.scale_ = scale
    standardized = scaler.transform(x_train)
    device = torch.device("cuda:0")
    x = torch.tensor(standardized, dtype=torch.float64, device=device)
    a_target = torch.tensor(train["a_t"], dtype=torch.float64, device=device)
    correction = torch.tensor(train["correction"], dtype=torch.float64, device=device)
    y = torch.tensor(train["y"], dtype=torch.long, device=device)
    model = nn.Linear(81, 1, dtype=torch.float64).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    initial_gate = float(torch.sigmoid(model(x)).mean().detach())
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=500, max_eval=625,
        tolerance_grad=1e-9, tolerance_change=1e-12, history_size=100,
        line_search_fn="strong_wolfe",
    )
    calls = 0
    history: list[float] = []

    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        gate = torch.sigmoid(model(x)).squeeze(1)
        state = a_target + gate[:, None] * correction
        logits = torch.cat([torch.zeros((len(state), 1), dtype=state.dtype, device=device), torch.cumsum(state, dim=1)], dim=1)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("SoftUtilityGate non-finite loss")
        loss.backward()
        calls += 1
        history.append(float(loss.detach()))
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        gate = torch.sigmoid(model(x)).squeeze(1)
        state = a_target + gate[:, None] * correction
        logits = torch.cat([torch.zeros((len(state), 1), dtype=state.dtype, device=device), torch.cumsum(state, dim=1)], dim=1)
        final_loss = float(F.cross_entropy(logits, y))
    diagnostics = {
        "initial_gate_mean": initial_gate, "final_gate_mean": float(gate.mean()),
        "final_train_ce": final_loss, "closure_calls": calls,
        "parameter_norm": float(torch.sqrt(sum((parameter.detach() ** 2).sum() for parameter in model.parameters()))),
        "finite": bool(np.isfinite(history).all()), "loss_first": history[0], "loss_last_closure": history[-1],
    }
    joblib.dump(scaler, output / "softutilitygate_scaler.joblib")
    torch.save({"model": model.state_dict(), "diagnostics": diagnostics, "feature_dim": 81}, output / "softutilitygate_final.pt")
    return scaler, model.cpu(), diagnostics


def soft_gate_predict(scaler: StandardScaler, model: nn.Linear, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(scaler.transform(x), dtype=torch.float64)
        return torch.sigmoid(model(tensor)).squeeze(1).numpy()


def write_gap_distribution(path: Path, values: dict) -> None:
    gap = np.abs(values["y"] - values["fellow_y"])
    groups = gap_group(gap)
    rows = []
    for group in ("GAP0", "GAP1", "GAP2", "GAP3+"):
        mask = groups == group
        rows.append({"gap_group": group, "patients": len(np.unique(values["patient"][mask])), "rows": int(mask.sum()), "fraction": float(mask.mean())})
    atomic_csv(path, pd.DataFrame(rows))


def main(precheck_only: bool = False) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal 1")
    if torch.cuda.device_count() != 1 or "A6000" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected exactly one visible RTX A6000")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    hashes = {"cache": sha256(CACHE_PATH), "stage_a": sha256(STAGE_A_PATH), "d2": sha256(D2_PATH)}
    expected = {"cache": EXPECTED_CACHE_SHA, "stage_a": EXPECTED_STAGE_A_SHA, "d2": EXPECTED_D2_SHA}
    if hashes != expected:
        raise RuntimeError(f"Lineage hash mismatch: {hashes}")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    manifest = pd.read_csv(SPLIT_PATH, dtype={"patient_id": str})
    expected_manifest = {"STAGE_A_TRAIN": 9834, "CORRECTION_TRAIN": 2459, "VALIDATION": 2635}
    if manifest["split"].value_counts().to_dict() != expected_manifest:
        raise RuntimeError("Patient manifest count mismatch")
    if len(manifest) != 14928 or manifest["patient_id"].nunique() != 14928:
        raise RuntimeError("Audited reduced-clean manifest must contain 14,928 unique non-test patients")
    correction_patients = set(manifest.loc[manifest.split == "CORRECTION_TRAIN", "patient_id"])
    screen_patients = set(manifest.loc[manifest.split == "VALIDATION", "patient_id"])
    if correction_patients & screen_patients:
        raise RuntimeError("Policy/Screen patient overlap")

    metadata = cache["metadata"]
    if len(metadata) != 35126:
        raise RuntimeError(f"D2 main cache must be a 35,126-row source pool, got {len(metadata)}")

    def stable_key(row: dict) -> tuple[str, str, str, str]:
        return (
            str(row["patient_id"]),
            str(row["target_eye"]),
            str(row["fellow_eye"]),
            str(row["sample_id"]),
        )

    key_to_index = {}
    patient_to_keys = defaultdict(list)
    for index, row in enumerate(metadata):
        key = stable_key(row)
        if key in key_to_index:
            raise RuntimeError(f"Duplicate stable directional key in source pool: {key}")
        key_to_index[key] = index
        patient_to_keys[key[0]].append(key)
    if len(key_to_index) != 35126 or len(patient_to_keys) != 17563:
        raise RuntimeError("D2 source-pool key cardinality mismatch")
    malformed = {patient: keys for patient, keys in patient_to_keys.items() if len(keys) != 2}
    if malformed:
        raise RuntimeError(f"Every reduced-clean patient must have exactly two directional rows; bad={len(malformed)}")
    manifest_patients = set(manifest["patient_id"])
    source_patients = set(patient_to_keys)
    if not manifest_patients.issubset(source_patients):
        raise RuntimeError("Audited reduced-clean manifest is not an exact subset of D2 source-pool patients")
    unmanifested_patients = source_patients - manifest_patients
    if len(unmanifested_patients) != 2635:
        raise RuntimeError(f"Expected 2,635 held-out test patients outside the manifest, got {len(unmanifested_patients)}")
    if any(
        metadata[key_to_index[key]]["run_split"] != "test"
        for patient in unmanifested_patients
        for key in patient_to_keys[patient]
    ):
        raise RuntimeError("Patients outside the audited non-test manifest must be reduced-clean test only")

    policy_keys = {key for patient in correction_patients for key in patient_to_keys[patient]}
    screen_keys = {key for patient in screen_patients for key in patient_to_keys[patient]}
    if policy_keys & screen_keys:
        raise RuntimeError("Policy/Screen stable-key overlap")
    all_source_keys = set(key_to_index)
    selected_keys = policy_keys | screen_keys
    if len(selected_keys) != 10188 or len(all_source_keys - selected_keys) != 24938:
        raise RuntimeError("Selected/non-selected source-pool key cardinality mismatch")
    policy_indices = sorted(key_to_index[key] for key in policy_keys)
    screen_indices = sorted(key_to_index[key] for key in screen_keys)
    if len(policy_indices) != 4918 or len(screen_indices) != 5270:
        raise RuntimeError(f"Directional count mismatch {len(policy_indices)}/{len(screen_indices)}")
    if any(cache["metadata"][i]["run_split"] != "train" for i in policy_indices):
        raise RuntimeError("Policy source split mismatch")
    if any(cache["metadata"][i]["run_split"] != "val" for i in screen_indices):
        raise RuntimeError("Screen source split mismatch")
    if {stable_key(metadata[i]) for i in policy_indices} != policy_keys:
        raise RuntimeError("Policy stable-key recovery mismatch")
    if {stable_key(metadata[i]) for i in screen_indices} != screen_keys:
        raise RuntimeError("Screen stable-key recovery mismatch")
    if len({cache["metadata"][i]["sample_id"] for i in policy_indices}) != 4918 or len({cache["metadata"][i]["sample_id"] for i in screen_indices}) != 5270:
        raise RuntimeError("Duplicate directional row")
    if any(row["run_split"] == "test" for row in [cache["metadata"][i] for i in policy_indices + screen_indices]):
        raise RuntimeError("Official/reduced test split accessed")
    output = choose_output()
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    log_path = output / "run.log"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    log(f"OUTPUT={output}")
    log("Precheck PASS: GPU1-only, source hashes, reduced patient split, no test access")
    atomic_text(output / "environment.txt", "\n".join([
        f"started_utc={started}", f"python={sys.version}", f"executable={sys.executable}",
        f"platform={platform.platform()}", f"torch={torch.__version__}", f"sklearn={sklearn.__version__}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}", f"visible_gpu_count={torch.cuda.device_count()}",
        f"visible_gpu_0={torch.cuda.get_device_name(0)}", "physical_gpu=1", "official_test_accessed=0",
    ]) + "\n")
    atomic_text(output / "checkpoint_hashes.txt", "\n".join(f"{key}={value}" for key, value in hashes.items()) + "\n")
    atomic_text(output / "source_lineage_audit.txt", "\n".join([
        f"cache={CACHE_PATH}", f"stage_a={STAGE_A_PATH}", f"d2={D2_PATH}",
        "all_hashes_match=TRUE", "Target/Fellow logits source=D2 main cache", "u/s source=frozen D2 checkpoint",
        "D11/D16/D17/D18 cache used=FALSE", "main baseline repo modified=FALSE",
        "Historical logical paths under baseline repo were absent; identical expected-hash artifacts were found in idea repo.",
    ]) + "\n")
    atomic_text(output / "data_integrity.txt", "\n".join([
        "d2_main_cache_role=SOURCE_POOL_ONLY", "d2_main_cache_rows=35126", "d2_main_cache_patients=17563",
        "policy_patients=2459", "policy_rows=4918", "policy_run_split=train",
        "screen_patients=2635", "screen_rows=5270", "screen_run_split=val",
        "patient_overlap=0", "duplicate_directional_rows=0", "same_patient_split_leakage=0",
        "selection_key=patient_id|target_eye|fellow_eye|sample_id",
        f"policy_stable_key_sha256={hashlib.sha256(chr(10).join('|'.join(key) for key in sorted(policy_keys)).encode()).hexdigest()}",
        f"screen_stable_key_sha256={hashlib.sha256(chr(10).join('|'.join(key) for key in sorted(screen_keys)).encode()).hexdigest()}",
        "non_selected_main_cache_rows_used_for_policy_or_evaluation=0",
        "official_test_accessed=0", "full_official_manifest_used=FALSE",
    ]) + "\n")
    if precheck_only:
        atomic_text(output / "PRECHECK_COMPLETE.txt", "PRECHECK PASS\nNO TRAINING EXECUTED\n")
        print(f"PRECHECK_OUTPUT={output}")
        return

    from ours.d2_two_head_experiment import TwoHeadStageBGate
    config = json.loads((ROOT / "configs/ours_seed42.json").read_text())
    model = TwoHeadStageBGate(int(config["adapter_dim"]), int(config["gate_hidden"])).cuda()
    checkpoint = torch.load(D2_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if checkpoint.get("epoch") != 30 or checkpoint.get("alpha_formula") != "utility * amount":
        raise RuntimeError("D2 checkpoint metadata mismatch")
    log("Frozen D2 loaded on physical GPU1")
    train = infer_indices(model, cache, policy_indices, torch.device("cuda:0"))
    screen = infer_indices(model, cache, screen_indices, torch.device("cuda:0"))
    del model
    torch.cuda.empty_cache()
    x_train, feature_names = compact81(train)
    x_screen, screen_feature_names = compact81(screen)
    if feature_names != screen_feature_names:
        raise RuntimeError("81D feature schema mismatch")
    np.savez_compressed(output / "compact_plus_us_v1.npz", train=x_train.astype(np.float32), screen=x_screen.astype(np.float32), feature_names=np.asarray(feature_names))
    write_gap_distribution(output / "policy_train_gap_distribution.csv", train)
    write_gap_distribution(output / "screen_gap_distribution.csv", screen)
    actual_screen_gap = Counter(gap_group(np.abs(screen["y"] - screen["fellow_y"])))
    if dict(actual_screen_gap) != {"GAP0": 4598, "GAP1": 446, "GAP2": 210, "GAP3+": 16}:
        raise RuntimeError(f"Screen GAP distribution mismatch: {actual_screen_gap}")
    target_probability_error = float(np.max(np.abs(stable_softmax(logits_from_boundaries(screen["a_t"])) - screen["p_t"])))
    if target_probability_error > 1e-5:
        raise RuntimeError(f"Target reconstruction error {target_probability_error}")
    log(f"Cache inference PASS; target reconstruction max abs probability error={target_probability_error:.3g}")

    soft_scaler, soft_model, soft_diagnostics = fit_soft_gate(x_train, train, output)
    g_soft_train = soft_gate_predict(soft_scaler, soft_model, x_train)
    g_soft = soft_gate_predict(soft_scaler, soft_model, x_screen)
    atomic_json(output / "softutilitygate_config.json", {
        "feature_schema": "compact_plus_us_v1", "feature_dim": 81, "model": "Linear(81,1)+sigmoid",
        "loss": "mean final 5-class cross entropy only", "optimizer": "LBFGS strong_wolfe",
        "optimizer_numerics": {"lr": 1.0, "max_iter": 500, "max_eval": 625, "tolerance_grad": 1e-9, "tolerance_change": 1e-12, "history_size": 100},
        "initialization": "zero weight and bias", "screen_selection": False, **soft_diagnostics,
    })
    log(f"SoftUtilityGate fit complete train_ce={soft_diagnostics['final_train_ce']:.6f} mean_g={soft_diagnostics['final_gate_mean']:.6f}")

    _, p_us_train, pred_us_train = final_from_gate(train, np.ones(len(train["y"])))
    _, p_us, pred_us = final_from_gate(screen, np.ones(len(screen["y"])))
    target_logits_train = logits_from_boundaries(train["a_t"])
    target_logits_screen = logits_from_boundaries(screen["a_t"])
    us_logits_train = logits_from_boundaries(train["a_t"] + train["correction"])
    delta_ce_train = ce_per_row(target_logits_train, train["y"]) - ce_per_row(us_logits_train, train["y"])
    d14_train_x = np.concatenate([train["p_t"], p_us_train, p_us_train - train["p_t"]], axis=1)
    d14_screen_x = np.concatenate([screen["p_t"], p_us, p_us - screen["p_t"]], axis=1)
    d14_wrong = train["p_t"].argmax(1) != train["y"]
    d14_scaler = StandardScaler().fit(d14_train_x[d14_wrong])
    d14_scale = d14_scaler.scale_.copy()
    d14_scale[d14_scale == 0] = 1.0
    d14_scaler.scale_ = d14_scale
    d14_model = LinearRegression(fit_intercept=True).fit(d14_scaler.transform(d14_train_x[d14_wrong]), delta_ce_train[d14_wrong])
    d14_hat_train = d14_model.predict(d14_scaler.transform(d14_train_x))
    d14_hat = d14_model.predict(d14_scaler.transform(d14_screen_x))
    g_d14 = (d14_hat > 0).astype(float)
    joblib.dump(d14_scaler, output / "d14_style_us_scaler.joblib")
    joblib.dump(d14_model, output / "d14_style_us_regressor.joblib")
    d14_train_prediction = d14_model.predict(d14_scaler.transform(d14_train_x[d14_wrong]))
    atomic_json(output / "d14_style_us_config.json", {
        "historical_source": str(D14_SOURCE), "feature_schema": "[p_T(5),p_US(5),p_US-p_T(5)]",
        "feature_dim": 15, "conditioning": "Target-wrong policy rows, matching historical D14 conditional utility",
        "target": "CE_Target - CE_USFull", "model": "StandardScaler + LinearRegression(fit_intercept=True)",
        "objective": "ordinary MSE", "decision": "predicted DeltaCE > 0; tie Target", "threshold_tuning": False,
        "fit_rows": int(d14_wrong.sum()), "train_r2": float(d14_model.score(d14_scaler.transform(d14_train_x[d14_wrong]), delta_ce_train[d14_wrong])),
        "train_pearson": float(pearsonr(d14_train_prediction, delta_ce_train[d14_wrong]).statistic),
        "train_spearman": float(spearmanr(d14_train_prediction, delta_ce_train[d14_wrong]).statistic),
    })
    log(f"D14-style-US fit complete conditional_rows={int(d14_wrong.sum())} ON={g_d14.mean():.6f}")

    b0_scaler = StandardScaler().fit(train["p_t"])
    b0_scale = b0_scaler.scale_.copy()
    b0_scale[b0_scale == 0] = 1.0
    b0_scaler.scale_ = b0_scale
    wrong_train = (train["p_t"].argmax(1) != train["y"]).astype(int)
    b0_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=SEED)
    b0_model.fit(b0_scaler.transform(train["p_t"]), wrong_train)
    if int(b0_model.n_iter_.max()) >= 5000:
        raise RuntimeError("B0 logistic failed to converge")
    p_wrong_train = b0_model.predict_proba(b0_scaler.transform(train["p_t"]))[:, 1]
    p_wrong = b0_model.predict_proba(b0_scaler.transform(screen["p_t"]))[:, 1]
    g_b0 = (p_wrong >= 0.5).astype(float)
    g_hybrid = np.logical_or(g_d14 == 1, g_b0 == 1).astype(float)
    joblib.dump(b0_scaler, output / "b0_style_us_scaler.joblib")
    joblib.dump(b0_model, output / "b0_style_us_logistic.joblib")
    atomic_json(output / "b0_style_us_config.json", {
        "historical_source": str(B0_SOURCE), "feature_schema": "Target probability 5D", "feature_dim": 5,
        "label": "argmax(Target)!=GT", "model": "StandardScaler + LogisticRegression(penalty=None,solver=lbfgs,max_iter=5000,random_state=42)",
        "class_weight": None, "sample_weight": None, "decision": "p_wrong >= 0.5",
        "wrong_prevalence": float(wrong_train.mean()), "train_auroc": float(roc_auc_score(wrong_train, p_wrong_train)),
        "train_auprc": float(average_precision_score(wrong_train, p_wrong_train)), "train_brier": float(brier_score_loss(wrong_train, p_wrong_train)),
        "n_iter": int(b0_model.n_iter_.max()), "threshold_tuning": False,
    })
    log(f"B0-style-US fit complete wrong_prevalence={wrong_train.mean():.6f} ON={g_b0.mean():.6f}")

    gates = {
        "Target": np.zeros(len(screen["y"])), "USFull": np.ones(len(screen["y"])),
        "SoftUtilityGate": g_soft, "D14-style-US": g_d14, "B0-style-US": g_b0,
        "Hybrid-style-US": g_hybrid,
    }
    logits: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    for arm, gate in gates.items():
        logits[arm], probabilities[arm], predictions[arm] = final_from_gate(screen, gate)
    if not np.array_equal(predictions["Target"], screen["p_t"].argmax(1)):
        raise RuntimeError("Target prediction mismatch")
    if not np.array_equal(predictions["USFull"], pred_us):
        raise RuntimeError("USFull common correction mismatch")
    target_pred = predictions["Target"]
    overall_rows, transition_rows, gap_rows = [], [], []
    target_metric = grading(screen["y"], target_pred)
    groups = gap_group(np.abs(screen["y"] - screen["fellow_y"]))
    for arm in ARMS:
        metric = {"arm": arm, **grading(screen["y"], predictions[arm])}
        overall_rows.append(metric)
        effect = transition(screen["y"], target_pred, predictions[arm])
        transition_rows.append({"arm": arm, **effect, "delta_qwk": metric["qwk"] - target_metric["qwk"], "delta_mae": metric["mae"] - target_metric["mae"]})
        for group in ("GAP0", "GAP1", "GAP2", "GAP3+"):
            mask = groups == group
            arm_metric = grading(screen["y"][mask], predictions[arm][mask])
            group_target = grading(screen["y"][mask], target_pred[mask])
            gap_rows.append({
                "arm": arm, "gap_group": group, "n_rows": int(mask.sum()), "n_patients": len(np.unique(screen["patient"][mask])),
                **arm_metric, "target_qwk": group_target["qwk"], "delta_qwk_vs_target": arm_metric["qwk"] - group_target["qwk"],
                "target_mae": group_target["mae"], "delta_mae_vs_target": arm_metric["mae"] - group_target["mae"],
                **transition(screen["y"][mask], target_pred[mask], predictions[arm][mask]), "small_n": bool(mask.sum() < 20),
            })
    overall_frame, transition_frame, gap_frame = pd.DataFrame(overall_rows), pd.DataFrame(transition_rows), pd.DataFrame(gap_rows)
    atomic_csv(output / "overall_metrics.csv", overall_frame)
    atomic_csv(output / "vs_target_transitions.csv", transition_frame)
    atomic_csv(output / "metrics_by_gap.csv", gap_frame)

    gate_rows = []
    for arm in ("SoftUtilityGate", "D14-style-US", "B0-style-US", "Hybrid-style-US"):
        for group, mask in [("Overall", np.ones(len(groups), bool))] + [(name, groups == name) for name in ("GAP0", "GAP1", "GAP2", "GAP3+")]:
            values = gates[arm][mask]
            row = {"arm": arm, "gap_group": group, "n": int(mask.sum()), "mean": float(values.mean()), "std": float(values.std()), "median": float(np.median(values))}
            if arm == "SoftUtilityGate":
                row.update(q10=float(np.quantile(values, .10)), q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)), q90=float(np.quantile(values, .90)), on_count=np.nan, off_count=np.nan, on_rate=np.nan)
            else:
                row.update(q10=np.nan, q25=np.nan, q75=np.nan, q90=np.nan, on_count=int(values.sum()), off_count=int(len(values) - values.sum()), on_rate=float(values.mean()))
            gate_rows.append(row)
    atomic_csv(output / "gate_behavior.csv", pd.DataFrame(gate_rows))

    overlap_rows = []
    overlap_groups = {
        "D14_OFF_B0_OFF": (g_d14 == 0) & (g_b0 == 0), "D14_ON_B0_OFF": (g_d14 == 1) & (g_b0 == 0),
        "D14_OFF_B0_ON": (g_d14 == 0) & (g_b0 == 1), "D14_ON_B0_ON": (g_d14 == 1) & (g_b0 == 1),
    }
    for name, mask in overlap_groups.items():
        effect = transition(screen["y"][mask], target_pred[mask], predictions["USFull"][mask])
        overlap_rows.append({"group": name, "n": int(mask.sum()), "rate": float(mask.mean()), **effect})
    atomic_csv(output / "policy_overlap.csv", pd.DataFrame(overlap_rows))

    prediction_frame = pd.DataFrame({
        "sample_key": [f"{p}|{eye}|{sample}" for p, eye, sample in zip(screen["patient"], screen["target_eye"], screen["sample_id"])],
        "patient_id": screen["patient"], "sample_id": screen["sample_id"], "target_eye": screen["target_eye"], "fellow_eye": screen["fellow_eye"],
        "y_target": screen["y"], "y_fellow": screen["fellow_y"], "gap": np.abs(screen["y"] - screen["fellow_y"]), "gap_group": groups,
        "pred_target": predictions["Target"], "pred_usfull": predictions["USFull"], "pred_softgate": predictions["SoftUtilityGate"],
        "pred_d14": predictions["D14-style-US"], "pred_b0": predictions["B0-style-US"], "pred_hybrid": predictions["Hybrid-style-US"],
        "g_target": gates["Target"], "g_usfull": gates["USFull"], "g_soft": g_soft, "g_d14": g_d14, "g_b0": g_b0, "g_hybrid": g_hybrid,
        "p_wrong_b0": p_wrong, "deltaCE_hat_d14": d14_hat,
    })
    for arm, prefix in (("Target", "target"), ("USFull", "usfull"), ("SoftUtilityGate", "soft"), ("D14-style-US", "d14"), ("B0-style-US", "b0"), ("Hybrid-style-US", "hybrid")):
        prediction_frame[f"{prefix}_error"] = np.abs(predictions[arm] - screen["y"])
        for grade in range(5):
            prediction_frame[f"{prefix}_logit_{grade}"] = logits[arm][:, grade]
            prediction_frame[f"{prefix}_prob_{grade}"] = probabilities[arm][:, grade]
    for boundary, name in enumerate(BOUNDARIES):
        prediction_frame[f"u_{name}"] = screen["u"][:, boundary]
        prediction_frame[f"s_{name}"] = screen["s"][:, boundary]
        prediction_frame[f"C_US_{name}"] = screen["correction"][:, boundary]
    atomic_csv(output / "screen_all_arms_predictions.csv", prediction_frame)

    metrics: list[dict] = []
    overall_index = overall_frame.set_index("arm")
    transition_index = transition_frame.set_index("arm")
    gap_index = gap_frame.set_index(["arm", "gap_group"])
    gate_index = pd.DataFrame(gate_rows).set_index(["arm", "gap_group"])
    for label, field, source in (
        ("Overall QWK", "qwk", overall_index), ("MAE", "mae", overall_index), ("Macro-F1", "macro_f1", overall_index),
        ("Balanced Accuracy", "balanced_accuracy", overall_index), ("Accuracy", "accuracy", overall_index),
        ("Benefit", "benefit", transition_index), ("Harm", "harm", transition_index), ("Neutral", "neutral", transition_index),
        ("Ordinal Reduced", "ordinal_reduced", transition_index), ("Ordinal Increased", "ordinal_increased", transition_index),
        ("Ordinal Unchanged", "ordinal_unchanged", transition_index), ("Ordinal Error Mass Delta", "ordinal_error_mass_delta", transition_index),
    ):
        metrics.append({"Metric": label, **{arm: source.loc[arm, field] for arm in ARMS}})
    for group in ("GAP0", "GAP1", "GAP2"):
        for label, field in ((f"{group} QWK", "qwk"), (f"{group} Delta QWK vs Target", "delta_qwk_vs_target"), (f"{group} MAE", "mae"), (f"{group} Benefit", "benefit"), (f"{group} Harm", "harm"), (f"{group} Ordinal Reduced", "ordinal_reduced"), (f"{group} Ordinal Increased", "ordinal_increased")):
            metrics.append({"Metric": label, **{arm: gap_index.loc[(arm, group), field] for arm in ARMS}})
    metrics.append({"Metric": "Mean g / ON rate", **{arm: (gates[arm].mean()) for arm in ARMS}})
    atomic_csv(output / "final_comparison_table.csv", pd.DataFrame(metrics))

    constants = {
        "arms": list(ARMS), "seed": SEED, "common_correction": "u*s*(a_F-a_T)", "sample_global_gate": True,
        "soft_feature": "compact_plus_us_v1 81D", "soft_loss": "final CE only", "soft_threshold": None,
        "d14_feature": "[pT,pUS,pUS-pT] 15D", "d14_condition": "Target-wrong", "d14_threshold": 0,
        "b0_feature": "Target probability 5D", "b0_threshold": 0.5, "hybrid": "D14 OR B0",
        "gap_training_use": False, "screen_selection": False, "official_test_accessed": 0,
        "full_dataset_run": False, "new_variant_auto_run": False,
    }
    atomic_json(output / "constant_audit.txt", constants)
    atomic_json(output / "run_config.json", constants | {"output": str(output), "feature_names": feature_names})

    oi, ti = overall_index, transition_index
    gi = gate_index
    questions = [
        f"1. USFull > Target overall QWK: {bool(oi.loc['USFull','qwk'] > oi.loc['Target','qwk'])} ({oi.loc['USFull','qwk']-oi.loc['Target','qwk']:+.6f}).",
        f"2. USFull GAP1/GAP2 delta QWK: {gap_index.loc[('USFull','GAP1'),'delta_qwk_vs_target']:+.6f}/{gap_index.loc[('USFull','GAP2'),'delta_qwk_vs_target']:+.6f}.",
        f"3. SoftGate preserved delta QWK vs Target: {oi.loc['SoftUtilityGate','qwk']-oi.loc['Target','qwk']:+.6f}; USFull delta={oi.loc['USFull','qwk']-oi.loc['Target','qwk']:+.6f}.",
        f"4. GAP1/2 Soft vs USFull harm: {int(gap_index.loc[('SoftUtilityGate','GAP1'),'harm'])}/{int(gap_index.loc[('SoftUtilityGate','GAP2'),'harm'])} vs {int(gap_index.loc[('USFull','GAP1'),'harm'])}/{int(gap_index.loc[('USFull','GAP2'),'harm'])}.",
        f"5. D14 coverage={gi.loc[('D14-style-US','Overall'),'on_rate']:.6f}; B/H={int(ti.loc['D14-style-US','benefit'])}/{int(ti.loc['D14-style-US','harm'])}.",
        f"6. B0 coverage={gi.loc[('B0-style-US','Overall'),'on_rate']:.6f}; D14 coverage={gi.loc[('D14-style-US','Overall'),'on_rate']:.6f}.",
        f"7. B0 delta QWK={oi.loc['B0-style-US','qwk']-oi.loc['Target','qwk']:+.6f}.",
        f"8. Hybrid-B0 QWK={oi.loc['Hybrid-style-US','qwk']-oi.loc['B0-style-US','qwk']:+.6f}.",
        f"9. Hybrid vs B0 harm={int(ti.loc['Hybrid-style-US','harm'])-int(ti.loc['B0-style-US','harm']):+d}.",
        f"10-11. B/H, QWK, MAE and ordinal mass are reported separately; B/H is not the primary success criterion.",
        f"12. Soft mean g GAP0/GAP1/GAP2={gi.loc[('SoftUtilityGate','GAP0'),'mean']:.6f}/{gi.loc[('SoftUtilityGate','GAP1'),'mean']:.6f}/{gi.loc[('SoftUtilityGate','GAP2'),'mean']:.6f}.",
        "13. All policy behaviors use the exact same frozen US correction; only the sample-level global gate differs.",
    ]
    summary = [
        "US POLICY 6-ARM GPU1 CONTROLLED SCREENING", "", *questions, "",
        "CAUTIONS:", "- GAP3+ has n=16 and is small-n diagnostic only.",
        "- The fixed historical D2 checkpoint was selected on the old validation split, which is the present Screen Holdout; no new policy selection used Screen Holdout.",
        "- D14-style-US ports the historical conditional utility OLS recipe and changes only the candidate action to USFull.",
        "- Benefit/Harm counts only wrong-to-correct and correct-to-wrong transitions; ordinal reduced/increased captures partial error changes.",
        "", "OFFICIAL TEST ACCESSED = 0", "GPU0 ACCESSED FOR COMPUTE = NO", "GPU0 PROCESS MODIFIED = NO",
        "MAIN REPO MODIFIED = NO", "FULL DATASET RUN = NO", "NEW VARIANT AUTO-RUN = NO",
    ]
    atomic_text(output / "analysis_summary.txt", "\n".join(summary) + "\n")
    atomic_text(output / "COMPLETE.txt", "COMPLETE\nOFFICIAL_TEST_ACCESSED=0\n")
    log("All six arms complete; outputs saved; stopping without follow-up")

    def line(arm: str) -> str:
        return f"{arm}: QWK={oi.loc[arm,'qwk']:.6f} MAE={oi.loc[arm,'mae']:.6f} B/H={int(ti.loc[arm,'benefit'])}/{int(ti.loc[arm,'harm'])}"

    print("========================================")
    print("US POLICY 6-ARM GPU1 = COMPLETE")
    print("========================================")
    print("GPU: Physical GPU1; CUDA_VISIBLE_DEVICES=1")
    print("DATA: Policy Train=2459/4918; Screen Holdout=2635/5270; Official Test Accessed=0")
    for arm in ARMS:
        print(line(arm))
    print(f"Soft mean g overall/GAP0/GAP1/GAP2={g_soft.mean():.6f}/{gi.loc[('SoftUtilityGate','GAP0'),'mean']:.6f}/{gi.loc[('SoftUtilityGate','GAP1'),'mean']:.6f}/{gi.loc[('SoftUtilityGate','GAP2'),'mean']:.6f}")
    print(f"D14/B0/Hybrid ON={g_d14.mean():.6f}/{g_b0.mean():.6f}/{g_hybrid.mean():.6f}")
    for group in ("GAP0", "GAP1", "GAP2"):
        print(group + " QWK: " + " ".join(f"{arm}={gap_index.loc[(arm,group),'qwk']:.6f}" for arm in ARMS))
    print("GPU0 ACCESSED FOR COMPUTE = NO")
    print("GPU0 PROCESS MODIFIED = NO")
    print("MAIN REPO MODIFIED = NO")
    print("OFFICIAL TEST ACCESSED = 0")
    print("FULL DATASET RUN = NO")
    print("NEW VARIANT AUTO-RUN = NO")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck-only", action="store_true")
    arguments = parser.parse_args()
    main(precheck_only=arguments.precheck_only)
