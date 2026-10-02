from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


METHOD = "Target-Anchored Soft Utility Gate with Fixed Boundary-wise US Correction"
SHORT_NAME = "TA-SUG"
SEED = 42
BOUNDARIES = ("01", "12", "23", "34")
BASELINE_ROOT = Path(os.environ.get("DR_BASELINE_ROOT", "/home/minjae/DR-Classification"))
IDEA_ROOT = Path(os.environ.get("DR_IDEA_ROOT", "/home/minjae/DR-Classification-idea"))
BASE_OUTPUT = Path(
    os.environ.get(
        "DR_IDEA_OUTPUT",
        str(IDEA_ROOT / "results/FINAL_SOFTUTILITY_FULL_SEED42"),
    )
)
PHYSICAL_GPU = int(os.environ.get("DR_PHYSICAL_GPU", "1"))
EXPECTED_GPU_NAME = os.environ.get("DR_EXPECTED_GPU_NAME", "A6000")
PYTHON = Path(sys.executable)
EXPECTED = {
    "eyepacs": {
        "manifest": BASELINE_ROOT / "outputs/official_split/data_audit/eyepacs_directional_manifest.csv",
        "rows": {"train": 35126, "val": 10906, "test": 42670},
        "patients": {"train": 17563, "val": 5453, "test": 21335},
    },
    "messidor2": {
        "manifest": BASELINE_ROOT / "outputs/data_audit/messidor2_directional_manifest.csv",
        "rows": {"train_source": 1392, "train": 1114, "inner_val": 278, "test": 348},
        "patients": {"train_source": 696, "train": 557, "inner_val": 139, "test": 174},
    },
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def stage_log(dataset_root: Path, name: str):
    path = dataset_root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        tee = Tee(sys.__stdout__, handle)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"[{utc()}] BEGIN {name}", flush=True)
            yield
            print(f"[{utc()}] END {name}", flush=True)


def choose_output() -> Path:
    if not BASE_OUTPUT.exists():
        return BASE_OUTPUT
    index = 1
    while Path(f"{BASE_OUTPUT}_{index:02d}").exists():
        index += 1
    return Path(f"{BASE_OUTPUT}_{index:02d}")


def load_final_module():
    path = IDEA_ROOT / "scripts/run_us_policy_6arm_gpu1.py"
    spec = importlib.util.spec_from_file_location("historical_final_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module, path


def load_rows(dataset: str) -> list[dict[str, str]]:
    rows = read_csv(EXPECTED[dataset]["manifest"])
    if dataset == "eyepacs":
        for row in rows:
            row["run_split"] = row["split"]
    else:
        train_patients = {row["patient_id"] for row in rows if row["split"] == "train"}
        ranked = sorted(
            train_patients,
            key=lambda patient: hashlib.sha256(f"2026:{patient}".encode()).hexdigest(),
        )
        val_ids = set(ranked[: round(len(ranked) * 0.2)])
        for row in rows:
            if row["split"] == "train":
                row["run_split"] = "inner_val" if row["patient_id"] in val_ids else "train"
            else:
                row["run_split"] = row["split"]
    validate_rows(dataset, rows)
    return rows


def validate_rows(dataset: str, rows: list[dict[str, str]]) -> None:
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise AssertionError(f"{dataset}: duplicate sample_id")
    split_patients = {
        split: {row["patient_id"] for row in rows if row["run_split"] == split}
        for split in sorted({row["run_split"] for row in rows})
    }
    splits = list(split_patients)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if split_patients[left] & split_patients[right]:
                raise AssertionError(f"{dataset}: patient leakage {left}/{right}")
    if any(row["target_eye"] == row["fellow_eye"] for row in rows):
        raise AssertionError(f"{dataset}: target/fellow eye collision")
    if any(int(row["y_true"]) not in range(5) or int(row["fellow_y_true"]) not in range(5) for row in rows):
        raise AssertionError(f"{dataset}: invalid grade")
    if any(sum(row["patient_id"] == other["patient_id"] for other in rows) != 2 for row in rows[:0]):
        raise AssertionError("unreachable")
    if dataset == "eyepacs":
        row_counts = Counter(row["run_split"] for row in rows)
        patient_counts = {key: len(value) for key, value in split_patients.items()}
        if dict(row_counts) != EXPECTED[dataset]["rows"] or patient_counts != EXPECTED[dataset]["patients"]:
            raise AssertionError(f"EyePACS count mismatch rows={row_counts} patients={patient_counts}")
    else:
        row_counts = Counter(row["run_split"] for row in rows)
        patient_counts = {key: len(value) for key, value in split_patients.items()}
        wanted_rows = {"train": 1114, "inner_val": 278, "test": 348}
        wanted_patients = {"train": 557, "inner_val": 139, "test": 174}
        if dict(row_counts) != wanted_rows or patient_counts != wanted_patients:
            raise AssertionError(f"Messidor count mismatch rows={row_counts} patients={patient_counts}")
    by_patient = Counter(row["patient_id"] for row in rows)
    if any(count != 2 for count in by_patient.values()):
        raise AssertionError(f"{dataset}: every patient must have exactly two directional rows")


def assert_gpu1() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal {PHYSICAL_GPU}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")
    if EXPECTED_GPU_NAME not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Unexpected visible GPU: {torch.cuda.get_device_name(0)}")


def image_path(dataset: str, row: dict[str, str], side: str) -> Path:
    root = BASELINE_ROOT / "data" / ("eyepacs" if dataset == "eyepacs" else "messidor2/IMAGES")
    return root / row[f"{side}_image_path"]


def audit_image_paths(dataset: str, rows: list[dict[str, str]]) -> None:
    missing = []
    for row in rows:
        for side in ("target", "fellow"):
            path = image_path(dataset, row, side)
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
    if missing:
        raise FileNotFoundError(f"{dataset}: missing images: {missing}")


def install_kernel_watch(run_root: Path):
    stop = threading.Event()
    status = run_root / "KERNEL_ERROR_DETECTED.txt"
    log = run_root / "kernel_watch.log"
    pattern = ("hardware error", "aer:", "i/o error", "nvme", "segfault")

    def worker():
        command = ["journalctl", "-kf", "--since", "now", "-o", "short-iso"]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            with log.open("a", encoding="utf-8", buffering=1) as handle:
                for line in process.stdout:
                    handle.write(line)
                    low = line.lower()
                    dangerous = (
                        "hardware error" in low
                        or "aer:" in low
                        or "i/o error" in low
                        or ("nvme" in low and any(word in low for word in ("error", "timeout", "reset", "abort")))
                        or "segfault" in low
                    )
                    if dangerous:
                        atomic_text(status, f"Detected at {utc()}\n{line}")
                        os.kill(os.getpid(), signal.SIGTERM)
                        break
                    if stop.is_set():
                        break
            process.terminate()
        except Exception as exc:
            atomic_text(run_root / "kernel_watch_unavailable.txt", f"{type(exc).__name__}: {exc}\n")

    thread = threading.Thread(target=worker, name="kernel-watch", daemon=True)
    thread.start()
    return stop


def precheck(run_root: Path, config: dict) -> dict[str, list[dict[str, str]]]:
    assert_gpu1()
    all_rows = {dataset: load_rows(dataset) for dataset in ("eyepacs", "messidor2")}
    for dataset, rows in all_rows.items():
        audit_image_paths(dataset, rows)
    if int(config["seed"]) != 42:
        raise AssertionError("seed must be 42")
    if int(config["num_workers"]) != 8 or int(config["eval_num_workers"]) != 8:
        raise AssertionError("image DataLoader workers must be 8")
    module, policy_source = load_final_module()
    dummy = {
        "p_t": np.full((2, 5), 0.2), "p_f": np.full((2, 5), 0.2),
        "delta": np.zeros((2, 4)), "e_t": np.ones((2, 256)), "e_f": np.ones((2, 256)),
        "u": np.full((2, 4), 0.5), "s": np.full((2, 4), 0.5),
    }
    features, names = module.compact81(dummy)
    if features.shape != (2, 81) or len(names) != 81:
        raise AssertionError("compact_plus_us_v1 audit failed")
    integrity = []
    for dataset, rows in all_rows.items():
        integrity.append(f"dataset={dataset}")
        integrity.append(f"manifest={EXPECTED[dataset]['manifest']}")
        integrity.append(f"manifest_sha256={sha256(EXPECTED[dataset]['manifest'])}")
        integrity.append(f"rows={len(rows)}")
        integrity.append(f"patients={len({r['patient_id'] for r in rows})}")
        integrity.append(f"run_split_rows={dict(Counter(r['run_split'] for r in rows))}")
        integrity.append("patient_overlap=0")
        integrity.append("duplicate_directional_rows=0")
    atomic_text(run_root / "dataset_integrity.txt", "\n".join(integrity) + "\n")
    atomic_text(run_root / "manifest_hash.txt", "\n".join(
        f"{dataset}={sha256(EXPECTED[dataset]['manifest'])}  {EXPECTED[dataset]['manifest']}"
        for dataset in ("eyepacs", "messidor2")
    ) + "\n")
    atomic_text(run_root / "source_lineage.txt", "\n".join([
        f"method={METHOD}", f"short_name={SHORT_NAME}",
        f"historical_policy_source={policy_source}", f"historical_policy_source_sha256={sha256(policy_source)}",
        f"core_source={IDEA_ROOT / 'ours/core.py'}", f"core_source_sha256={sha256(IDEA_ROOT / 'ours/core.py')}",
        f"stage_a_source={IDEA_ROOT / 'ours/stage_a.py'}", f"stage_a_source_sha256={sha256(IDEA_ROOT / 'ours/stage_a.py')}",
        f"d2_source={IDEA_ROOT / 'ours/d2_two_head_experiment.py'}", f"d2_source_sha256={sha256(IDEA_ROOT / 'ours/d2_two_head_experiment.py')}",
        "reduced_task_trained_weights_reused=NO", "main_repo_write=NO",
    ]) + "\n")
    atomic_text(run_root / "environment.txt", "\n".join([
        f"started_utc={utc()}", f"python={sys.version}", f"executable={sys.executable}",
        f"platform={platform.platform()}", f"torch={torch.__version__}", f"sklearn={sklearn.__version__}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}", f"physical_gpu={PHYSICAL_GPU}",
        f"visible_gpu_name={torch.cuda.get_device_name(0)}",
    ]) + "\n")
    atomic_json(run_root / "run_config.json", config)
    atomic_text(run_root / "PRECHECK_COMPLETE.txt", "PRECHECK PASS\nREADY FOR FULL TRAINING\nOFFICIAL TEST ACCESSED=0\n")
    return all_rows


def monkey_stage_a(rows: list[dict[str, str]]):
    from ours import stage_a
    original = stage_a.load_rows
    stage_a.load_rows = lambda config, dataset: [dict(row) for row in rows]
    return stage_a, original


def train_stage_a(config: dict, dataset: str, dataset_root: Path, non_test_rows):
    stage_a, original = monkey_stage_a(non_test_rows)
    try:
        stage_a.train_main(config, dataset, dataset_root)
    finally:
        stage_a.load_rows = original


def train_oof(config: dict, dataset: str, dataset_root: Path, non_test_rows):
    stage_a, original = monkey_stage_a(non_test_rows)
    try:
        mapping = dataset_root / "outputs/tau" / f"{dataset}_seed42_oof_patient_folds.csv"
        if not mapping.is_file():
            stage_a.create_oof_mapping(config, dataset, dataset_root)
        for fold in range(int(config["oof_folds"])):
            final = dataset_root / "outputs/checkpoints" / dataset / "oof" / f"fold{fold}/final.pt"
            prediction = dataset_root / "outputs/tau" / f"{dataset}_seed42_oof_fold{fold}_predictions.csv"
            if not (final.is_file() and prediction.is_file()):
                stage_a.train_oof_fold(config, dataset, fold, dataset_root)
    finally:
        stage_a.load_rows = original


def cache_stage_a(config: dict, dataset: str, dataset_root: Path, rows, destination: Path):
    from ours.core import StageAModel, boundaries
    from ours import stage_a
    checkpoint = dataset_root / "outputs/checkpoints" / dataset / "stage_a/seed42/best.pt"
    model = StageAModel(pretrained=False).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()
    loader = stage_a._loader(config, dataset, rows, False, paired=True)
    values = {key: [] for key in ("h_target", "h_fellow", "z_target", "z_fellow", "p_target", "p_fellow", "a_target", "a_fellow")}
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            combined = torch.cat([batch["target"].cuda(non_blocking=True), batch["fellow"].cuda(non_blocking=True)], dim=0)
            features, logits = model(combined)
            h_target, h_fellow = features.chunk(2)
            z_target, z_fellow = logits.chunk(2)
            batch_values = {
                "h_target": h_target, "h_fellow": h_fellow, "z_target": z_target, "z_fellow": z_fellow,
                "p_target": z_target.softmax(1), "p_fellow": z_fellow.softmax(1),
                "a_target": boundaries(z_target), "a_fellow": boundaries(z_fellow),
            }
            for key, tensor in batch_values.items():
                values[key].append(tensor.float().cpu())
            offset += len(batch["target"])
            if offset % 4096 < len(batch["target"]):
                print(f"CACHE {dataset} {offset}/{len(rows)}", flush=True)
    metadata = []
    for row in rows:
        run_split = "val" if row["run_split"] == "inner_val" else row["run_split"]
        metadata.append({
            key: (run_split if key == "run_split" else row[key])
            for key in ("dataset", "run_split", "sample_id", "patient_id", "target_eye", "fellow_eye", "y_true", "fellow_y_true", "target_image_path", "fellow_image_path")
        })
    payload = {"metadata": metadata, **{key: torch.cat(tensors) for key, tensors in values.items()}}
    if any(len(payload[key]) != len(rows) for key in values):
        raise AssertionError("Feature cache length mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(destination)
    print(f"CACHE COMPLETE {dataset} rows={len(rows)} path={destination}", flush=True)
    return payload


def generate_utility_supervision(config: dict, dataset: str, dataset_root: Path, cache: dict):
    from ours.core import boundaries, reconstruct_logits
    predictions = []
    sources = []
    for fold in range(int(config["oof_folds"])):
        source = dataset_root / "outputs/tau" / f"{dataset}_seed42_oof_fold{fold}_predictions.csv"
        sources.append(source)
        predictions.extend(read_csv(source))
    expected = {row["sample_id"] for row in cache["metadata"] if row["run_split"] == "train"}
    if len(predictions) != len(expected) or {row["sample_id"] for row in predictions} != expected:
        raise AssertionError("Utility OOF/cache mismatch")
    output_rows = []
    for source in predictions:
        zt = torch.tensor([[float(source[f"logit_{grade}"]) for grade in range(5)]], dtype=torch.float64)
        zf = torch.tensor([[float(source[f"fellow_logit_{grade}"]) for grade in range(5)]], dtype=torch.float64)
        at, af = boundaries(zt), boundaries(zf)
        label = torch.tensor([int(source["y_true"])])
        row = {"sample_id": source["sample_id"], "patient_id": source["patient_id"], "y_true": int(source["y_true"]), "fellow_y_true": int(source["fellow_y_true"])}
        for boundary, name in enumerate(BOUNDARIES):
            endpoint = at.clone()
            endpoint[:, boundary] = af[:, boundary]
            loss0 = float(F.cross_entropy(reconstruct_logits(at), label))
            loss1 = float(F.cross_entropy(reconstruct_logits(endpoint), label))
            best, worst = min(loss0, loss1), max(loss0, loss1)
            row.update({
                f"tau_utility_{name}": int(loss1 < loss0),
                f"weight_utility_{name}": min(1.0, max(0.0, (worst - best) / (worst + 1e-8))),
                f"loss_t0_{name}": loss0, f"loss_t1_{name}": loss1,
            })
        output_rows.append(row)
    path = dataset_root / "outputs/tau" / f"{dataset}_seed42_utility_endpoint.csv"
    atomic_csv(path, pd.DataFrame(output_rows))
    atomic_json(path.with_suffix(".provenance.json"), {
        "rows": len(output_rows), "target_gt_only": True, "semantic_threshold": None,
        "tie_break": "L1 < L0 else target-anchored zero", "eps_num": 1e-8,
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in sources],
    })
    source_by_id = {row["sample_id"]: row for row in output_rows}
    tau = torch.zeros((len(cache["metadata"]), 4), dtype=torch.float32)
    weight = torch.zeros_like(tau)
    joined = set()
    for index, metadata in enumerate(cache["metadata"]):
        if metadata["run_split"] != "train":
            continue
        row = source_by_id[metadata["sample_id"]]
        joined.add(metadata["sample_id"])
        for boundary, name in enumerate(BOUNDARIES):
            tau[index, boundary] = float(row[f"tau_utility_{name}"])
            weight[index, boundary] = float(row[f"weight_utility_{name}"])
    if joined != set(source_by_id):
        raise AssertionError("Utility supervision/cache join mismatch")
    return tau, weight, path


def d2_evaluate(model, cache, indices, device):
    from ours.core import metrics
    from ours.d2_two_head_experiment import forward_model
    model.eval()
    predictions, labels = [], []
    tensor = torch.tensor(indices, dtype=torch.long)
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(tensor), batch_size=1024, num_workers=0):
            logits, _, _, _ = forward_model(model, cache, batch, device)
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels.extend(int(cache["metadata"][index]["y_true"]) for index in batch.tolist())
    return metrics(np.asarray(labels), np.asarray(predictions))


def train_d2(config: dict, dataset: str, dataset_root: Path, cache, tau, confidence, source):
    from ours.core import atomic_torch_save, metric_key, restore_rng_state, rng_state, seed_everything
    from ours.d2_two_head_experiment import TwoHeadStageBGate, fallback_loss, forward_model, utility_loss
    from ours.stage_b import random_fellow_mapping, split_indices
    seed_everything(int(config["seed"]))
    output = dataset_root / "d2"
    output.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output / "best.pt", output / "last.pt"
    train_indices = split_indices(cache, "train")
    val_indices = split_indices(cache, "val")
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    device = torch.device("cuda:0")
    model = TwoHeadStageBGate(int(config["adapter_dim"]), int(config["gate_hidden"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["stage_b_lr"]), weight_decay=float(config["stage_b_wd"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["stage_b_epochs"]))
    best_key, best_epoch, history, start_epoch = (-float("inf"),) * 3, 0, [], 1
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        best_key, best_epoch, history, start_epoch = tuple(saved["best_key"]), int(saved["best_epoch"]), list(saved["history"]), int(saved["epoch"]) + 1
        restore_rng_state(saved["rng"])
        print(f"RESUME D2 {dataset} epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, int(config["stage_b_epochs"]) + 1):
        started = time.time(); model.train()
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        loader = DataLoader(TensorDataset(torch.arange(len(train_indices))), batch_size=int(config["stage_b_batch"]), shuffle=True, generator=generator, num_workers=0)
        random_map = random_fellow_mapping(cache, train_indices, int(config["random_fellow_seed"]) + epoch)
        values = {key: [] for key in ("loss", "grade", "utility", "fallback", "alpha", "amount")}
        for (local,) in loader:
            indices = train_tensor[local]
            labels = torch.tensor([int(cache["metadata"][i]["y_true"]) for i in indices.tolist()], dtype=torch.long, device=device)
            logits, alpha, utility, amount = forward_model(model, cache, indices, device)
            grade = F.cross_entropy(logits, labels)
            utility_value = utility_loss(utility, tau[indices].to(device), confidence[indices].to(device))
            fallback = fallback_loss(model, cache, indices, random_map[local], device)
            loss = grade + 0.2 * utility_value + float(config["lambda_fallback"]) * fallback
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("D2 non-finite loss/gradient")
            optimizer.step()
            for key, value in (("loss", loss), ("grade", grade), ("utility", utility_value), ("fallback", fallback), ("alpha", alpha.mean()), ("amount", amount.mean())):
                values[key].append(float(value.detach()))
        scheduler.step()
        val = d2_evaluate(model, cache, val_indices, device)
        key = metric_key(val)
        entry = {"epoch": epoch, "seconds": time.time() - started, "lr": scheduler.get_last_lr()[0], **{f"train_{name}": float(np.mean(series)) for name, series in values.items()}, **{f"val_{name}": value for name, value in val.items()}}
        history.append(entry)
        if key > best_key:
            best_key, best_epoch = key, epoch
            atomic_torch_save(best_path, {"model": model.state_dict(), "epoch": epoch, "val": val, "experiment": "EXP_D2_UTILITY_STRENGTH", "fresh_initialization": True, "relation_dim": 522, "lambda_utility": 0.2, "lambda_fallback": config["lambda_fallback"], "alpha_formula": "utility * amount", "supervision_path": str(source)})
        atomic_torch_save(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_key": best_key, "best_epoch": best_epoch, "history": history, "rng": rng_state()})
        atomic_csv(output / "history.csv", pd.DataFrame(history))
        print(f"D2 {dataset} epoch={epoch}/{config['stage_b_epochs']} loss={entry['train_loss']:.6f} val_qwk={val['qwk']:.6f} seconds={entry['seconds']:.1f}", flush=True)
    atomic_json(output / "summary.json", {"best_epoch": best_epoch, "best_key": list(best_key), "selection_data": "validation_only", "test_used_for_selection": False})
    return best_path


def infer(model, cache, indices, policy):
    return policy.infer_indices(model, cache, indices, torch.device("cuda:0"))


def grading(y, prediction):
    return {
        "n": int(len(y)),
        "qwk": float(cohen_kappa_score(y, prediction, labels=np.arange(5), weights="quadratic")) if len(y) else float("nan"),
        "mae": float(np.mean(np.abs(prediction - y))) if len(y) else float("nan"),
        "macro_f1": float(f1_score(y, prediction, labels=np.arange(5), average="macro", zero_division=0)) if len(y) else float("nan"),
        "balanced_accuracy": float(recall_score(y, prediction, labels=np.arange(5), average="macro", zero_division=0)) if len(y) else float("nan"),
        "accuracy": float(np.mean(prediction == y)) if len(y) else float("nan"),
    }


def transition(y, target, prediction):
    target_correct, method_correct = target == y, prediction == y
    benefit, harm = (~target_correct) & method_correct, target_correct & (~method_correct)
    target_error, method_error = np.abs(target - y), np.abs(prediction - y)
    reduced, increased = method_error < target_error, method_error > target_error
    return {
        "benefit": int(benefit.sum()), "harm": int(harm.sum()), "neutral": int((~(benefit | harm)).sum()),
        "benefit_rate": float(benefit.mean()), "harm_rate": float(harm.mean()), "neutral_rate": float((~(benefit | harm)).mean()),
        "benefit_minus_harm": int(benefit.sum() - harm.sum()),
        "ordinal_reduced": int(reduced.sum()), "ordinal_increased": int(increased.sum()), "ordinal_unchanged": int((~(reduced | increased)).sum()),
        "ordinal_error_mass_delta": int(target_error.sum() - method_error.sum()),
    }


def evaluate_arms(values, gates, policy, output_prefix: Path):
    groups = policy.gap_group(np.abs(values["y"] - values["fellow_y"]))
    predictions, probabilities = {}, {}
    for arm, gate in gates.items():
        _, probabilities[arm], predictions[arm] = policy.final_from_gate(values, gate)
    target = predictions["Target"]
    overall, transitions, by_gap = [], [], []
    target_metric = grading(values["y"], target)
    for arm in ("Target", "USFull", "SoftGate"):
        metric = grading(values["y"], predictions[arm])
        overall.append({"arm": arm, **metric, "delta_qwk_vs_target": metric["qwk"] - target_metric["qwk"], "delta_mae_vs_target": metric["mae"] - target_metric["mae"]})
        transitions.append({"arm": arm, **transition(values["y"], target, predictions[arm])})
        for group in ("GAP0", "GAP1", "GAP2", "GAP3+"):
            mask = groups == group
            metric_group = grading(values["y"][mask], predictions[arm][mask])
            target_group = grading(values["y"][mask], target[mask])
            by_gap.append({"arm": arm, "gap_group": group, "n_rows": int(mask.sum()), "n_patients": len(np.unique(values["patient"][mask])), **metric_group, "delta_qwk_vs_target": metric_group["qwk"] - target_group["qwk"], "delta_mae_vs_target": metric_group["mae"] - target_group["mae"], **transition(values["y"][mask], target[mask], predictions[arm][mask]), "small_n": bool(mask.sum() < 20)})
    atomic_csv(output_prefix.with_name(output_prefix.name + "_overall_metrics.csv"), pd.DataFrame(overall))
    atomic_csv(output_prefix.with_name(output_prefix.name + "_vs_target_transitions.csv"), pd.DataFrame(transitions))
    atomic_csv(output_prefix.with_name(output_prefix.name + "_metrics_by_gap.csv"), pd.DataFrame(by_gap))
    soft = gates["SoftGate"]
    gate_rows = []
    for group, mask in [("Overall", np.ones(len(groups), dtype=bool))] + [(name, groups == name) for name in ("GAP0", "GAP1", "GAP2", "GAP3+")]:
        g = soft[mask]
        gate_rows.append({"gap_group": group, "n": int(mask.sum()), "mean": float(g.mean()), "std": float(g.std()), "median": float(np.median(g)), "q10": float(np.quantile(g, .10)), "q25": float(np.quantile(g, .25)), "q75": float(np.quantile(g, .75)), "q90": float(np.quantile(g, .90))})
    atomic_csv(output_prefix.with_name(output_prefix.name + "_gate_behavior.csv"), pd.DataFrame(gate_rows))
    frame = pd.DataFrame({
        "patient_id": values["patient"], "sample_id": values["sample_id"], "target_eye": values["target_eye"], "fellow_eye": values["fellow_eye"],
        "y_target": values["y"], "y_fellow": values["fellow_y"], "gap": np.abs(values["y"] - values["fellow_y"]), "gap_group": groups,
        "pred_target": target, "pred_usfull": predictions["USFull"], "pred_softgate": predictions["SoftGate"], "g_softgate": soft,
    })
    for grade in range(5):
        frame[f"prob_target_{grade}"] = probabilities["Target"][:, grade]
        frame[f"prob_usfull_{grade}"] = probabilities["USFull"][:, grade]
        frame[f"prob_softgate_{grade}"] = probabilities["SoftGate"][:, grade]
    atomic_csv(output_prefix.with_name(output_prefix.name + "_predictions.csv"), frame)
    return overall, transitions, by_gap, gate_rows


def assert_gate_identities(values: dict, policy, output: Path, tolerance: float = 1e-5) -> dict:
    """Prove the two endpoint identities on frozen validation features."""
    count = len(values["a_t"])
    _, probability_g0, prediction_g0 = policy.final_from_gate(values, np.zeros(count))
    _, probability_g1, prediction_g1 = policy.final_from_gate(values, np.ones(count))
    target_probability = values["p_t"]
    usfull_probability = policy.stable_softmax(
        policy.logits_from_boundaries(values["a_t"] + values["correction"])
    )
    error_g0 = float(np.max(np.abs(probability_g0 - target_probability)))
    error_g1 = float(np.max(np.abs(probability_g1 - usfull_probability)))
    prediction_g0_equal = bool(
        np.array_equal(prediction_g0, target_probability.argmax(axis=1))
    )
    prediction_g1_equal = bool(
        np.array_equal(prediction_g1, usfull_probability.argmax(axis=1))
    )
    result = {
        "status": "PASS"
        if error_g0 < tolerance
        and error_g1 < tolerance
        and prediction_g0_equal
        and prediction_g1_equal
        else "FAIL",
        "source_split": "validation",
        "n_rows": count,
        "tolerance": tolerance,
        "g0_target_max_abs_probability_error": error_g0,
        "g0_target_prediction_equal": prediction_g0_equal,
        "g1_usfull_max_abs_probability_error": error_g1,
        "g1_usfull_prediction_equal": prediction_g1_equal,
    }
    atomic_json(output, result)
    if result["status"] != "PASS":
        raise AssertionError(f"Gate endpoint identity check failed: {result}")
    print(f"GATE IDENTITY ASSERTIONS PASS {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def train_dataset(config: dict, dataset: str, rows, dataset_root: Path, policy):
    if (dataset_root / "COMPLETE.txt").is_file():
        raise FileExistsError(f"Refusing to rerun completed dataset: {dataset_root}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    atomic_json(dataset_root / "run_config.json", {**config, "dataset": dataset, "method": METHOD, "short_name": SHORT_NAME, "stage_workers": {"image": 8, "cache_tensor": 0}, "official_test_accessed": 0})
    non_test = [row for row in rows if row["run_split"] != "test"]
    test_rows = [row for row in rows if row["run_split"] == "test"]
    atomic_text(dataset_root / "dataset_integrity.txt", f"rows_total={len(rows)}\nrows_non_test={len(non_test)}\nrows_test={len(test_rows)}\npatient_overlap=0\n")
    checkpoint = dataset_root / "outputs/checkpoints" / dataset / "stage_a/seed42/best.pt"
    with stage_log(dataset_root, "training_log_stageA.txt"):
        if not checkpoint.is_file():
            train_stage_a(config, dataset, dataset_root, non_test)
    with stage_log(dataset_root, "training_log_stageA_OOF.txt"):
        train_oof(config, dataset, dataset_root, non_test)
    cache_path = dataset_root / "outputs/features" / f"{dataset}_seed42_non_test_cache.pt"
    with stage_log(dataset_root, "training_log_cache.txt"):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path.is_file() else cache_stage_a(config, dataset, dataset_root, non_test, cache_path)
    tau, confidence, supervision = generate_utility_supervision(config, dataset, dataset_root, cache)
    with stage_log(dataset_root, "training_log_D2.txt"):
        d2_checkpoint = train_d2(config, dataset, dataset_root, cache, tau, confidence, supervision)
    from ours.d2_two_head_experiment import TwoHeadStageBGate
    d2 = TwoHeadStageBGate(int(config["adapter_dim"]), int(config["gate_hidden"])).cuda()
    d2_state = torch.load(d2_checkpoint, map_location="cpu", weights_only=False)
    d2.load_state_dict(d2_state["model"]); d2.eval(); d2.requires_grad_(False)
    train_indices = [i for i, row in enumerate(cache["metadata"]) if row["run_split"] == "train"]
    val_indices = [i for i, row in enumerate(cache["metadata"]) if row["run_split"] == "val"]
    train_values, val_values = infer(d2, cache, train_indices, policy), infer(d2, cache, val_indices, policy)
    assert_gate_identities(
        val_values,
        policy,
        dataset_root / "gate_identity_assertions.json",
    )
    x_train, feature_names = policy.compact81(train_values)
    x_val, val_names = policy.compact81(val_values)
    if feature_names != val_names:
        raise RuntimeError("81D feature schema mismatch")
    atomic_json(dataset_root / "feature_schema.json", {"name": "compact_plus_us_v1", "dim": 81, "columns": feature_names, "sha256": hashlib.sha256("\n".join(feature_names).encode()).hexdigest()})
    with stage_log(dataset_root, "training_log_softgate.txt"):
        scaler, soft_model, diagnostics = policy.fit_soft_gate(x_train, train_values, dataset_root)
    g_val = policy.soft_gate_predict(scaler, soft_model, x_val)
    val_results = evaluate_arms(val_values, {"Target": np.zeros(len(x_val)), "USFull": np.ones(len(x_val)), "SoftGate": g_val}, policy, dataset_root / "validation")
    for suffix, final_name in (("overall_metrics.csv", "validation_metrics.csv"),):
        shutil.copy2(dataset_root / f"validation_{suffix}", dataset_root / final_name)
    atomic_json(dataset_root / "softgate_config.json", {"feature_schema": "compact_plus_us_v1", "feature_dim": 81, "model": "Linear(81,1)+sigmoid", "loss": "mean final 5-class cross entropy only", "optimizer": "LBFGS strong_wolfe", "workers": 0, **diagnostics})
    # Freeze is complete here. This marker is written before the first test image is opened.
    atomic_text(dataset_root / "MODEL_FREEZE_COMPLETE.txt", "Stage-A frozen\nD2 frozen\nSoftGate frozen\npreprocessing frozen\nfeature recipe frozen\ndecision rule frozen\nofficial_test_accessed=0\n")
    test_cache_path = dataset_root / "outputs/features" / f"{dataset}_seed42_test_cache.pt"
    atomic_text(dataset_root / "OFFICIAL_TEST_ACCESS.txt", f"first_access_utc={utc()}\naccess_count=1\nreason=one-shot final evaluation after freeze\n")
    with stage_log(dataset_root, "test_evaluation_log.txt"):
        test_cache = cache_stage_a(config, dataset, dataset_root, test_rows, test_cache_path)
        test_values = infer(d2, test_cache, list(range(len(test_rows))), policy)
        x_test, test_names = policy.compact81(test_values)
        if feature_names != test_names:
            raise RuntimeError("Test 81D schema mismatch")
        g_test = policy.soft_gate_predict(scaler, soft_model, x_test)
        test_results = evaluate_arms(test_values, {"Target": np.zeros(len(x_test)), "USFull": np.ones(len(x_test)), "SoftGate": g_test}, policy, dataset_root / "test")
    rename = {
        "test_overall_metrics.csv": "test_overall_metrics.csv", "test_metrics_by_gap.csv": "test_metrics_by_gap.csv",
        "test_vs_target_transitions.csv": "test_vs_target_transitions.csv", "test_gate_behavior.csv": "test_gate_behavior.csv",
        "test_predictions.csv": "test_predictions.csv",
    }
    if not all((dataset_root / name).is_file() for name in rename):
        raise RuntimeError("Missing final test artifact")
    hashes = {
        "stageA": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "D2": {"path": str(d2_checkpoint), "sha256": sha256(d2_checkpoint)},
        "SoftGate": {"path": str(dataset_root / "softutilitygate_final.pt"), "sha256": sha256(dataset_root / "softutilitygate_final.pt")},
        "main_non_test_cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "test_cache": {"path": str(test_cache_path), "sha256": sha256(test_cache_path)},
    }
    atomic_text(dataset_root / "checkpoint_hashes.txt", "\n".join(f"{key}={value['sha256']}  {value['path']}" for key, value in hashes.items()) + "\n")
    overall_frame = pd.DataFrame(test_results[0]).set_index("arm")
    transition_frame = pd.DataFrame(test_results[1]).set_index("arm")
    gate_frame = pd.DataFrame(test_results[3]).set_index("gap_group")
    lines = [
        f"METHOD={METHOD}", f"DATASET={dataset}", "PHYSICAL_GPU=1", "SEED=42",
        f"Target QWK={overall_frame.loc['Target','qwk']:.6f} MAE={overall_frame.loc['Target','mae']:.6f}",
        f"USFull QWK={overall_frame.loc['USFull','qwk']:.6f} DeltaQWK={overall_frame.loc['USFull','delta_qwk_vs_target']:+.6f} MAE={overall_frame.loc['USFull','mae']:.6f} B/H={int(transition_frame.loc['USFull','benefit'])}/{int(transition_frame.loc['USFull','harm'])}",
        f"SoftGate QWK={overall_frame.loc['SoftGate','qwk']:.6f} DeltaQWK={overall_frame.loc['SoftGate','delta_qwk_vs_target']:+.6f} MAE={overall_frame.loc['SoftGate','mae']:.6f} B/H={int(transition_frame.loc['SoftGate','benefit'])}/{int(transition_frame.loc['SoftGate','harm'])}",
        f"SoftGate ordinal reduced/increased={int(transition_frame.loc['SoftGate','ordinal_reduced'])}/{int(transition_frame.loc['SoftGate','ordinal_increased'])}",
        f"SoftGate error mass delta={int(transition_frame.loc['SoftGate','ordinal_error_mass_delta'])}",
        f"mean_g={gate_frame.loc['Overall','mean']:.6f}", f"GAP0_mean_g={gate_frame.loc['GAP0','mean']:.6f}", f"GAP1_mean_g={gate_frame.loc['GAP1','mean']:.6f}", f"GAP2_mean_g={gate_frame.loc['GAP2','mean']:.6f}",
        "official_test_used_only_after_model_freeze=YES", "gpu0_modified=NO", "main_repo_modified=NO", "extra_variant_run=NO", "extra_seed_run=NO",
    ]
    atomic_text(dataset_root / "analysis_summary.txt", "\n".join(lines) + "\n")
    atomic_text(dataset_root / "COMPLETE.txt", f"completed_utc={utc()}\nOFFICIAL TEST ACCESSED=1\n")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--precheck-only", action="store_true")
    args = parser.parse_args()
    if not str(IDEA_ROOT) in sys.path:
        sys.path.insert(0, str(IDEA_ROOT))
    run_root = args.run_root or choose_output()
    if run_root.exists() and not args.run_root:
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True, exist_ok=bool(args.run_root))
    config = json.loads((IDEA_ROOT / "configs/ours_seed42.json").read_text())
    config.update({
        "method": METHOD, "short_name": SHORT_NAME, "seed": 42,
        "num_workers": 8, "eval_num_workers": 8, "prefetch_factor": 2,
        "baseline_root": str(BASELINE_ROOT), "idea_root": str(IDEA_ROOT),
        "datasets": ["eyepacs", "messidor2"], "physical_gpu": PHYSICAL_GPU,
        "official_test_policy": "one-shot only after Stage-A, D2, SoftGate and recipe freeze",
    })
    log_path = run_root / "run.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as handle, redirect_stdout(Tee(sys.__stdout__, handle)), redirect_stderr(Tee(sys.__stderr__, handle)):
        try:
            print(f"[{utc()}] RUN_ROOT={run_root}", flush=True)
            rows = precheck(run_root, config)
            print(f"[{utc()}] PRECHECK PASS", flush=True)
            if args.precheck_only:
                return
            watch_stop = install_kernel_watch(run_root)
            policy, _ = load_final_module()
            summaries = []
            for dataset in ("eyepacs", "messidor2"):
                print(f"[{utc()}] DATASET START {dataset}", flush=True)
                summaries.extend(train_dataset(config, dataset, rows[dataset], run_root / dataset, policy))
                print(f"[{utc()}] DATASET COMPLETE {dataset}", flush=True)
            watch_stop.set()
            atomic_text(run_root / "FINAL_SUMMARY.txt", "\n".join([
                "FINAL SOFT UTILITY GATE — SEED42 COMPLETE", f"METHOD: {METHOD}", "GPU: Physical GPU1", "SEED: 42", *summaries,
                "GPU0 MODIFIED = NO", "MAIN REPO MODIFIED = NO", "EXTRA VARIANT RUN = NO", "EXTRA SEED RUN = NO",
            ]) + "\n")
            atomic_text(run_root / "ALL_COMPLETE.txt", f"completed_utc={utc()}\n")
        except BaseException as exc:
            atomic_text(run_root / "ERROR_STATUS.txt", f"failed_utc={utc()}\ntype={type(exc).__name__}\nmessage={exc}\n\n{traceback.format_exc()}")
            print(f"[{utc()}] FATAL {type(exc).__name__}: {exc}", flush=True)
            raise


if __name__ == "__main__":
    main()
