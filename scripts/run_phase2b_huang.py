"""Run the predeclared matched-backbone Huang paired-feature-fusion comparator.

The comparator follows Huang et al.'s official second-stage implementation: freeze a
trained single-eye encoder, concatenate ordered target/fellow features, and train only
the scalar MSE fusion head with SGD for 20 epochs. The ConvNeXt feature dimensions and
this repository's image preprocessing are controlled adaptations and are reported as
such; this is not labeled an exact reproduction of the original ResNet-50 result.
"""

import argparse
import ctypes
import json
import os
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from drpair.data.images import make_loader
from drpair.data.manifest import read_manifest, validate_manifest, write_csv
from drpair.evaluation.metrics import qwk
from drpair.models import build_expert
from drpair.runtime import (
    experiment_fingerprint,
    file_hash,
    load_config,
    save_json,
    seed_everything,
    setup_device,
)

REFERENCE = {
    "paper": "Huang et al., Diagnostics 2023, 13(10):1664",
    "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/",
    "official_code": "https://github.com/YijinHuang/pytorch-classification",
    "official_commit_inspected": "e6cd6b49a27612166f7e60f68bf511ac5030140b",
}
RUNS = (
    ("configs/phase1.toml", "phase1_convnext_s_seed42_staged_v1"),
    ("configs/phase2_seed123.toml", "phase2_convnext_s_seed123_replication_v1"),
    ("configs/phase2_seed128.toml", "phase2_convnext_s_seed128_replication_v1"),
)


class HuangFusionHead(nn.Module):
    """Dimension-matched form of the official 4096->2048->pool->512->pool->1 head."""

    def __init__(self, feature_dim):
        super().__init__()
        if feature_dim % 8:
            raise ValueError("Huang matched head requires feature_dim divisible by 8")
        self.network = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.MaxPool1d(2, 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.MaxPool1d(2, 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 8, 1, bias=False),
        )

    def forward(self, target, fellow):
        joined = torch.cat((target, fellow), dim=1).unsqueeze(1)
        return self.network(joined).flatten()


class FrozenHuangComparator(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = HuangFusionHead(encoder.num_features)
        self.encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, target, fellow):
        with torch.no_grad():
            target_features = self.encoder(target)
            fellow_features = self.encoder(fellow)
        return self.head(target_features, fellow_features)


def decode_score(score):
    """Official half-integer threshold decoder, equivalent to clipped rounding."""
    return np.clip(np.floor(np.asarray(score) + 0.5), 0, 4).astype(np.int64)


def aligned_checkpoint(cfg, source_run):
    checkpoint = Path(cfg["data"]["results"]) / source_run / "checkpoints" / "final_single.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = experiment_fingerprint(cfg)
    compatible = all(saved.get("fingerprint", {}).get(k) == v for k, v in expected.items())
    if not compatible:
        authorization_path = (
            checkpoint.parent.parent / "provenance" / "analysis_source_upgrade.json"
        )
        authorization = (
            json.loads(authorization_path.read_text(encoding="utf-8"))
            if authorization_path.is_file()
            else {}
        )
        inference_critical_code_unchanged = (
            subprocess.run(
                [
                    "git",
                    "diff",
                    "--quiet",
                    authorization.get("to_git_commit", "missing"),
                    "HEAD",
                    "--",
                    "src/drpair/models",
                    "src/drpair/data/images.py",
                    "src/drpair/training",
                ],
                check=False,
            ).returncode
            == 0
        )
        compatible = (
            authorization.get("schema") == 1
            and authorization.get("from_source_sha256")
            == saved.get("fingerprint", {}).get("source_sha256")
            and authorization.get("config_sha256") == expected.get("config_sha256")
            and authorization.get("manifest_sha256") == expected.get("manifest_sha256")
            and authorization.get("fold_manifest_sha256") == expected.get("fold_manifest_sha256")
            and authorization.get("artifacts", {}).get("single", {}).get("checkpoint_sha256")
            == file_hash(checkpoint)
            and inference_critical_code_unchanged
        )
    if not saved.get("complete") or saved.get("paired") or not compatible:
        raise ValueError(f"Incompatible frozen single checkpoint: {checkpoint}")
    return checkpoint, saved


def forward_score(model, batch, device, dtype):
    target = batch["target"].to(device, non_blocking=True)
    fellow = batch["fellow"].to(device, non_blocking=True)
    if any(
        p.ndim == 4 and p.is_contiguous(memory_format=torch.channels_last)
        for p in model.encoder.parameters()
    ):
        target = target.contiguous(memory_format=torch.channels_last)
        fellow = fellow.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
        return model(target, fellow)


@torch.inference_mode()
def predict(model, loader, device, dtype):
    model.eval()
    scores = np.full(len(loader.dataset), np.nan, dtype=np.float32)
    for batch in loader:
        value = forward_score(model, batch, device, dtype)
        scores[batch["index"].numpy()] = value.float().cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite Huang comparator predictions")
    return scores


def grouped_metrics(rows, scores):
    labels = np.array([r["label"] for r in rows])
    fellow = np.array([r["fellow_label"] for r in rows])
    patients = np.array([r["patient"] for r in rows])
    pred = decode_score(scores)
    gap = np.abs(labels - fellow)
    groups = {
        "overall": np.ones(len(rows), dtype=bool),
        "gap0": gap == 0,
        "gap1": gap == 1,
        "gap_ge1": gap >= 1,
        "gap_ge2": gap >= 2,
    }
    result = []
    for name, mask in groups.items():
        result.append(
            {
                "group": name,
                "n_eyes": int(mask.sum()),
                "n_patients": len(np.unique(patients[mask])),
                "qwk": qwk(labels[mask], pred[mask]),
                "accuracy": float(np.mean(labels[mask] == pred[mask])),
                "mae": float(np.mean(np.abs(labels[mask] - pred[mask]))),
            }
        )
    return result


def fit_one(cfg, source_run, output, epochs=20, batch=64):
    seed = cfg["experiment"]["seed"]
    seed_everything(seed, cfg["runtime"]["deterministic"])
    device, dtype = setup_device(cfg)
    rows = read_manifest(cfg["data"]["manifest"])
    validate_manifest(rows, cfg["data"]["root"], official=True)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    checkpoint, saved = aligned_checkpoint(cfg, source_run)
    single = build_expert(cfg, paired=False)
    single.load_state_dict(saved["model"])
    model = FrozenHuangComparator(single.encoder).to(device)
    del single, saved

    optimizer = torch.optim.SGD(
        model.head.parameters(), lr=0.02, momentum=0.9, nesterov=True, weight_decay=0.0005
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    train_loader = make_loader(train_rows, cfg, paired=True, train=True, seed=seed, batch=batch)
    val_loader = make_loader(val_rows, cfg, paired=True, train=False, seed=seed, batch=batch)
    best_qwk, best_epoch, best_state = -float("inf"), None, None
    records = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss, seen = 0.0, 0
        for step, batch_data in enumerate(train_loader, 1):
            labels = batch_data["label"].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            scores = forward_score(model, batch_data, device, dtype)
            loss = nn.functional.mse_loss(scores.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.head.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
            if step == 1 or step == len(train_loader) or step % 100 == 0:
                print(
                    f"[huang seed {seed}] epoch {epoch}/{epochs} step "
                    f"{step}/{len(train_loader)} loss={total_loss / seen:.5f}",
                    flush=True,
                )
        val_scores = predict(model, val_loader, device, dtype)
        val_qwk = grouped_metrics(val_rows, val_scores)[0]["qwk"]
        record = {
            "epoch": epoch,
            "train_mse": total_loss / seen,
            "validation_qwk": val_qwk,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(f"[huang seed {seed}] validation qwk={val_qwk:.6f}", flush=True)
        if val_qwk > best_qwk:
            best_qwk, best_epoch = val_qwk, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No Huang validation checkpoint selected")

    model.head.load_state_dict(best_state)
    model.eval()
    # Test is touched only after all 20 validation comparisons have selected one checkpoint.
    test_loader = make_loader(test_rows, cfg, paired=True, train=False, seed=seed, batch=batch)
    test_scores = predict(model, test_loader, device, dtype)
    metrics = grouped_metrics(test_rows, test_scores)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head": best_state,
            "best_epoch": best_epoch,
            "validation_qwk": best_qwk,
            "complete": True,
        },
        output / "huang_matched_head.pt",
    )
    np.savez_compressed(
        output / "test_predictions.npz",
        score=test_scores,
        grade=decode_score(test_scores),
        patient=np.array([r["patient"] for r in test_rows]),
        side=np.array([r["side"] for r in test_rows]),
        y=np.array([r["label"] for r in test_rows]),
        fellow_y=np.array([r["fellow_label"] for r in test_rows]),
    )
    write_csv(output / "training_log.csv", records)
    write_csv(output / "test_metrics.csv", metrics)
    save_json(
        output / "provenance.json",
        {
            **REFERENCE,
            "protocol_name": "huang_pff_convnext_s_matched",
            "claim": "matched-backbone adaptation, not exact original ResNet-50 reproduction",
            "seed": seed,
            "source_run": source_run,
            "frozen_single_checkpoint": str(checkpoint),
            "frozen_single_checkpoint_sha256": file_hash(checkpoint),
            "feature_order": ["target", "fellow"],
            "feature_dim": model.encoder.num_features,
            "head": f"{2 * model.encoder.num_features}->{model.encoder.num_features}->pool2->"
            f"{model.encoder.num_features // 4}->pool2->{model.encoder.num_features // 8}->1",
            "optimizer": "SGD(momentum=0.9,nesterov=True,weight_decay=0.0005)",
            "learning_rate": 0.02,
            "schedule": "constant",
            "loss": "MSE scalar grade regression",
            "epochs": epochs,
            "batch": batch,
            "selection": "best official-validation QWK after 20 fixed epochs",
            "decoder": "half-integer thresholds [-0.5,0.5,1.5,2.5,3.5]",
            "test_access": "once after validation checkpoint selection",
            "controlled_adaptations": [
                "ConvNeXt-S frozen encoder instead of ResNet-50",
                "dimension-proportional fusion head",
                "repository ImageNet normalization and paired augmentation",
            ],
        },
    )
    return {"seed": seed, "best_epoch": best_epoch, "validation_qwk": best_qwk, "metrics": metrics}


def aggregate(output, results):
    rows = []
    for result in results:
        for metric in result["metrics"]:
            rows.append({"seed": result["seed"], **metric})
    summary = []
    for group in ("overall", "gap0", "gap1", "gap_ge1", "gap_ge2"):
        subset = [r for r in rows if r["group"] == group]
        summary.append(
            {
                "group": group,
                "qwk_mean": float(np.mean([r["qwk"] for r in subset])),
                "qwk_sample_sd": float(np.std([r["qwk"] for r in subset], ddof=1)),
                "accuracy_mean": float(np.mean([r["accuracy"] for r in subset])),
                "mae_mean": float(np.mean([r["mae"] for r in subset])),
            }
        )
    write_csv(output / "per_seed_metrics.csv", rows)
    write_csv(output / "summary.csv", summary)
    lines = [
        "# PHASE 2B Huang matched-backbone comparator",
        "",
        "This is a controlled ConvNeXt-S adaptation of Huang PFF, not the reported ResNet-50 result.",
        "The single encoder is frozen; only the official-style scalar fusion head is trained.",
        "",
        "| Group | QWK mean +/- sample SD | Accuracy mean | MAE mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['group']} | {row['qwk_mean']:.4f} +/- {row['qwk_sample_sd']:.4f} | "
            f"{row['accuracy_mean']:.4f} | {row['mae_mean']:.4f} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(
        output / "completion.json",
        {"phase": "2B", "status": "complete", "seeds": [42, 123, 128], **REFERENCE},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if args.epochs != 20 or args.batch != 64:
        raise SystemExit("Locked Huang protocol requires --epochs 20 --batch 64")
    output = Path("experimental_results/phase2b_huang_matched_v1").resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    status = {
        "pid": os.getpid(),
        "phase": "2B",
        "status": "running",
        "stage": "huang_matched",
        "seeds": [42, 123, 128],
        "started_utc": datetime.now(UTC).isoformat(),
        "automatic_next_phase": False,
    }
    save_json(output / "execution_status.json", status)
    sleep_prevention = False
    if os.name == "nt":
        sleep_prevention = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        results = []
        for config_path, source_run in RUNS:
            cfg = load_config(config_path)
            status.update(
                current_seed=cfg["experiment"]["seed"], updated_utc=datetime.now(UTC).isoformat()
            )
            save_json(output / "execution_status.json", status)
            results.append(
                fit_one(
                    cfg,
                    source_run,
                    output / f"seed{cfg['experiment']['seed']}",
                    args.epochs,
                    args.batch,
                )
            )
        aggregate(output, results)
        status.update(
            status="complete", stage="finished", updated_utc=datetime.now(UTC).isoformat()
        )
        save_json(output / "execution_status.json", status)
        print(f"PHASE 2B Huang comparator finished: {output / 'REPORT.md'}", flush=True)
    except BaseException as exc:
        status.update(status="failed", error=repr(exc), updated_utc=datetime.now(UTC).isoformat())
        save_json(output / "execution_status.json", status)
        traceback.print_exc()
        raise
    finally:
        lock.unlink(missing_ok=True)
        if os.name == "nt" and sleep_prevention:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    raise SystemExit(main())
