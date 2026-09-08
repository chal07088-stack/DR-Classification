import gc
from pathlib import Path

import numpy as np
import torch

from drpair.data.images import make_loader
from drpair.data.manifest import (
    patient_folds,
    read_fold_assignment,
    read_manifest,
    validate_manifest,
)
from drpair.models import build_expert
from drpair.runtime import run_directory, save_json, seed_everything, setup_device
from drpair.training.engine import fit
from drpair.training.predictions import predict, save_predictions


def expert_jobs(folds=5):
    return ["final_single", "final_paired"] + [
        f"oof{f + 1}_{kind}" for f in range(folds) for kind in ("single", "paired")
    ]


def run_experts(cfg, jobs):
    device, dtype = setup_device(cfg)
    rows = read_manifest(cfg["data"]["manifest"])
    validate_manifest(rows, cfg["data"]["root"], official=True)
    train = [r for r in rows if r["split"] == "train"]
    folds = read_fold_assignment(cfg["data"]["fold_manifest"], train, "train")
    run = run_directory(cfg)
    save_json(
        run / "expert_schedule.json",
        {
            "jobs": expert_jobs(),
            "inner_selection": "patient-disjoint subset of each job's training patients",
            "full_refit": True,
            "test_used_for_selection": False,
            "retained_experts": 12,
            "selection_pilots": 12,
        },
    )
    for job in jobs:
        if job not in expert_jobs():
            raise ValueError(f"Unknown job: {job}")
        paired = job.endswith("paired")
        fold = None if job.startswith("final") else int(job[3]) - 1
        fit_rows = train if fold is None else [r for r, f in zip(train, folds) if f != fold]
        held_rows = None if fold is None else [r for r, f in zip(train, folds) if f == fold]
        seed = cfg["experiment"]["seed"]
        # Model selection stays inside the job's training patients, followed by a full refit.
        inner = patient_folds(fit_rows, max(2, round(1 / cfg["training"]["pilot_fraction"])), seed)
        pilot_train = [r for r, f in zip(fit_rows, inner) if f != 0]
        pilot_val = [r for r, f in zip(fit_rows, inner) if f == 0]
        if held_rows and ({r["patient"] for r in fit_rows} & {r["patient"] for r in held_rows}):
            raise ValueError("OOF leakage")
        save_json(
            run / "provenance" / f"{job}.json",
            {
                "fit_patients": sorted({r["patient"] for r in fit_rows}, key=int),
                "selection_patients": sorted({r["patient"] for r in pilot_val}, key=int),
                "holdout_patients": sorted({r["patient"] for r in held_rows or []}, key=int),
                "seed": seed,
                "outer_fold": fold,
            },
        )
        seed_everything(seed, cfg["runtime"]["deterministic"])
        model = build_expert(cfg, paired)
        train_loader = make_loader(pilot_train, cfg, paired, True, seed)
        val_loader = make_loader(pilot_val, cfg, paired, False, seed)
        model, selected_epochs = fit(
            model,
            train_loader,
            cfg,
            device,
            dtype,
            paired,
            run / "checkpoints" / f"{job}_selection.pt",
            run / "logs" / f"{job}_selection.jsonl",
            validation_loader=val_loader,
        )
        del model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
        seed_everything(seed, cfg["runtime"]["deterministic"])
        model = build_expert(cfg, paired)
        train_loader = make_loader(fit_rows, cfg, paired, True, seed)
        model, _ = fit(
            model,
            train_loader,
            cfg,
            device,
            dtype,
            paired,
            run / "checkpoints" / f"{job}.pt",
            run / "logs" / f"{job}.jsonl",
            epochs=selected_epochs,
        )
        if held_rows:
            loader = make_loader(held_rows, cfg, paired, False, seed)
            q = predict(model, loader, device, dtype, paired)
            save_predictions(
                run / "predictions" / f"{job}.npz",
                held_rows,
                q,
                np.full(len(held_rows), fold),
                seed,
            )
            del loader
        else:
            for split in ("val", "test"):
                subset = [r for r in rows if r["split"] == split]
                loader = make_loader(subset, cfg, paired, False, seed)
                q = predict(model, loader, device, dtype, paired)
                save_predictions(run / "predictions" / f"{job}_{split}.npz", subset, q, seed=seed)
                del loader
        del model, train_loader
        gc.collect()
        torch.cuda.empty_cache()
    aggregate_oof(cfg)


def aggregate_oof(cfg):
    run = run_directory(cfg)
    rows = [r for r in read_manifest(cfg["data"]["manifest"]) if r["split"] == "train"]
    fold_ids = read_fold_assignment(cfg["data"]["fold_manifest"], rows, "train")
    for kind in ("single", "paired"):
        paths = [run / "predictions" / f"oof{f + 1}_{kind}.npz" for f in range(5)]
        if not all(p.exists() for p in paths):
            continue
        q = np.full((len(rows), 4), np.nan, np.float32)
        keys = {(r["patient"], r["side"]): i for i, r in enumerate(rows)}
        seen = set()
        for fold, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as d:
                for p, s, value in zip(d["patient"], d["side"], d["q"]):
                    idx = keys[(str(p), str(s))]
                    if idx in seen or fold_ids[idx] != fold:
                        raise ValueError("Duplicated or wrong OOF fold")
                    q[idx] = value
                    seen.add(idx)
        save_predictions(run / "predictions" / f"oof_{kind}.npz", rows, q, fold_ids)


def status(cfg):
    run = Path(cfg["data"]["results"]) / cfg["experiment"]["run_name"]
    return [
        {"job": job, "checkpoint": (run / "checkpoints" / f"{job}.pt").exists()}
        for job in expert_jobs()
    ]
