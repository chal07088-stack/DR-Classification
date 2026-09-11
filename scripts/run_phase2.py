"""Run locked PHASE 2A replications sequentially on one GPU, then aggregate seeds."""

import argparse
import ctypes
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from drpair.data.manifest import prepare_fold_manifest, read_fold_assignment, read_manifest
from drpair.evaluation.multiseed import make_multiseed_report
from drpair.evaluation.report import make_report
from drpair.experiments.experts import run_experts
from drpair.experiments.gate_study import run_gate_study
from drpair.runtime import load_config, run_directory, save_json

DEFAULT_CONFIGS = ("configs/phase2_seed123.toml", "configs/phase2_seed128.toml")
REFERENCE_RUN = "phase1_convnext_s_seed42_staged_v1"


def ensure_folds(cfg):
    path = Path(cfg["data"]["fold_manifest"])
    rows = read_manifest(cfg["data"]["manifest"])
    if not path.exists():
        prepare_fold_manifest(
            cfg["data"]["manifest"],
            path,
            cfg["experiment"]["folds"],
            cfg["experiment"]["seed"],
        )
    for split in ("train", "val"):
        subset = [row for row in rows if row["split"] == split]
        read_fold_assignment(
            path,
            subset,
            split,
            cfg["experiment"]["seed"],
            cfg["experiment"]["folds"],
        )


def run_seed(cfg, update):
    ensure_folds(cfg)
    run = run_directory(cfg)
    lock = run / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        update(cfg, "final_experts")
        run_experts(
            cfg,
            ["final_single", "final_paired"],
            evaluation_splits=("val", "test"),
        )
        update(cfg, "validation_crossfit_gates")
        run_gate_study(cfg, include_expert_oof=False)
        update(cfg, "seed_report")
        report = make_report(cfg)
        update(cfg, "seed_complete", "complete", report=str(report))
        return cfg["experiment"]["run_name"]
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    args = parser.parse_args()
    configs = [load_config(path) for path in args.configs]
    if [cfg["experiment"]["seed"] for cfg in configs] != [123, 128]:
        raise ValueError("PHASE 2A must run the locked seed order 123, 128")
    results_root = Path(configs[0]["data"]["results"])
    master = results_root / "phase2_multiseed_v1"
    status = {
        "pid": os.getpid(),
        "phase": 2,
        "stage": "startup",
        "status": "running",
        "seeds": [42, 123, 128],
        "started_utc": datetime.now(UTC).isoformat(),
        "automatic_next_phase": False,
        "gpu_jobs_parallel": False,
        "reason": "One 16 GiB GPU; each expert fit reserves about 8.8 GiB",
    }

    def update(cfg, stage, state="running", **extra):
        status.update(
            current_seed=cfg["experiment"]["seed"],
            stage=stage,
            status=state,
            updated_utc=datetime.now(UTC).isoformat(),
            **extra,
        )
        save_json(master / "execution_status.json", status)
        print(f"PHASE 2 seed {cfg['experiment']['seed']} {stage}: {state}", flush=True)

    if os.name == "nt":
        status["prevent_system_sleep"] = bool(
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        )
    completed = []
    try:
        for cfg in configs:
            completed.append((cfg["experiment"]["seed"], run_seed(cfg, update)))
        report = make_multiseed_report(
            results_root,
            [(42, REFERENCE_RUN), *completed],
        )
        status.update(
            stage="finished",
            status="complete",
            report=str(report),
            updated_utc=datetime.now(UTC).isoformat(),
        )
        save_json(master / "execution_status.json", status)
        print(f"PHASE 2A finished: {report}", flush=True)
    except BaseException as exc:
        status.update(
            status="failed",
            error=repr(exc),
            updated_utc=datetime.now(UTC).isoformat(),
        )
        save_json(master / "execution_status.json", status)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    sys.exit(main())
