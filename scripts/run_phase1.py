"""Run the explicitly authorized PHASE 1 once, then stop. No scheduler or notifications."""

import argparse
import ctypes
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from drpair.runtime import load_config, run_directory, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1.toml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = run_directory(cfg)
    lock = run / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    status = {
        "pid": os.getpid(),
        "phase": 1,
        "seed": 42,
        "folds": 5,
        "started_utc": datetime.now(UTC).isoformat(),
        "config": str(Path(args.config).resolve()),
        "automatic_next_phase": False,
    }
    if os.name == "nt":
        # Keep training awake without forcing the user's display to remain on.
        status["prevent_system_sleep"] = bool(
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        )

    def update(stage, state="running", **extra):
        status.update(stage=stage, status=state, updated_utc=datetime.now(UTC).isoformat(), **extra)
        save_json(run / "execution_status.json", status)
        print(f"PHASE 1 {stage}: {state}", flush=True)

    try:
        from drpair.evaluation.report import make_report
        from drpair.experiments.experts import expert_jobs, run_experts
        from drpair.experiments.gate_study import run_gate_study

        update("experts")
        run_experts(cfg, expert_jobs())
        update("teachers_and_gates")
        run_gate_study(cfg)
        update("report")
        report = make_report(cfg)
        update("finished", "complete", report=str(report))
    except BaseException as exc:
        update(status.get("stage", "startup"), "failed", error=repr(exc))
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
