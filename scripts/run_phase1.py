"""Default: experts + validation diagnostics, then stop for PHASE 1A review."""

import argparse
import ctypes
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from drpair.experiments.precheck import BoundaryReviewRequired
from drpair.experiments.stages import execute_stage, require_review
from drpair.runtime import load_config, run_directory, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1.toml")
    parser.add_argument("--stage", choices=("1a", "1b"), default="1a")
    parser.add_argument(
        "--reviewed-1a",
        action="store_true",
        help="Explicit confirmation to run OOF and gates after reviewing 1A",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "1b":
        require_review(cfg, args.reviewed_1a)
    run = run_directory(cfg)
    lock = run / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    status = {
        "pid": os.getpid(),
        "phase": 1,
        "requested_stage": args.stage,
        "seed": 42,
        "folds": 5,
        "started_utc": datetime.now(UTC).isoformat(),
        "config": str(Path(args.config).resolve()),
        "automatic_next_phase": False,
        "automatic_next_stage": False,
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
        execute_stage(cfg, args.stage, args.reviewed_1a, update)
    except BoundaryReviewRequired as exc:
        update("expert_boundary_precheck", "needs_review", error=str(exc))
        print(str(exc), flush=True)
        return 2
    except BaseException as exc:
        update(status.get("stage", "startup"), "failed", error=repr(exc))
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
