"""Atomically lock every seed-42 comparator before official-test access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from drpair.full_baselines.runtime import atomic_json, sha256


DATASETS = ("eyepacs", "messidor2")
METHODS = ("huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn", "d_odr")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    repo = args.repo.resolve()
    locked: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in METHODS:
            status_path = repo / "outputs/status" / f"{dataset}_{method}_seed{args.seed}.trained"
            if not status_path.is_file():
                raise RuntimeError(f"training freeze candidate missing: {status_path}")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            checkpoint = Path(status["checkpoint"])
            if status.get("status") != "trained_awaiting_global_freeze":
                raise RuntimeError(f"invalid training status: {status_path}")
            if not checkpoint.is_file() or sha256(checkpoint) != status["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
            if any(
                status.get(key) not in (0, False)
                for key in (
                    "official_test_manifest_loaded",
                    "official_test_image_forward",
                    "official_test_metric",
                )
            ):
                raise RuntimeError(f"pre-freeze test-lock violation: {status_path}")
            locked.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": args.seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": status["checkpoint_sha256"],
                }
            )
    marker = repo / "outputs/status/MODEL_FREEZE_COMPLETE.json"
    atomic_json(
        marker,
        {
            "status": "frozen",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "locked_runs": locked,
            "pre_freeze_test_manifest_loaded": 0,
            "pre_freeze_test_image_forward": 0,
            "pre_freeze_test_metric": 0,
            "test_cache_allowed_after_this_marker": True,
            "test_evaluation_allowed_after_this_marker": True,
        },
    )
    print(f"MODEL_FREEZE_COMPLETE={marker}")
    print("PRE_FREEZE_TEST_LABEL_READ=0")
    print("PRE_FREEZE_TEST_IMAGE_FORWARD=0")
    print("PRE_FREEZE_TEST_METRIC=0")


if __name__ == "__main__":
    main()
