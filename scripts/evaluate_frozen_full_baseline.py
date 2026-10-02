"""One-shot official-test evaluation, gated by the global freeze marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from drpair.full_baselines.runtime import (
    atomic_json,
    build_model,
    image_root,
    method_transform,
    rows_for_run,
    seed_everything,
    sha256,
)
from train_full_baseline import TEST_BATCH_SIZE, dependency_checkpoint, evaluate, stats


METHODS = ("huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn", "d_odr")
HIGH_RES = {"huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=("eyepacs", "messidor2"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    repo = args.repo.resolve()
    marker_path = repo / "outputs/status/MODEL_FREEZE_COMPLETE.json"
    if not marker_path.is_file():
        raise RuntimeError("official test is locked: MODEL_FREEZE_COMPLETE.json is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    identity = (args.dataset, args.method, args.seed)
    locked = {
        (item["dataset"], item["method"], int(item["seed"])): item
        for item in marker.get("locked_runs", [])
    }
    if marker.get("status") != "frozen" or identity not in locked:
        raise RuntimeError(f"run is not present in the global freeze marker: {identity}")

    run_dir = repo / "outputs/checkpoints" / args.dataset / args.method / f"seed{args.seed}"
    checkpoint_path = run_dir / "best.pt"
    done_path = repo / "outputs/status" / f"{args.dataset}_{args.method}_seed{args.seed}.done"
    if done_path.exists():
        raise RuntimeError(
            f"official test already evaluated for this frozen run; refusing a second access: {done_path}"
        )
    if sha256(checkpoint_path) != locked[identity]["checkpoint_sha256"]:
        raise RuntimeError(f"frozen checkpoint changed after lock: {checkpoint_path}")

    # Only now, after the marker and hash check, is the manifest loaded with
    # official test rows and labels.
    rows = rows_for_run(repo, args.dataset, include_test=True)
    test_rows = [row for row in rows if row["run_split"] == "test"]
    if not test_rows:
        raise RuntimeError("official test split is empty")
    if args.method in HIGH_RES:
        root = image_root(repo, args.dataset, args.method)
        missing = [row["target_image_path"] for row in test_rows if not (root / row["target_image_path"]).is_file()]
        if missing:
            raise RuntimeError(
                f"post-freeze test cache is incomplete ({len(missing)} missing); "
                "build it only after MODEL_FREEZE_COMPLETE: " + str(missing[:5])
            )

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    parent = dependency_checkpoint(repo, args.dataset, args.method, args.seed, smoke=False)
    model = build_model(
        repo,
        args.method,
        pretrained=False,
        parent_checkpoint=parent,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    mean, std = stats(repo, args.dataset, args.method)
    metrics = evaluate(
        repo,
        args.dataset,
        args.method,
        args.seed,
        model,
        rows,
        "test",
        method_transform(args.method, False, mean, std),
        device,
        TEST_BATCH_SIZE,
    )
    result = {
        "status": "complete",
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "freeze_marker": str(marker_path),
        "freeze_marker_sha256": sha256(marker_path),
        "official_test_access_count": 1,
        "test_metrics": metrics,
    }
    atomic_json(
        done_path,
        result,
    )
    print(f"OFFICIAL TEST COMPLETE {args.dataset}/{args.method}/seed{args.seed}")


if __name__ == "__main__":
    main()
