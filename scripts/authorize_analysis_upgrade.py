"""Authorize reuse of frozen experts after a verified analysis-only source update."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from drpair.runtime import (
    experiment_fingerprint,
    file_hash,
    git_commit,
    load_config,
    save_json,
)

ALLOWED_CHANGED_FILES = {
    "experimental_results/design_power/boundary_mde_test.json",
    "experimental_results/design_power/boundary_mde_val.json",
    "scripts/authorize_analysis_upgrade.py",
    "scripts/boundary_mde.py",
    "src/drpair/evaluation/feasibility.py",
    "src/drpair/experiments/experts.py",
    "src/drpair/runtime.py",
    "tests/test_analysis_upgrade.py",
    "tests/test_boundary_mde.py",
    "tests/test_shrinkage_null.py",
}


def git_lines(*args):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, encoding="utf-8"
    )
    return [line for line in result.stdout.splitlines() if line]


def same_except_source(saved, expected):
    return all(saved.get(key) == value for key, value in expected.items() if key != "source_sha256")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1.toml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = Path(cfg["data"]["results"]) / cfg["experiment"]["run_name"]
    record_path = run / "run.json"
    summary_path = run / "diagnostics" / "phase1a" / "summary.json"
    if not record_path.is_file() or not summary_path.is_file():
        raise SystemExit("A completed PHASE 1A run is required")
    if git_lines("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Tracked working tree must be clean before authorization")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    current = experiment_fingerprint(cfg)
    base = record.get("git_commit")
    head = git_commit()
    if not base or not head:
        raise SystemExit("Both source commits must be available")
    subprocess.run(["git", "merge-base", "--is-ancestor", base, head], check=True)
    changed = git_lines("diff", "--name-only", f"{base}..{head}")
    unsafe = sorted(set(changed) - ALLOWED_CHANGED_FILES)
    if unsafe:
        raise SystemExit("Training-relevant or unknown changes detected: " + ", ".join(unsafe))
    if not changed:
        raise SystemExit("No source upgrade needs authorization")
    if not same_except_source(record, current):
        raise SystemExit("Config or data fingerprints changed")
    if not summary.get("complete") or not same_except_source(summary.get("fingerprint", {}), current):
        raise SystemExit("PHASE 1A summary is incomplete or incompatible")

    artifacts = {}
    old_source = record.get("source_sha256")
    for kind in ("single", "paired"):
        checkpoint = run / "checkpoints" / f"final_{kind}.pt"
        prediction = run / "predictions" / f"final_{kind}_val.npz"
        if not checkpoint.is_file() or not prediction.is_file():
            raise SystemExit(f"Missing final {kind} artifact")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fingerprint = saved.get("fingerprint", {})
        if (
            not saved.get("complete")
            or saved.get("paired") != (kind == "paired")
            or fingerprint.get("source_sha256") != old_source
            or not same_except_source(fingerprint, current)
        ):
            raise SystemExit(f"Incompatible final {kind} checkpoint")
        artifacts[kind] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint),
            "validation_prediction": str(prediction),
            "validation_prediction_sha256": file_hash(prediction),
        }
        if summary["prediction_sha256"][kind] != artifacts[kind]["validation_prediction_sha256"]:
            raise SystemExit(f"Final {kind} validation prediction changed after review")

    authorization = {
        "schema": 1,
        "reason": "Reuse frozen experts after analysis-only source changes; weights are unchanged.",
        "authorized_utc": datetime.now(UTC).isoformat(),
        "from_git_commit": base,
        "to_git_commit": head,
        "from_source_sha256": old_source,
        "to_source_sha256": current["source_sha256"],
        "config_sha256": current["config_sha256"],
        "manifest_sha256": current["manifest_sha256"],
        "fold_manifest_sha256": current["fold_manifest_sha256"],
        "changed_files": changed,
        "artifacts": artifacts,
    }
    destination = run / "provenance" / "analysis_source_upgrade.json"
    save_json(destination, authorization)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
