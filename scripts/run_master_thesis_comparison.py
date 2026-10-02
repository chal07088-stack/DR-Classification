"""Run the isolated three-paper, three-seed thesis comparison.

All artifacts are written below the explicitly supplied experiment root.  The
base repository is read for code and immutable data only.  Runs stop on the
first failure and are resumable through the per-epoch ``last.pt`` checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


SEEDS = (42, 123, 128)
DATASETS = ("eyepacs", "messidor2")
METHODS = (
    "huang_single",
    "huang_pff",
    "drgcnn_single",
    "drgcnn_bffn",
    "d_odr",
)
ENVIRONMENTS = {
    "huang_single": "huang_env",
    "huang_pff": "huang_env",
    "drgcnn_single": "drgcnn_env",
    "drgcnn_bffn": "drgcnn_env",
    "d_odr": "dodr_env",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def patient_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row["split"], set()).add(row["patient_id"])
    return {split: len(ids) for split, ids in sorted(by_split.items())}


def audit_data(experiment: Path) -> dict[str, object]:
    audit_root = experiment / "outputs/data_audit"
    eye_path = audit_root / "eyepacs_directional_manifest.csv"
    mes_path = audit_root / "messidor2_directional_manifest.csv"
    eye = read_manifest(eye_path)
    mes = read_manifest(mes_path)
    eye_splits = Counter(row["split"] for row in eye)
    mes_splits = Counter(row["split"] for row in mes)
    expected_eye = {"train": 35126, "val": 10906, "test": 42670}
    if len(eye) != 88702 or dict(eye_splits) != expected_eye:
        raise RuntimeError(f"EyePACS FULL audit failed: rows={len(eye)} splits={dict(eye_splits)}")
    expected_eye_patients = {"train": 17563, "val": 5453, "test": 21335}
    actual_eye_patients = patient_counts(eye)
    if actual_eye_patients != expected_eye_patients:
        raise RuntimeError(f"EyePACS patient audit failed: {actual_eye_patients}")
    if len({row["sample_id"] for row in eye}) != len(eye):
        raise RuntimeError("EyePACS sample_id values are not unique")
    if len(mes) != 1740 or dict(mes_splits) != {"train": 1392, "test": 348}:
        raise RuntimeError(f"Messidor paired audit failed: rows={len(mes)} splits={dict(mes_splits)}")
    actual_mes_patients = patient_counts(mes)
    if actual_mes_patients != {"train": 696, "test": 174}:
        raise RuntimeError(f"Messidor patient audit failed: {actual_mes_patients}")
    result = {
        "status": "passed",
        "eyepacs": {
            "rows": len(eye),
            "patients": sum(actual_eye_patients.values()),
            "split_rows": dict(eye_splits),
            "split_patients": actual_eye_patients,
            "manifest": str(eye_path),
            "sha256": sha256(eye_path),
        },
        "messidor2": {
            "rows": len(mes),
            "patients": sum(actual_mes_patients.values()),
            "split_rows": dict(mes_splits),
            "split_patients": actual_mes_patients,
            "manifest": str(mes_path),
            "sha256": sha256(mes_path),
            "note": "870 complete graded bilateral pairs; four other graded images are excluded from the common bilateral benchmark",
        },
    }
    atomic_json(experiment / "data_audit.json", result)
    print("DATA_AUDIT_PASS " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run_and_tee(command: list[str], log: Path, cwd: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("LAUNCH " + " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== START {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} =====\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
        stream.write(f"===== EXIT {code} =====\n")
    if code:
        raise subprocess.CalledProcessError(code, command)


def valid_done(experiment: Path, dataset: str, method: str, seed: int) -> bool:
    marker = experiment / "outputs/status" / f"{dataset}_{method}_seed{seed}.done"
    if not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        checkpoint = Path(value["checkpoint"])
        return (
            value.get("status") == "complete"
            and checkpoint.is_file()
            and value.get("checkpoint_sha256") == sha256(checkpoint)
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def summarize(experiment: Path) -> None:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                marker = experiment / "outputs/status" / f"{dataset}_{method}_seed{seed}.done"
                value = json.loads(marker.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "best_epoch": value["best_epoch"],
                        **value["test_metrics"],
                        "checkpoint_sha256": value["checkpoint_sha256"],
                    }
                )
    aggregate = experiment / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (aggregate / "all_seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metric_names = ("qwk", "macro_f1", "balanced_accuracy", "mae", "accuracy")
    summary_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in METHODS:
            selected = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            record: dict[str, object] = {"dataset": dataset, "method": method, "n_seeds": len(selected)}
            for metric in metric_names:
                values = [float(row[metric]) for row in selected if metric in row]
                if values:
                    mean = sum(values) / len(values)
                    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                    record[f"{metric}_mean"] = mean
                    record[f"{metric}_sample_std"] = variance**0.5
            summary_rows.append(record)
    summary_fields = sorted({key for row in summary_rows for key in row})
    with (aggregate / "three_seed_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    atomic_json(aggregate / "three_seed_summary.json", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_repo.resolve()
    experiment = args.experiment_root.resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    audit = audit_data(experiment)
    protocol = {
        "experiment": "석사졸업논문 비교실험",
        "datasets": list(DATASETS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "execution_order": "all EyePACS seeds, then all Messidor2 seeds",
        "eyepacs_population": audit["eyepacs"],
        "messidor_population": audit["messidor2"],
        "method_notes": {
            "huang_pff": "paper-explicit 20 epochs; same-seed frozen encoder with public-code BatchNorm buffer updates",
            "drgcnn_bffn": "faithful reconstruction; public-code BatchNorm buffer updates enabled",
            "d_odr": "official code; EyePACS and common Messidor comparison are protocol-adapted",
        },
        "test_policy": "fixed best validation checkpoint, test evaluated once per completed run",
        "gpu": 0,
    }
    atomic_json(experiment / "protocol.json", protocol)

    conda = "/opt/miniforge3/bin/conda"
    train_script = base / "scripts/train_full_baseline.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # The per-method conda environments are intentionally isolated and do not
    # all have this repository installed as a package.  Import the exact
    # source tree pinned by this run without mutating any environment.
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(base / "src") + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["DRGCNN_BFFN_UPDATE_BN"] = "1"
    # Apply the requested worker cap to future GPU0 training only. The active
    # CPU preprocessing process is independent and keeps its four workers.
    env["DR_NUM_WORKERS"] = "8"
    env.setdefault("DR_PREFETCH_FACTOR", "4")
    env["DR_EVAL_NUM_WORKERS"] = "8"
    env.setdefault("DR_EVAL_PREFETCH_FACTOR", "2")
    env["DR_BFFN_NUM_WORKERS"] = "8"
    env.setdefault("DR_BFFN_PREFETCH_FACTOR", "2")

    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                label = f"{dataset}/{method}/seed{seed}"
                if valid_done(experiment, dataset, method, seed):
                    print(f"SKIP verified {label}", flush=True)
                    continue
                failed = experiment / "outputs/status" / f"{dataset}_{method}_seed{seed}.failed"
                failed.unlink(missing_ok=True)
                command = [
                    conda,
                    "run",
                    "--no-capture-output",
                    "-n",
                    ENVIRONMENTS[method],
                    "python",
                    "-X",
                    "faulthandler",
                    str(train_script),
                    "--repo",
                    str(experiment),
                    "--dataset",
                    dataset,
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                ]
                try:
                    run_and_tee(
                        command,
                        experiment / "logs" / dataset / f"{method}_seed{seed}.log",
                        experiment,
                        env,
                    )
                except BaseException as exc:
                    atomic_json(
                        failed,
                        {
                            "status": "failed",
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "error": repr(exc),
                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                    )
                    print(f"BLOCKING_FAILURE {label}: {exc!r}", flush=True)
                    raise
                print(f"STAGE_COMPLETE {label}", flush=True)
    summarize(experiment)
    atomic_json(
        experiment / "status/all_complete.json",
        {
            "status": "complete",
            "datasets": list(DATASETS),
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print("MASTER_THESIS_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
