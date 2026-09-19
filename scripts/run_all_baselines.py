from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

SEEDS = [42, 123, 128]
METHODS = ["huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn", "d_odr", "aor_dr"]
ENV = {
    "huang_single": "huang_env",
    "huang_pff": "huang_env",
    "drgcnn_single": "drgcnn_env",
    "drgcnn_bffn": "drgcnn_env",
    "d_odr": "dodr_env",
    "aor_dr": "aordr_env",
}
CONFIG = {
    "huang_single": "configs/huang/main.json",
    "huang_pff": "configs/huang/main.json",
    "drgcnn_single": "configs/drgcnn/main.json",
    "drgcnn_bffn": "configs/drgcnn/main.json",
    "d_odr": "configs/d_odr/main.json",
    "aor_dr": "configs/aor_dr/main.json",
}
PARENT = {"huang_pff": "huang_single", "drgcnn_bffn": "drgcnn_single"}


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_done(repo: Path, dataset: str, method: str, seed: int) -> bool:
    marker = repo / "outputs/status" / f"{dataset}_{method}_seed{seed}.done"
    try:
        value = json.loads(marker.read_text())
        checkpoint = Path(value["checkpoint"])
        valid = (
            value["status"] == "complete"
            and checkpoint.is_file()
            and checksum(checkpoint) == value["checkpoint_sha256"]
            and value.get("protocol_sha256") == checksum(repo / "configs/common/protocol.json")
            and value.get("method_config_sha256") == checksum(repo / CONFIG[method])
        )
        if valid and method in PARENT:
            parent = (
                repo / "outputs/checkpoints" / dataset / PARENT[method] / f"seed{seed}" / "best.pt"
            )
            valid = parent.is_file() and value.get("dependency_checkpoint_sha256") == checksum(
                parent
            )
        return valid
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_logged(command: list[str], log: Path, cwd: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("LAUNCH", " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== START {time.strftime('%F %T')} {' '.join(command)} =====\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONPATH": str(cwd / "src")},
        )
        assert process.stdout
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def conda_python(env: str) -> list[str]:
    return ["/opt/miniforge3/bin/conda", "run", "--no-capture-output", "-n", env, "python"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    logs = repo / "outputs/logs"
    marker = repo / "data/.transfer_complete"
    while not marker.is_file():
        print("WAITING data transfer marker; training has not started", flush=True)
        time.sleep(60)
    run_logged(
        conda_python("huang_env")
        + [
            str(repo / "scripts/audit_full_data.py"),
            "--repo-root",
            str(repo),
            "--messidor-root",
            str(repo / "data/messidor2/IMAGES"),
        ],
        logs / "00_data_audit.log",
        repo,
    )
    run_logged(
        conda_python("huang_env") + [str(repo / "scripts/record_full_environment.py")],
        logs / "01_environment.log",
        repo,
    )
    for dataset in ("eyepacs", "messidor2"):
        for seed in SEEDS:
            for method in METHODS:
                if valid_done(repo, dataset, method, seed):
                    print(f"SKIP verified {dataset}/{method}/seed{seed}", flush=True)
                    continue
                smoke_marker = repo / "outputs/status" / f"smoke_{dataset}_{method}.done"
                if not smoke_marker.is_file():
                    smoke_command = conda_python(ENV[method]) + [
                        str(repo / "scripts/train_full_baseline.py"),
                        "--repo",
                        str(repo),
                        "--dataset",
                        dataset,
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--smoke",
                    ]
                    smoke_error = None
                    for attempt in (1, 2):
                        try:
                            run_logged(smoke_command, logs / f"smoke_{dataset}_{method}.log", repo)
                            smoke_error = None
                            break
                        except subprocess.CalledProcessError as exc:
                            smoke_error = exc
                            print(
                                f"RETRYABLE SMOKE FAILURE attempt={attempt}/2 {dataset}/{method}: {exc}",
                                flush=True,
                            )
                            time.sleep(10)
                    if smoke_error is not None:
                        failed = repo / "outputs/status" / f"smoke_{dataset}_{method}.failed"
                        failed.write_text(
                            json.dumps(
                                {
                                    "status": "failed",
                                    "dataset": dataset,
                                    "method": method,
                                    "error": str(smoke_error),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        raise smoke_error
                    smoke_marker.write_text("passed\n", encoding="utf-8")
                    (repo / "outputs/status" / f"smoke_{dataset}_{method}.failed").unlink(
                        missing_ok=True
                    )
                command = conda_python(ENV[method]) + [
                    str(repo / "scripts/train_full_baseline.py"),
                    "--repo",
                    str(repo),
                    "--dataset",
                    dataset,
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                ]
                error = None
                for attempt in (1, 2):
                    try:
                        run_logged(command, logs / f"{dataset}_{method}_seed{seed}.log", repo)
                        error = None
                        break
                    except subprocess.CalledProcessError as exc:
                        error = exc
                        print(
                            f"RETRYABLE FAILURE attempt={attempt}/2 {dataset}/{method}/seed{seed}: {exc}",
                            flush=True,
                        )
                        time.sleep(10)
                if error is not None:
                    failed = repo / "outputs/status" / f"{dataset}_{method}_seed{seed}.failed"
                    failed.write_text(
                        json.dumps(
                            {
                                "status": "failed",
                                "dataset": dataset,
                                "method": method,
                                "seed": seed,
                                "error": str(error),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    raise error
    run_logged(
        conda_python("huang_env")
        + [str(repo / "scripts/evaluate_full_results.py"), "--repo", str(repo)],
        logs / "90_evaluation.log",
        repo,
    )
    print("FINISHED ALL 36 RUNS + EVALUATION", flush=True)


if __name__ == "__main__":
    main()
