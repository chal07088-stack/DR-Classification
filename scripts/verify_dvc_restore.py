"""Restore selected real files from the local remote into a new empty cache."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = repo / "data" / "eyepacs"
    # Both eyes, at least one patient per split, without restoring a second 95GB copy.
    selected = []
    for split in ("train", "val", "test"):
        files = sorted((source / split).glob("*/*.jpeg"))
        left = next(p for p in files if p.stem.endswith("_left"))
        right_name = left.name.replace("_left", "_right")
        right = next(p for p in files if p.name == right_name)
        selected.extend([left.relative_to(repo), right.relative_to(repo)])
    with tempfile.TemporaryDirectory(prefix="drpair-dvc-restore-") as tmp:
        target = Path(tmp)
        subprocess.run(["git", "init", "--quiet", str(target)], check=True)

        def dvc(*args):
            subprocess.run([sys.executable, "-m", "dvc", *args], cwd=target, check=True)

        dvc("init", "--quiet")
        dvc("config", "core.analytics", "false")
        dvc("remote", "add", "--default", "local", str(args.store.resolve()))
        (target / "data").mkdir()
        shutil.copy2(repo / "data" / "eyepacs.dvc", target / "data" / "eyepacs.dvc")
        records = []
        for relative in selected:
            dvc("pull", relative.as_posix(), "--remote", "local")
            expected, actual = digest(repo / relative), digest(target / relative)
            if actual != expected:
                raise ValueError(f"Hash mismatch after DVC pull: {relative}")
            records.append({"path": relative.as_posix(), "sha256": actual, "match": True})
    output = repo / "experimental_results" / "repository_validation" / "dvc_restore.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "passed",
                "empty_cache": True,
                "full_dataset_registered": True,
                "restored_files": records,
                "scope": "Six representative files restored from local remote; not a second full copy",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
