"""Create a deterministic SHA256 manifest for an image tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(job: tuple[Path, Path]) -> tuple[str, int, str]:
    root, path = job
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return path.relative_to(root).as_posix(), path.stat().st_size, value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("splits", nargs="+", choices=["train", "val", "test"])
    args = parser.parse_args()
    root = args.root.resolve()
    files = sorted(
        path
        for split in args.splits
        for path in (root / split).rglob("*")
        if path.is_file()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "bytes", "sha256"])
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, row in enumerate(
                executor.map(digest, ((root, path) for path in files)), start=1
            ):
                writer.writerow(row)
                if index % 1000 == 0 or index == len(files):
                    print(f"HASH {index}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
