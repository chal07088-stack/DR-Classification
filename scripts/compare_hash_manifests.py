#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


def load(path: str) -> dict[str, tuple[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return {
            row["path"]: (row["bytes"], row["sha256"])
            for row in csv.DictReader(stream)
        }


expected = load(sys.argv[1])
observed = load(sys.argv[2])
mismatched = sorted(
    path for path, value in expected.items() if observed.get(path) != value
)
extra = sorted(set(observed) - set(expected))
print(
    f"HASH_AUDIT expected={len(expected)} observed={len(observed)} "
    f"mismatch={len(mismatched)} extra={len(extra)}"
)
for path in mismatched[:50]:
    print(f"MISMATCH {path} expected={expected[path]} observed={observed.get(path)}")
for path in extra[:50]:
    print(f"EXTRA {path} observed={observed[path]}")
raise SystemExit(4 if mismatched or extra else 0)
