"""Audit official Messidor-2 pairs against third-party adjudicated grades."""

import argparse
import json
from pathlib import Path

from drpair.benchmark.messidor import audit_messidor_pairing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--left-column", required=True)
    parser.add_argument("--right-column", required=True)
    parser.add_argument("--examination-column")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/data_audit"))
    args = parser.parse_args()
    result = audit_messidor_pairing(
        args.pairing,
        args.grades,
        args.output,
        args.left_column,
        args.right_column,
        args.examination_column,
        args.image_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
