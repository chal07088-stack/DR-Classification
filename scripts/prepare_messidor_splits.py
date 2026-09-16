"""Build the patient-level 70/15/15 Messidor-2 split after exact-ID audit."""

import argparse
import json
from pathlib import Path

from drpair.benchmark.messidor import prepare_messidor_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complete-pairs",
        type=Path,
        default=Path("results/data_audit/messidor_complete_pairs.csv"),
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_messidor_splits(
        args.complete_pairs, args.image_root, args.repo_root, seed=args.seed
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
