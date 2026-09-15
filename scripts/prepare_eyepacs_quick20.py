"""Create and hard-gate the deterministic EyePACS Quick20 patient split."""

import argparse
import json
from pathlib import Path

from drpair.benchmark.eyepacs import prepare_quick20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True, help="Actual local trainLabels.csv")
    parser.add_argument("--image-root", type=Path, default=Path("data/eyepacs/train"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_quick20(args.labels, args.image_root, args.repo_root, seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

