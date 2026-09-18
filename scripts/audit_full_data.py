from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from drpair.full_baselines.data import audit_all
from drpair.full_baselines.runtime import crop_black, read_rows


def messidor_stats(repo: Path, image_root: Path, output_root: Path) -> None:
    manifest = read_rows(output_root / "messidor2_directional_manifest.csv")
    files = sorted({r["target_image_path"] for r in manifest if r["split"] == "train"})
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq = np.zeros(3, dtype=np.float64)
    pixels = 0
    for name in files:
        image = crop_black(Image.open(image_root / name)).resize(
            (512, 512), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float64) / 255.0
        flat = array.reshape(-1, 3)
        channel_sum += flat.sum(0)
        channel_sq += np.square(flat).sum(0)
        pixels += len(flat)
    mean = channel_sum / pixels
    std = np.sqrt(channel_sq / pixels - np.square(mean))
    (output_root / "messidor_rgb_stats.json").write_text(
        json.dumps(
            {
                "source": "outer_train_only_black_crop_resize512",
                "images": len(files),
                "pixels": pixels,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--eyepacs-root", type=Path, default=None)
    parser.add_argument("--messidor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    summary = audit_all(
        repo,
        (args.eyepacs_root or repo / "data" / "eyepacs").resolve(),
        args.messidor_root.resolve(),
        (args.output_root or repo / "outputs" / "data_audit").resolve(),
    )
    messidor_stats(
        repo,
        args.messidor_root.resolve(),
        (args.output_root or repo / "outputs" / "data_audit").resolve(),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
