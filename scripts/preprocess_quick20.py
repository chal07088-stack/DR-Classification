"""Precompute the official Huang black-border crop for all Quick20 images."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def square_bbox(image: Image.Image):
    width, height = image.size
    left = max((width - height) // 2, 0)
    return left, 0, min(width - (width - height) // 2, width), height


def convert(job):
    source_path, target_path, size = job
    target_path = Path(target_path)
    if target_path.is_file():
        return "cached"
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        blurred = np.asarray(image.filter(ImageFilter.BLUR))
        height, width, _ = blurred.shape
        bbox = None
        if width > 1.2 * height:
            edge = max(1, width // 32)
            background = np.maximum(
                blurred[:, :edge, :].max(axis=(0, 1)).astype(int),
                blurred[:, -edge:, :].max(axis=(0, 1)).astype(int),
            )
            foreground = (blurred > background + 10).astype(np.uint8)
            candidate = Image.fromarray(foreground).getbbox()
            if candidate is not None:
                left, upper, right, lower = candidate
                if right - left >= 0.8 * height and lower - upper >= 0.8 * height:
                    bbox = candidate
        cropped = image.crop(bbox or square_bbox(image)).resize(
            (size, size), Image.Resampling.LANCZOS
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
        cropped.save(temporary, format="JPEG", quality=95, subsampling=0)
        temporary.replace(target_path)
    return "created"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=[
        "data/splits/eyepacs_quick20_symmetry_train.csv",
        "data/splits/eyepacs_quick20_symmetry_val.csv",
        "data/splits/eyepacs_quick20_symmetry_test.csv",
    ])
    parser.add_argument("--output", default="data/cache/quick20_huang_512")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = {}
    for split in args.splits:
        with Path(split).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                paths[Path(row["image_path"]).stem] = Path(row["image_path"])
                paths[Path(row["fellow_path"]).stem] = Path(row["fellow_path"])
    output = Path(args.output)
    jobs = [(source, output / f"{stem}.jpg", args.size) for stem, source in sorted(paths.items())]
    created = cached = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, status in enumerate(pool.map(convert, jobs), 1):
            created += status == "created"
            cached += status == "cached"
            if index % 250 == 0 or index == len(jobs):
                print(f"preprocess {index}/{len(jobs)} created={created} cached={cached}", flush=True)
    print(f"complete images={len(jobs)} created={created} cached={cached}")


if __name__ == "__main__":
    main()
