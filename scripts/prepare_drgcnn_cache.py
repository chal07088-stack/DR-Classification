"""Build the deterministic DRGCNN 512 px image cache from authoritative manifests.

This reproduces third_party/DRGCNN/eye_pre_process/crop.py without modifying
the source images. Only images in the verified bilateral population are cached.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def square_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    left = max((width - height) // 2, 0)
    return left, 0, min(width - (width - height) // 2, width), height


def official_crop(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    blurred = image.filter(ImageFilter.BLUR)
    array = np.asarray(blurred)
    height, width, _ = array.shape
    bbox = None
    if width > 1.2 * height:
        edge = max(width // 32, 1)
        left_max = array[:, :edge, :].max(axis=(0, 1)).astype(int)
        right_max = array[:, -edge:, :].max(axis=(0, 1)).astype(int)
        max_background = np.maximum(left_max, right_max)
        foreground = (array > max_background + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()
        if bbox is not None:
            left, upper, right, lower = bbox
            if right - left < 0.8 * height or lower - upper < 0.8 * height:
                bbox = None
    if bbox is None:
        bbox = square_bbox(image)
    return image.crop(bbox).resize((512, 512), Image.Resampling.LANCZOS)


def convert_one(job: tuple[str, str]) -> tuple[str, str | None]:
    source_text, target_text = job
    source, target = Path(source_text), Path(target_text)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            converted = official_crop(image)
            suffix = target.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                converted.save(target, quality=100, subsampling=0)
            else:
                converted.save(target)
        with Image.open(target) as check:
            if check.size != (512, 512):
                raise RuntimeError(f"invalid cached dimensions {check.size}")
        return source_text, None
    except Exception as exc:  # returned to the parent for a complete audit
        return source_text, f"{type(exc).__name__}: {exc}"


def image_roots(repo: Path, dataset: str) -> tuple[Path, Path, Path]:
    if dataset == "eyepacs":
        source = repo / "data" / "eyepacs"
        official = (
            repo
            / "outputs"
            / "official_split"
            / "data_audit"
            / "eyepacs_directional_manifest.csv"
        )
        manifest = (
            official
            if official.is_file()
            else repo / "outputs" / "data_audit" / "eyepacs_directional_manifest.csv"
        )
    else:
        source = repo / "data" / "messidor2" / "IMAGES"
        manifest = repo / "outputs" / "data_audit" / "messidor2_directional_manifest.csv"
    target = repo / "data" / "cache" / "drgcnn" / dataset
    return source, target, manifest


def channel_stats(paths: list[Path]) -> tuple[list[float], list[float]]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq = np.zeros(3, dtype=np.float64)
    pixels = 0
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        flat = array.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sq += np.square(flat).sum(axis=0)
        pixels += len(flat)
    mean = channel_sum / pixels
    variance = np.maximum(channel_sq / pixels - np.square(mean), 0.0)
    return mean.tolist(), np.sqrt(variance).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=["eyepacs", "messidor2"], required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        help="Optional manifest splits to cache (use train/val before model freeze).",
    )
    parser.add_argument("--source-overlay", type=Path)
    parser.add_argument("--target-root", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source_root, target_root, manifest_path = image_roots(repo, args.dataset)
    overlay_root = args.source_overlay.resolve() if args.source_overlay else None
    if args.target_root:
        target_root = args.target_root.resolve()
    rows = read_rows(manifest_path)
    selected_splits = set(args.splits or {row["split"] for row in rows})
    selected_rows = [row for row in rows if row["split"] in selected_splits]
    relative_paths = sorted(
        {
            row[key]
            for row in selected_rows
            for key in ("target_image_path", "fellow_image_path")
        }
    )
    def source_for(path: str) -> Path:
        overlay = overlay_root / path if overlay_root else None
        return overlay if overlay is not None and overlay.is_file() else source_root / path

    missing_source = [path for path in relative_paths if not source_for(path).is_file()]
    if missing_source:
        raise RuntimeError(f"missing source images ({len(missing_source)}): {missing_source[:10]}")
    valid_existing: list[str] = []
    for path in relative_paths:
        cached = target_root / path
        if not cached.is_file():
            continue
        try:
            with Image.open(cached) as check:
                if check.size == (512, 512):
                    # A stopped worker may have left a JPEG with a readable
                    # header but truncated pixels. Decode before resuming.
                    check.load()
                    valid_existing.append(path)
        except OSError:
            pass
    valid_existing_set = set(valid_existing)
    jobs = [
        (str(source_for(path)), str(target_root / path))
        for path in relative_paths
        if path not in valid_existing_set
    ]
    failures: list[dict[str, str]] = []
    completed = len(valid_existing)
    print(
        f"CACHE {args.dataset} resume_valid={completed} remaining={len(jobs)}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for source, error in executor.map(convert_one, jobs, chunksize=8):
            completed += 1
            if error:
                failures.append({"source": source, "error": error})
            if completed % 1000 == 0 or completed == len(relative_paths):
                print(
                    f"CACHE {args.dataset} {completed}/{len(relative_paths)} failures={len(failures)}",
                    flush=True,
                )
    missing_cache = [path for path in relative_paths if not (target_root / path).is_file()]
    audit: dict[str, object] = {
        "status": "passed" if not failures and not missing_cache else "failed",
        "dataset": args.dataset,
        "manifest_rows": len(rows),
        "selected_manifest_rows": len(selected_rows),
        "selected_splits": sorted(selected_splits),
        "manifest_sha256": checksum(manifest_path),
        "unique_manifest_images": len(relative_paths),
        "completed": completed,
        "resumed_valid_cache": len(valid_existing),
        "missing_source": missing_source,
        "missing_cache": missing_cache,
        "failures": failures,
        "algorithm": "official_blur_threshold_square_fallback_lanczos_512",
        "source_root": str(source_root),
        "source_overlay": str(overlay_root) if overlay_root else None,
        "cache_root": str(target_root),
    }
    if not failures and not missing_cache and args.dataset == "messidor2":
        train_paths = sorted(
            {
                row["target_image_path"]
                for row in selected_rows
                if row["split"] == "train"
            }
        )
        mean, std = channel_stats([target_root / path for path in train_paths])
        audit["train_rgb_stats"] = {"mean": mean, "std": std, "images": len(train_paths)}
        stats_path = repo / "outputs" / "data_audit" / "drgcnn_messidor2_rgb_stats.json"
        stats_path.write_text(json.dumps(audit["train_rgb_stats"], indent=2), encoding="utf-8")
    audit_path = repo / "outputs" / "data_audit" / f"drgcnn_{args.dataset}_cache.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if audit["status"] != "passed":
        raise RuntimeError(f"DRGCNN cache audit failed: {audit_path}")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
