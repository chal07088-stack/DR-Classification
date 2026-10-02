"""Materialize the official GDRBench MESSIDOR split and preprocessing.

This is intentionally separate from the common patient-level benchmark.  It
reproduces the image-level GDRBench protocol used by the D-ODR paper and never
modifies the original Messidor-2 images or common split files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/chehx/DGDR/main/GDRBench/splits/MESSIDOR_train.txt",
    "crossval": "https://raw.githubusercontent.com/chehx/DGDR/main/GDRBench/splits/MESSIDOR_crossval.txt",
}
EXPECTED = {"train": 1384, "crossval": 360}
GRADE_DIR = {0: "nodr", 1: "mild_npdr", 2: "moderate_npdr", 3: "severe_npdr", 4: "pdr"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    payload = urllib.request.urlopen(url, timeout=60).read()
    path.write_bytes(payload)


def read_split(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        relative, label = line.rsplit(maxsplit=1)
        grade = int(label)
        if grade not in range(5):
            raise ValueError(f"invalid grade in {path}: {line!r}")
        rows.append((relative, grade))
    return rows


def official_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            value = row["adjudicated_dr_grade"].strip()
            if not value:
                continue
            name = Path(row["image_id"]).name.lower()
            grade = int(value)
            if grade not in range(5):
                raise ValueError(f"invalid adjudicated grade: {row}")
            if name in labels and labels[name] != grade:
                raise ValueError(f"conflicting local grade: {name}")
            labels[name] = grade
    return labels


def get_mask_bz(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    threshold = np.mean(gray) / 3 - 5
    _, mask = cv2.threshold(gray, max(0, threshold), 1, cv2.THRESH_BINARY)
    flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    new_mask = (1 - mask).astype(np.uint8)
    _, new_mask, _, _ = cv2.floodFill(new_mask, flood_mask, (0, 0), 0, cv2.FLOODFILL_MASK_ONLY)
    _, new_mask, _, _ = cv2.floodFill(
        new_mask,
        flood_mask,
        (new_mask.shape[1] - 1, new_mask.shape[0] - 1),
        0,
        cv2.FLOODFILL_MASK_ONLY,
    )
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    return cv2.dilate(cv2.erode(mask, kernel), kernel)


def fundus_mask(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    half = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    normalized = cv2.normalize(half, None, 0, 255, cv2.NORM_MINMAX)
    rough = get_mask_bz(normalized)
    rows = rough.sum(axis=1)
    cols = rough.sum(axis=0)
    center = [np.where(rows > rows.max() * 0.95)[0].mean(), np.where(cols > cols.max() * 0.95)[0].mean()]
    ksize = max(rough.shape[1] // 400 * 2 + 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    edge = cv2.morphologyEx(rough.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    indexes = np.where(edge > 0)
    distances = np.sqrt((indexes[0] - center[0]) ** 2 + (indexes[1] - center[1]) ** 2)
    counts = np.bincount(np.ceil(distances).astype(np.int_))
    radius = int(np.where(counts > counts.max() * 0.995)[0].max() * 2)
    center = [center[0] * 2, center[1] * 2]
    start_h = max(0, int(center[0] - radius))
    start_w = max(0, int(center[1] - radius))
    bbox = (start_h, start_w, min(height - start_h, 2 * radius), min(width - start_w, 2 * radius))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(center[1]), int(center[0])), radius, 1, -1)
    return mask, bbox


def preprocess(source: Path, destination: Path) -> None:
    bgr = cv2.imread(str(source))
    if bgr is None:
        raise FileNotFoundError(source)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask, (start_h, start_w, crop_h, crop_w) = fundus_mask(rgb)
    rgb[mask <= 0] = 0
    cropped = rgb[start_h : start_h + crop_h, start_w : start_w + crop_w]
    side = max(cropped.shape[:2])
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top = int(side / 2 - cropped.shape[0] / 2)
    left = int(side / 2 - cropped.shape[1] / 2)
    square[top : top + cropped.shape[0], left : left + cropped.shape[1]] = cropped
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(square).resize((512, 512), Image.Resampling.BILINEAR).save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    split_root = output / "splits"
    image_root = output / "images"
    split_root.mkdir(parents=True, exist_ok=True)
    source_root = repo / "data/messidor2/IMAGES"
    label_path = repo / "data/messidor2/messidor_data.csv"
    local_files = {path.name.lower(): path for path in source_root.iterdir() if path.is_file()}
    labels = official_labels(label_path)
    if len(local_files) != 1748 or len(labels) != 1744:
        raise RuntimeError(f"unexpected local inventory: images={len(local_files)} graded={len(labels)}")

    all_names: list[str] = []
    gdrbench_only_labels: list[dict[str, object]] = []
    split_audit: dict[str, object] = {}
    for split, url in SPLIT_URLS.items():
        path = split_root / f"MESSIDOR_{split}.txt"
        download(url, path)
        rows = read_split(path)
        if len(rows) != EXPECTED[split]:
            raise RuntimeError(f"{split}: {len(rows)} rows != {EXPECTED[split]}")
        distribution = Counter()
        for relative, grade in rows:
            name = Path(relative).name
            key = name.lower()
            if key not in local_files:
                raise FileNotFoundError(f"GDRBench ID absent locally: {name}")
            if key in labels and labels[key] != grade:
                raise RuntimeError(f"grade mismatch {name}: GDRBench={grade} local={labels[key]}")
            if key not in labels:
                gdrbench_only_labels.append({"filename": name, "grade": grade, "split": split})
            expected_dir = GRADE_DIR[grade]
            if Path(relative).parent.name != expected_dir:
                raise RuntimeError(f"grade directory mismatch: {relative}")
            destination = image_root / relative
            if not destination.is_file():
                preprocess(local_files[key], destination)
            distribution[grade] += 1
            all_names.append(key)
        split_audit[split] = {
            "rows": len(rows),
            "grade_distribution": dict(sorted(distribution.items())),
            "sha256": sha256(path),
        }
        print(f"PREPARED {split}: {len(rows)}", flush=True)

    if len(all_names) != 1744 or len(set(all_names)) != 1744:
        raise RuntimeError(f"split union invalid: rows={len(all_names)} unique={len(set(all_names))}")
    if gdrbench_only_labels:
        raise RuntimeError(f"GDRBench contains IDs without adjudicated grades: {gdrbench_only_labels}")
    processed = list((image_root / "MESSIDOR").glob("*/*"))
    if len(processed) != 1744:
        raise RuntimeError(f"processed cache count {len(processed)} != 1744")
    audit = {
        "status": "passed",
        "protocol": "official_GDRBench_MESSIDOR_image_level_split_and_rvsize_preprocessing",
        "source_images": len(local_files),
        "exact_matched_adjudicated_images": len(labels),
        "gdrbench_only_labels": gdrbench_only_labels,
        "processed_images": len(processed),
        "splits": split_audit,
        "note": "All 1,744 non-empty adjudicated grades exact-match the official GDRBench split. Four ungraded local images are excluded by GDRBench.",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
