"""Audit and materialize the official EyePACS train/public/private split.

The source manifest contains the original Kaggle image locations and labels.
The competition solution is used as an independent exact-ID audit for the
10,906 public-validation and 42,670 private-test images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_SPLITS = {"train": 35126, "val": 10906, "test": 42670}
USAGE_TO_SPLIT = {"Public": "val", "Private": "test"}
FIELDS = [
    "dataset",
    "split",
    "sample_id",
    "patient_id",
    "target_eye",
    "fellow_eye",
    "target_image_path",
    "fellow_image_path",
    "y_true",
    "fellow_y_true",
    "inter_eye_gap",
    "gap_group",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = args.output_root.resolve()
    source_path = repo / "data/splits/eyepacs.csv"
    solution_path = repo / "data/eyepacs/retinopathy_solution.csv"
    image_root = repo / "data/eyepacs"

    source_rows = read_csv(source_path)
    solution_rows = read_csv(solution_path)
    if Counter(row["split"] for row in source_rows) != Counter(EXPECTED_SPLITS):
        raise RuntimeError(f"unexpected source split counts: {Counter(row['split'] for row in source_rows)}")

    solution: dict[str, tuple[int, str]] = {}
    for row in solution_rows:
        image_id = row["image"].strip()
        if image_id in solution:
            raise RuntimeError(f"duplicate solution image ID: {image_id}")
        usage = row["Usage"].strip()
        if usage not in USAGE_TO_SPLIT:
            raise RuntimeError(f"unexpected Usage={usage!r} for {image_id}")
        solution[image_id] = (int(row["level"]), USAGE_TO_SPLIT[usage])
    if len(solution) != 53576:
        raise RuntimeError(f"solution row count is {len(solution)}, expected 53576")

    by_patient: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    missing_images: list[str] = []
    audited_solution_ids: set[str] = set()
    for row in source_rows:
        patient = row["patient"].strip()
        side = row["side"].strip().lower()
        split = row["split"].strip()
        if side not in {"left", "right"} or split not in EXPECTED_SPLITS:
            raise RuntimeError(f"invalid row: patient={patient} side={side} split={split}")
        if side in by_patient[patient]:
            raise RuntimeError(f"duplicate patient/side: {patient}/{side}")
        relative = Path(row["path"])
        if not (image_root / relative).is_file():
            missing_images.append(relative.as_posix())
        image_id = relative.stem
        grade = int(row["label"])
        if split != "train":
            if image_id not in solution:
                raise RuntimeError(f"official val/test image absent from solution: {image_id}")
            solution_grade, solution_split = solution[image_id]
            if (grade, split) != (solution_grade, solution_split):
                raise RuntimeError(
                    f"solution mismatch {image_id}: source={(grade, split)} solution={(solution_grade, solution_split)}"
                )
            audited_solution_ids.add(image_id)
        by_patient[patient][side] = row
    if missing_images:
        raise RuntimeError(f"missing images ({len(missing_images)}): {missing_images[:10]}")
    if audited_solution_ids != set(solution):
        raise RuntimeError(
            f"solution exact-ID audit failed: matched={len(audited_solution_ids)} labels={len(solution)}"
        )

    directional: list[dict[str, object]] = []
    split_patients: Counter[str] = Counter()
    for patient, pair in sorted(by_patient.items(), key=lambda item: item[0]):
        if set(pair) != {"left", "right"}:
            raise RuntimeError(f"incomplete pair: {patient} has {sorted(pair)}")
        if pair["left"]["split"] != pair["right"]["split"]:
            raise RuntimeError(f"split leakage inside pair: {patient}")
        split = pair["left"]["split"]
        split_patients[split] += 1
        left_grade, right_grade = int(pair["left"]["label"]), int(pair["right"]["label"])
        gap = abs(left_grade - right_grade)
        for target, fellow in (("left", "right"), ("right", "left")):
            directional.append(
                {
                    "dataset": "eyepacs",
                    "split": split,
                    "sample_id": f"{patient}_{target}",
                    "patient_id": patient,
                    "target_eye": target,
                    "fellow_eye": fellow,
                    "target_image_path": pair[target]["path"],
                    "fellow_image_path": pair[fellow]["path"],
                    "y_true": pair[target]["label"],
                    "fellow_y_true": pair[fellow]["label"],
                    "inter_eye_gap": gap,
                    "gap_group": "gap0" if gap == 0 else ("gap1" if gap == 1 else "gap2plus"),
                }
            )

    actual = Counter(str(row["split"]) for row in directional)
    if actual != Counter(EXPECTED_SPLITS):
        raise RuntimeError(f"official directional split mismatch: {actual}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "eyepacs_directional_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(directional)
    audit = {
        "status": "passed",
        "protocol": "official_kaggle_eyepacs_train_public_validation_private_test",
        "eyes": dict(sorted(actual.items())),
        "patients": dict(sorted(split_patients.items())),
        "solution_exact_matches": len(audited_solution_ids),
        "source_manifest_sha256": sha256(source_path),
        "solution_sha256": sha256(solution_path),
        "directional_manifest_sha256": sha256(manifest_path),
    }
    (output_root / "official_eyepacs_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
