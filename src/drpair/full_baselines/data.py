from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

EYE_EXPECTED = {"train": 12293, "val": 2635, "test": 2635}
MESSIDOR_EXPECTED = {"train": 696, "test": 174}
EXPECTED_GRADES = {
    "eyepacs": {0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708},
    "messidor2": {0: 1013, 1: 270, 2: 347, 3: 75, 4: 35},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grade(value: str, context: str) -> int:
    try:
        grade = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid grade at {context}: {value!r}") from error
    if grade not in range(5):
        raise ValueError(f"Grade outside 0..4 at {context}: {grade}")
    return grade


def _split_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["outer_split"]) for row in rows).items()))


def _directional_rows(dataset: str, patients: list[dict[str, object]]):
    rows: list[dict[str, object]] = []
    for patient in patients:
        patient_id = str(patient["patient_id"])
        for target, fellow in (("left", "right"), ("right", "left")):
            rows.append(
                {
                    "dataset": dataset,
                    "split": patient["outer_split"],
                    "sample_id": f"{patient_id}_{target}",
                    "patient_id": patient_id,
                    "target_eye": target,
                    "fellow_eye": fellow,
                    "target_image_path": patient[f"{target}_image_path"],
                    "fellow_image_path": patient[f"{fellow}_image_path"],
                    "y_true": patient[f"{target}_grade"],
                    "fellow_y_true": patient[f"{fellow}_grade"],
                    "inter_eye_gap": patient["gap"],
                    "gap_group": "gap0"
                    if patient["gap"] == 0
                    else ("gap1" if patient["gap"] == 1 else "gap2plus"),
                }
            )
    return rows


def _audit_common(dataset: str, patients: list[dict[str, object]], expected: dict[str, int]):
    ids = [str(row["patient_id"]) for row in patients]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: duplicate patient IDs")
    if _split_counts(patients) != expected:
        raise ValueError(f"{dataset}: split counts {_split_counts(patients)} != {expected}")
    split_sets = {
        split: {str(row["patient_id"]) for row in patients if row["outer_split"] == split}
        for split in expected
    }
    for left_index, left in enumerate(expected):
        for right in list(expected)[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"{dataset}: patient leakage {left}/{right}: {len(overlap)}")
    image_paths = [
        str(row[key]) for row in patients for key in ("left_image_path", "right_image_path")
    ]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError(f"{dataset}: duplicate image paths")
    grades = Counter(int(row[key]) for row in patients for key in ("left_grade", "right_grade"))
    if dict(sorted(grades.items())) != EXPECTED_GRADES[dataset]:
        raise ValueError(
            f"{dataset}: grade distribution {dict(grades)} != {EXPECTED_GRADES[dataset]}"
        )
    return grades


def build_eyepacs(
    authoritative_split: Path,
    source_manifest: Path,
    image_root: Path,
) -> list[dict[str, object]]:
    split_rows = _read_csv(authoritative_split)
    source_rows = _read_csv(source_manifest)
    source: dict[tuple[str, str], dict[str, str]] = {}
    for row in source_rows:
        key = (row["patient"].strip(), row["side"].strip().lower())
        if key in source:
            raise ValueError(f"EyePACS source duplicate: {key}")
        source[key] = row
    patients = []
    for row in split_rows:
        patient_id = row["patient_id"].strip()
        if row.get("included", "True").strip().lower() not in {"true", "1", "yes"}:
            raise ValueError(f"EyePACS authoritative row unexpectedly excluded: {patient_id}")
        outer = row["outer_split"].strip()
        pair: dict[str, object] = {"patient_id": patient_id, "outer_split": outer}
        for side in ("left", "right"):
            key = (patient_id, side)
            if key not in source:
                raise FileNotFoundError(f"EyePACS source manifest missing {key}")
            source_row = source[key]
            grade = _grade(row[f"{side}_grade"], f"EyePACS {patient_id}/{side}")
            if _grade(source_row["label"], f"EyePACS source {patient_id}/{side}") != grade:
                raise ValueError(f"EyePACS label mismatch: {patient_id}/{side}")
            relative = Path(source_row["path"])
            path = image_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            fellow_relative = Path(source_row["fellow_path"])
            if not (image_root / fellow_relative).is_file():
                raise FileNotFoundError(image_root / fellow_relative)
            pair[f"{side}_grade"] = grade
            pair[f"{side}_image_path"] = relative.as_posix()
        pair["gap"] = abs(int(pair["left_grade"]) - int(pair["right_grade"]))
        if pair["gap"] != int(row["gap"]):
            raise ValueError(f"EyePACS gap mismatch: {patient_id}")
        patients.append(pair)
    _audit_common("eyepacs", patients, EYE_EXPECTED)
    return patients


def build_messidor(
    authoritative_split: Path,
    image_root: Path,
) -> list[dict[str, object]]:
    rows = _read_csv(authoritative_split)
    image_index = {path.name.lower(): path for path in image_root.iterdir() if path.is_file()}
    if len(image_index) != 1748:
        raise ValueError(f"Messidor local image count is {len(image_index)}, expected 1748")
    patients = []
    for row in rows:
        patient_id = row["examination_id"].strip()
        pair: dict[str, object] = {
            "patient_id": patient_id,
            "outer_split": row["outer_split"].strip(),
        }
        for side in ("left", "right"):
            image_id = row[f"{side}_image_id"].strip()
            if image_id.lower() not in image_index:
                raise FileNotFoundError(f"Messidor image missing: {image_id}")
            pair[f"{side}_image_path"] = image_id
            pair[f"{side}_grade"] = _grade(row[f"{side}_grade"], f"Messidor {patient_id}/{side}")
        pair["gap"] = abs(int(pair["left_grade"]) - int(pair["right_grade"]))
        if pair["gap"] != int(row["gap"]):
            raise ValueError(f"Messidor gap mismatch: {patient_id}")
        patients.append(pair)
    _audit_common("messidor2", patients, MESSIDOR_EXPECTED)
    return patients


def _gap_rows(dataset: str, patients: list[dict[str, object]]):
    output = []
    for split in sorted({str(row["outer_split"]) for row in patients}) + ["all"]:
        selected = (
            patients if split == "all" else [row for row in patients if row["outer_split"] == split]
        )
        for group, predicate in (
            ("gap0", lambda value: value == 0),
            ("gap1", lambda value: value == 1),
            ("gap2plus", lambda value: value >= 2),
        ):
            group_rows = [row for row in selected if predicate(int(row["gap"]))]
            grades = Counter(
                int(row[key]) for row in group_rows for key in ("left_grade", "right_grade")
            )
            output.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "gap_group": group,
                    "patient_count": len(group_rows),
                    "percent": 100 * len(group_rows) / len(selected) if selected else 0,
                    **{f"grade_{grade}_eyes": grades[grade] for grade in range(5)},
                }
            )
    return output


def _joint_rows(dataset: str, patients: list[dict[str, object]]):
    counts = Counter(
        (str(row["outer_split"]), int(row["left_grade"]), int(row["right_grade"]))
        for row in patients
    )
    return [
        {
            "dataset": dataset,
            "split": split,
            "left_grade": left,
            "right_grade": right,
            "patient_count": count,
        }
        for (split, left, right), count in sorted(counts.items())
    ]


def audit_all(
    repo_root: Path,
    eyepacs_root: Path,
    messidor_root: Path,
    output_root: Path,
) -> dict[str, object]:
    split_root = repo_root / "data" / "splits"
    eye_split = split_root / "eyepacs_proposed_split_seed2026.csv"
    mess_split = split_root / "messidor2_proposed_split_seed2026.csv"
    for path in (eye_split, mess_split):
        if not path.is_file():
            raise FileNotFoundError(f"Authoritative split missing: {path}")
    eye = build_eyepacs(eye_split, split_root / "eyepacs.csv", eyepacs_root)
    mess = build_messidor(mess_split, messidor_root)
    output_root.mkdir(parents=True, exist_ok=True)
    patient_fields = [
        "patient_id",
        "left_image_path",
        "right_image_path",
        "left_grade",
        "right_grade",
        "gap",
        "outer_split",
    ]
    direction_fields = [
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
    _write_csv(output_root / "eyepacs_full_pair_population.csv", eye, patient_fields)
    _write_csv(output_root / "messidor_complete_pairs.csv", mess, patient_fields)
    _write_csv(
        output_root / "eyepacs_directional_manifest.csv",
        _directional_rows("eyepacs", eye),
        direction_fields,
    )
    _write_csv(
        output_root / "messidor2_directional_manifest.csv",
        _directional_rows("messidor2", mess),
        direction_fields,
    )
    gap = _gap_rows("eyepacs", eye) + _gap_rows("messidor2", mess)
    _write_csv(
        output_root / "gap_distribution.csv",
        gap,
        ["dataset", "split", "gap_group", "patient_count", "percent"]
        + [f"grade_{g}_eyes" for g in range(5)],
    )
    _write_csv(
        output_root / "eyepacs_gap_distribution.csv",
        [r for r in gap if r["dataset"] == "eyepacs"],
        ["dataset", "split", "gap_group", "patient_count", "percent"]
        + [f"grade_{g}_eyes" for g in range(5)],
    )
    _write_csv(
        output_root / "messidor_gap_distribution.csv",
        [r for r in gap if r["dataset"] == "messidor2"],
        ["dataset", "split", "gap_group", "patient_count", "percent"]
        + [f"grade_{g}_eyes" for g in range(5)],
    )
    _write_csv(
        output_root / "joint_grade_distribution.csv",
        _joint_rows("eyepacs", eye) + _joint_rows("messidor2", mess),
        ["dataset", "split", "left_grade", "right_grade", "patient_count"],
    )
    summary = {
        "status": "passed",
        "authoritative_split_sha256": {
            "eyepacs": _sha256(eye_split),
            "messidor2": _sha256(mess_split),
        },
        "datasets": {
            "eyepacs": {"patients": len(eye), "eyes": 2 * len(eye), "splits": _split_counts(eye)},
            "messidor2": {
                "patients": len(mess),
                "eyes": 2 * len(mess),
                "splits": _split_counts(mess),
            },
        },
    }
    (output_root / "data_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
