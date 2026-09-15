"""Exact-ID Messidor-2 pairing and adjudicated-grade provenance audit."""

from __future__ import annotations

import csv
import unicodedata
from collections import Counter
from pathlib import Path, PurePath

from drpair.data.manifest import write_csv
from drpair.runtime import file_hash, save_json


def normalize_image_id(value: str) -> str:
    """Conservative filename normalization: basename, Unicode normalization, case-folding."""
    text = unicodedata.normalize("NFKC", str(value).strip()).replace("\\", "/")
    name = PurePath(text).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid image ID: {value!r}")
    return name.casefold()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader), list(reader.fieldnames)


def audit_messidor_pairing(
    pairing_path: Path,
    grade_path: Path,
    output_root: Path,
    left_column: str,
    right_column: str,
    examination_column: str | None = None,
    pairing_source: str = "https://www.adcis.net/en/third-party/messidor2/",
    grade_source: str = "https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades",
) -> dict[str, object]:
    pairing_path, grade_path, output_root = map(Path, (pairing_path, grade_path, output_root))
    pairing_rows, pairing_schema = _read_csv(pairing_path)
    grade_rows, grade_schema = _read_csv(grade_path)
    required_pairing = {left_column, right_column}
    if examination_column:
        required_pairing.add(examination_column)
    if not required_pairing.issubset(pairing_schema):
        raise ValueError(f"Pairing columns {sorted(required_pairing)} not in {pairing_schema}")
    required_grades = {"image_id", "adjudicated_dr_grade", "adjudicated_gradable"}
    if not required_grades.issubset(grade_schema):
        raise ValueError(f"Grade columns {sorted(required_grades)} not in {grade_schema}")

    grades_by_id: dict[str, dict[str, str]] = {}
    duplicate_grade_ids = []
    conflicting_labels = []
    for row in grade_rows:
        normalized = normalize_image_id(row["image_id"])
        if normalized in grades_by_id:
            duplicate_grade_ids.append(normalized)
            prior = grades_by_id[normalized]
            if (
                prior["adjudicated_dr_grade"] != row["adjudicated_dr_grade"]
                or prior["adjudicated_gradable"] != row["adjudicated_gradable"]
            ):
                conflicting_labels.append(normalized)
            continue
        grades_by_id[normalized] = row

    join_rows = []
    complete_pairs = []
    official_ids = []
    duplicate_official_ids = []
    seen_official = set()
    for index, row in enumerate(pairing_rows, start=1):
        examination_id = row[examination_column].strip() if examination_column else f"pairing-row-{index}"
        joined = {}
        for side, column in (("left", left_column), ("right", right_column)):
            official_id = row[column].strip()
            normalized = normalize_image_id(official_id)
            official_ids.append(normalized)
            if normalized in seen_official:
                duplicate_official_ids.append(normalized)
            seen_official.add(normalized)
            grade = grades_by_id.get(normalized)
            gradable = grade is not None and grade["adjudicated_gradable"].strip() == "1"
            grade_text = grade["adjudicated_dr_grade"].strip() if grade else ""
            valid_grade = gradable and grade_text in {"0", "1", "2", "3", "4"}
            joined[side] = (official_id, normalized, grade_text, valid_grade)
            join_rows.append(
                {
                    "examination_id": examination_id,
                    "side": side,
                    "official_image_id": official_id,
                    "normalized_image_id": normalized,
                    "grade_row_matched": grade is not None,
                    "gradable": gradable,
                    "adjudicated_dr_grade": grade_text,
                    "status": "matched_gradable" if valid_grade else "unmatched_or_ungradable",
                }
            )
        if joined["left"][3] and joined["right"][3]:
            complete_pairs.append(
                {
                    "examination_id": examination_id,
                    "left_image_id": joined["left"][0],
                    "right_image_id": joined["right"][0],
                    "left_grade": joined["left"][2],
                    "right_grade": joined["right"][2],
                    "gap": abs(int(joined["left"][2]) - int(joined["right"][2])),
                }
            )

    official_set = set(official_ids)
    grade_set = set(grades_by_id)
    unmatched_official = sorted(official_set - grade_set)
    unmatched_grades = sorted(grade_set - official_set)
    unmatched_rows = [
        {"source": "official_pairing", "normalized_image_id": value} for value in unmatched_official
    ] + [{"source": "grade_annotations", "normalized_image_id": value} for value in unmatched_grades]
    gradable_rows = sum(row["adjudicated_gradable"].strip() == "1" for row in grade_rows)
    exact_matched = sum(row["grade_row_matched"] for row in join_rows)
    failures = []
    if duplicate_official_ids:
        failures.append(f"duplicate official image IDs={len(set(duplicate_official_ids))}")
    if duplicate_grade_ids:
        failures.append(f"duplicate grade image IDs={len(set(duplicate_grade_ids))}")
    if conflicting_labels:
        failures.append(f"conflicting grade labels={len(set(conflicting_labels))}")
    if len(pairing_rows) != 874:
        failures.append(f"official examination rows={len(pairing_rows)}, expected 874")
    if len(official_ids) != 1748:
        failures.append(f"official image rows={len(official_ids)}, expected 1748")

    summary = {
        "phase": "PHASE 1 - Messidor-2 exact-ID join audit",
        "status": "passed" if not failures else "failed",
        "pairing_source": pairing_source,
        "grade_source": grade_source,
        "pairing_local_filename": pairing_path.name,
        "grade_local_filename": grade_path.name,
        "pairing_sha256": file_hash(pairing_path),
        "grade_sha256": file_hash(grade_path),
        "pairing_schema": pairing_schema,
        "grade_schema": grade_schema,
        "official_examinations": len(pairing_rows),
        "official_images": len(official_ids),
        "grade_annotation_rows": len(grade_rows),
        "gradable_annotation_rows": gradable_rows,
        "exact_matched_images": exact_matched,
        "unmatched_official_images": len(unmatched_official),
        "unmatched_grade_rows": len(unmatched_grades),
        "duplicate_official_ids": len(set(duplicate_official_ids)),
        "duplicate_grade_ids": len(set(duplicate_grade_ids)),
        "conflicting_labels": len(set(conflicting_labels)),
        "complete_examinations_with_both_eye_grades": len(complete_pairs),
        "incomplete_examinations": len(pairing_rows) - len(complete_pairs),
        "complete_pair_grade_counts": dict(
            sorted(Counter(row["left_grade"] for row in complete_pairs).items())
        ),
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "messidor_pair_join_audit.csv", join_rows)
    write_csv(
        output_root / "messidor_complete_pairs.csv",
        complete_pairs,
        fields=["examination_id", "left_image_id", "right_image_id", "left_grade", "right_grade", "gap"],
    )
    write_csv(
        output_root / "messidor_unmatched.csv",
        unmatched_rows,
        fields=["source", "normalized_image_id"],
    )
    save_json(output_root / "messidor_pair_join_summary.json", summary)
    if failures:
        raise RuntimeError("Messidor-2 audit failed: " + "; ".join(failures))
    return summary

