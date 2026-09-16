"""Exact-ID Messidor-2 pairing and adjudicated-grade provenance audit."""

from __future__ import annotations

import csv
import unicodedata
from collections import Counter
from pathlib import Path, PurePath

from drpair.data.manifest import write_csv
from drpair.runtime import file_hash, save_json

from .eyepacs import (
    PatientPair,
    _audit_rows,
    _patient_sort_key,
    _split_summary,
    assign_patient_splits,
)

MESSIDOR_SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
MESSIDOR_SPLIT_VERSION = "messidor_complete_pairs_70_15_15_v1"


def normalize_image_id(value: str) -> str:
    """Exact filename key with only basename, case, and image-extension normalization."""
    text = unicodedata.normalize("NFKC", str(value).strip()).replace("\\", "/")
    name = PurePath(text).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid image ID: {value!r}")
    suffix = PurePath(name).suffix.casefold()
    if suffix in {".png", ".jpg", ".jpeg"}:
        name = name[: -len(suffix)]
    return name.casefold()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        fieldnames = [field.strip() for field in reader.fieldnames]
        if len(set(fieldnames)) != len(fieldnames):
            raise ValueError(f"Duplicate CSV headers after whitespace normalization: {path}")
        rows = []
        for raw_row in reader:
            rows.append(
                {
                    key.strip(): value.strip() if isinstance(value, str) else value
                    for key, value in raw_row.items()
                    if key is not None
                }
            )
        return rows, fieldnames


def audit_messidor_pairing(
    pairing_path: Path,
    grade_path: Path,
    output_root: Path,
    left_column: str,
    right_column: str,
    examination_column: str | None = None,
    image_root: Path | None = None,
    pairing_source: str = "https://www.adcis.net/en/third-party/messidor2/",
    grade_source: str = "https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades",
) -> dict[str, object]:
    pairing_path, grade_path, output_root = map(Path, (pairing_path, grade_path, output_root))
    image_root = Path(image_root) if image_root is not None else None
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
    local_image_paths: dict[str, Path] = {}
    duplicate_local_ids = []
    if image_root is not None:
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing local Messidor image directory: {image_root}")
        for image_path in image_root.rglob("*"):
            if not image_path.is_file():
                continue
            normalized = normalize_image_id(image_path.name)
            if normalized in local_image_paths:
                duplicate_local_ids.append(normalized)
            else:
                local_image_paths[normalized] = image_path
    local_set = set(local_image_paths)
    unmatched_local_images = sorted(local_set - grade_set)
    unmatched_label_ids = sorted(grade_set - local_set) if image_root is not None else unmatched_grades
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
    if image_root is not None:
        if len(local_image_paths) != 1748:
            failures.append(f"local image files={len(local_image_paths)}, expected 1748")
        if duplicate_local_ids:
            failures.append(f"duplicate local image IDs={len(set(duplicate_local_ids))}")
        if unmatched_local_images:
            failures.append(f"unmatched local image files={len(unmatched_local_images)}")
        if unmatched_label_ids:
            failures.append(f"unmatched label IDs={len(unmatched_label_ids)}")

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
        "local_image_count": len(local_image_paths) if image_root is not None else None,
        "gradable_annotation_rows": gradable_rows,
        "exact_matched_images": exact_matched,
        "unmatched_official_images": len(unmatched_official),
        "unmatched_grade_rows": len(unmatched_grades),
        "duplicate_official_ids": len(set(duplicate_official_ids)),
        "duplicate_grade_ids": len(set(duplicate_grade_ids)),
        "conflicting_labels": len(set(conflicting_labels)),
        "duplicate_local_image_ids": len(set(duplicate_local_ids)),
        "unmatched_local_image_filenames": [local_image_paths[key].name for key in unmatched_local_images],
        "unmatched_label_ids": unmatched_label_ids,
        "adjudicated_dr_grade_distribution": dict(
            sorted(Counter(row["adjudicated_dr_grade"].strip() for row in grade_rows).items())
        ),
        "adjudicated_gradable_distribution": dict(
            sorted(Counter(row["adjudicated_gradable"].strip() for row in grade_rows).items())
        ),
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
    write_csv(
        output_root / "messidor_local_image_audit.csv",
        [
            {"category": "unmatched_local_image", "value": local_image_paths[key].name}
            for key in unmatched_local_images
        ]
        + [{"category": "unmatched_label_id", "value": key} for key in unmatched_label_ids]
        + [{"category": "duplicate_local_image_id", "value": key} for key in duplicate_local_ids],
        fields=["category", "value"],
    )
    save_json(output_root / "messidor_pair_join_summary.json", summary)
    if failures:
        raise RuntimeError("Messidor-2 audit failed: " + "; ".join(failures))
    return summary


def _path_for_split(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _image_index(image_root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    duplicate = []
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        key = normalize_image_id(path.name)
        if key in indexed:
            duplicate.append(key)
        else:
            indexed[key] = path
    if duplicate:
        raise ValueError(f"Duplicate normalized Messidor image IDs: {sorted(set(duplicate))[:10]}")
    if not indexed:
        raise FileNotFoundError(f"No image files found under {image_root}")
    return indexed


def prepare_messidor_splits(
    complete_pairs_path: Path,
    image_root: Path,
    repo_root: Path,
    seed: int = 42,
) -> dict[str, object]:
    """Create the all-pairs 70/15/15 Messidor in-domain split after exact-ID audit."""
    complete_pairs_path, image_root, repo_root = map(Path, (complete_pairs_path, image_root, repo_root))
    if not complete_pairs_path.is_file():
        raise FileNotFoundError(f"Missing audited complete-pair table: {complete_pairs_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing Messidor image root: {image_root}")
    with complete_pairs_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"examination_id", "left_image_id", "right_image_id", "left_grade", "right_grade"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected complete-pair schema: {list(rows[0]) if rows else []}")
    image_by_id = _image_index(image_root)
    pairs = []
    original_ids = {}
    for row in rows:
        patient_id = row["examination_id"].strip()
        if not patient_id or patient_id in original_ids:
            raise ValueError(f"Duplicate or empty Messidor examination ID: {patient_id!r}")
        left_id, right_id = row["left_image_id"].strip(), row["right_image_id"].strip()
        left_key, right_key = normalize_image_id(left_id), normalize_image_id(right_id)
        if left_key not in image_by_id or right_key not in image_by_id:
            missing = [key for key in (left_key, right_key) if key not in image_by_id]
            raise FileNotFoundError(f"Missing Messidor paired images for {patient_id}: {missing}")
        pairs.append(
            PatientPair(patient_id, int(row["left_grade"]), int(row["right_grade"]), left_key, right_key)
        )
        original_ids[patient_id] = (left_id, right_id)
    assignment, targets = assign_patient_splits(pairs, seed=seed, fractions=MESSIDOR_SPLIT_FRACTIONS)
    if len(assignment) != len(pairs):
        raise AssertionError("All Messidor complete pairs must be assigned to 70/15/15")
    by_patient = {pair.patient_id: pair for pair in pairs}
    split_patients = {
        split: [by_patient[patient] for patient, value in assignment.items() if value == split]
        for split in MESSIDOR_SPLIT_FRACTIONS
    }
    for values in split_patients.values():
        values.sort(key=_patient_sort_key)

    def patient_row(pair: PatientPair, split: str):
        return {
            "patient_id": pair.patient_id,
            "left_image_id": original_ids[pair.patient_id][0],
            "right_image_id": original_ids[pair.patient_id][1],
            "left_grade": pair.left_grade,
            "right_grade": pair.right_grade,
            "symmetry_group": pair.symmetry_group,
            "patient_max_grade": pair.patient_max_grade,
            "patient_min_grade": pair.patient_min_grade,
            "gap": pair.gap,
            "gap_bucket": pair.gap_bucket,
            "split": split,
        }

    def eye_rows(pair: PatientPair, split: str):
        for target_eye, fellow_eye in (("left", "right"), ("right", "left")):
            target_id = getattr(pair, f"{target_eye}_image")
            fellow_id = getattr(pair, f"{fellow_eye}_image")
            yield {
                "patient_id": pair.patient_id,
                "image_id": original_ids[pair.patient_id][0 if target_eye == "left" else 1],
                "target_eye": target_eye,
                "fellow_eye": fellow_eye,
                "target_grade": getattr(pair, f"{target_eye}_grade"),
                "fellow_grade": getattr(pair, f"{fellow_eye}_grade"),
                "patient_max_grade": pair.patient_max_grade,
                "patient_min_grade": pair.patient_min_grade,
                "symmetry_group": pair.symmetry_group,
                "gap": pair.gap,
                "gap_bucket": pair.gap_bucket,
                "split": split,
                "image_path": _path_for_split(image_by_id[target_id], repo_root),
                "fellow_path": _path_for_split(image_by_id[fellow_id], repo_root),
            }

    patient_rows = {
        split: [patient_row(pair, split) for pair in values]
        for split, values in split_patients.items()
    }
    eye_rows_by_split = {
        split: [row for pair in values for row in eye_rows(pair, split)]
        for split, values in split_patients.items()
    }
    full_patient_rows = [patient_row(pair, "full") for pair in sorted(pairs, key=_patient_sort_key)]
    full_eye_rows = [row for pair in pairs for row in eye_rows(pair, "full")]
    summaries = {
        "full": _split_summary(full_patient_rows, full_eye_rows),
        **{
            split: _split_summary(patient_rows[split], eye_rows_by_split[split])
            for split in MESSIDOR_SPLIT_FRACTIONS
        },
    }
    patient_sets = {split: {row["patient_id"] for row in values} for split, values in patient_rows.items()}
    image_sets = {split: {row["image_id"] for row in values} for split, values in eye_rows_by_split.items()}
    split_order = tuple(MESSIDOR_SPLIT_FRACTIONS)
    patient_overlap = sum(
        len(patient_sets[left] & patient_sets[right])
        for index, left in enumerate(split_order)
        for right in split_order[index + 1 :]
    )
    image_overlap = sum(
        len(image_sets[left] & image_sets[right])
        for index, left in enumerate(split_order)
        for right in split_order[index + 1 :]
    )
    full_rate = summaries["full"]["symmetric_rate"]
    failures = []
    if patient_overlap:
        failures.append(f"patient leakage={patient_overlap}")
    if image_overlap:
        failures.append(f"image leakage={image_overlap}")
    for split in split_order:
        summary = summaries[split]
        if summary["images"] != 2 * summary["patients"]:
            failures.append(f"{split} does not contain exactly two images per patient")
        if abs(summary["symmetric_rate"] - full_rate) > 1 / summary["patients"] + 1e-12:
            failures.append(f"{split} symmetry ratio exceeds integer-rounding tolerance")
    split_dir = repo_root / "data" / "splits"
    for split in split_order:
        write_csv(split_dir / f"messidor_{split}.csv", eye_rows_by_split[split])
    write_csv(split_dir / "messidor_patients.csv", full_patient_rows)
    audit_root = repo_root / "results" / "data_audit"
    write_csv(audit_root / "messidor_split_audit_long.csv", _audit_rows(summaries, patient_overlap, image_overlap))
    summary = {
        "phase": "PHASE 1 - Messidor-2 in-domain split audit",
        "status": "passed" if not failures else "failed",
        "split_version": MESSIDOR_SPLIT_VERSION,
        "seed": seed,
        "complete_pair_source": str(complete_pairs_path.resolve()),
        "complete_pair_sha256": file_hash(complete_pairs_path),
        "image_root": str(image_root.resolve()),
        "target_fractions": MESSIDOR_SPLIT_FRACTIONS,
        "target_patient_counts": targets,
        "splits": summaries,
        "patient_overlap": patient_overlap,
        "image_overlap": image_overlap,
        "hard_gate_passed": not failures,
        "failures": failures,
    }
    save_json(audit_root / "messidor_split_summary.json", summary)
    if failures:
        raise RuntimeError("Messidor split hard gate failed: " + "; ".join(failures))
    return summary
