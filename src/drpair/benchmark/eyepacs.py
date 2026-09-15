"""Deterministic patient-level EyePACS Quick20 split construction and audit."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from drpair.data.manifest import write_csv
from drpair.runtime import file_hash, save_json

SPLIT_FRACTIONS = {"train": 0.10, "val": 0.03, "test": 0.07}
SPLIT_ORDER = tuple(SPLIT_FRACTIONS)


@dataclass(frozen=True)
class PatientPair:
    patient_id: str
    left_grade: int
    right_grade: int
    left_image: str
    right_image: str

    @property
    def patient_max_grade(self) -> int:
        return max(self.left_grade, self.right_grade)

    @property
    def gap(self) -> int:
        return abs(self.left_grade - self.right_grade)

    @property
    def gap_bucket(self) -> int:
        return min(self.gap, 2)

    @property
    def stratum(self) -> tuple[int, int]:
        return self.patient_max_grade, self.gap_bucket


def _parse_image_id(image_id: str) -> tuple[str, str]:
    for side in ("left", "right"):
        suffix = f"_{side}"
        if image_id.endswith(suffix) and image_id[: -len(suffix)].isdigit():
            return image_id[: -len(suffix)], side
    raise ValueError(f"Invalid EyePACS image ID: {image_id!r}")


def read_train_labels(path: Path) -> tuple[list[PatientPair], dict[str, int]]:
    """Read the local official trainLabels.csv and require one complete L/R pair per patient."""
    path = Path(path)
    by_patient: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"image", "level"}.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected trainLabels schema: {reader.fieldnames}")
        for row in reader:
            image_id = row["image"].strip()
            try:
                grade = int(row["level"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid grade for {image_id!r}: {row['level']!r}") from error
            if not 0 <= grade <= 4:
                raise ValueError(f"Grade outside 0..4 for {image_id!r}: {grade}")
            if image_id in seen:
                if seen[image_id] != grade:
                    raise ValueError(f"Conflicting duplicate label: {image_id}")
                raise ValueError(f"Duplicate image ID: {image_id}")
            seen[image_id] = grade
            patient_id, side = _parse_image_id(image_id)
            if side in by_patient[patient_id]:
                raise ValueError(f"Duplicate patient side: {patient_id}/{side}")
            by_patient[patient_id][side] = (image_id, grade)

    incomplete = {patient: sorted(eyes) for patient, eyes in by_patient.items() if set(eyes) != {"left", "right"}}
    if incomplete:
        preview = dict(sorted(incomplete.items())[:10])
        raise ValueError(f"Incomplete EyePACS pairs ({len(incomplete)}): {preview}")

    pairs = []
    for patient_id in sorted(by_patient, key=int):
        left, right = by_patient[patient_id]["left"], by_patient[patient_id]["right"]
        pairs.append(PatientPair(patient_id, left[1], right[1], left[0], right[0]))
    return pairs, dict(sorted(Counter(seen.values()).items()))


def _stable_order(pair: PatientPair, seed: int) -> str:
    payload = f"eyepacs-quick20:{seed}:{pair.patient_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def assign_patient_splits(
    pairs: list[PatientPair],
    seed: int = 42,
    fractions: dict[str, float] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Allocate exact global targets while approximating every (max grade, gap bucket) quota."""
    fractions = dict(fractions or SPLIT_FRACTIONS)
    if tuple(fractions) != SPLIT_ORDER:
        raise ValueError(f"Split order must be {SPLIT_ORDER}")
    if any(value <= 0 for value in fractions.values()) or sum(fractions.values()) >= 1:
        raise ValueError("Quick benchmark fractions must be positive and sum to less than one")

    strata: dict[tuple[int, int], list[PatientPair]] = defaultdict(list)
    for pair in pairs:
        strata[pair.stratum].append(pair)
    for stratum, items in strata.items():
        items.sort(key=lambda pair: (_stable_order(pair, seed), int(pair.patient_id)))

    targets = {split: round(len(pairs) * fractions[split]) for split in SPLIT_ORDER}
    allocation = {
        stratum: {split: int(len(items) * fractions[split]) for split in SPLIT_ORDER}
        for stratum, items in strata.items()
    }
    deficits = {
        split: targets[split] - sum(row[split] for row in allocation.values())
        for split in SPLIT_ORDER
    }
    while any(deficit > 0 for deficit in deficits.values()):
        candidates = []
        for stratum, items in strata.items():
            used = sum(allocation[stratum].values())
            if used >= len(items):
                continue
            for split in SPLIT_ORDER:
                if deficits[split] <= 0:
                    continue
                ideal = len(items) * fractions[split]
                shortfall = ideal - allocation[stratum][split]
                candidates.append((shortfall, len(items) - used, split, stratum))
        if not candidates:
            raise RuntimeError(f"Unable to satisfy split targets: {deficits}")
        _, _, split, stratum = max(
            candidates,
            key=lambda item: (item[0], item[1], -SPLIT_ORDER.index(item[2]), item[3]),
        )
        allocation[stratum][split] += 1
        deficits[split] -= 1

    assignment: dict[str, str] = {}
    for stratum in sorted(strata):
        cursor = 0
        for split in SPLIT_ORDER:
            count = allocation[stratum][split]
            for pair in strata[stratum][cursor : cursor + count]:
                assignment[pair.patient_id] = split
            cursor += count
    if Counter(assignment.values()) != Counter(targets):
        raise AssertionError("Final patient counts do not match requested targets")
    return assignment, targets


def _image_path(image_root: Path, grade: int, image_id: str) -> Path:
    return image_root / str(grade) / f"{image_id}.jpeg"


def _patient_row(pair: PatientPair, split: str) -> dict[str, object]:
    return {
        "patient_id": pair.patient_id,
        "left_grade": pair.left_grade,
        "right_grade": pair.right_grade,
        "patient_max_grade": pair.patient_max_grade,
        "gap": pair.gap,
        "gap_bucket": pair.gap_bucket,
        "split": split,
        "included": split != "unused",
    }


def _eye_rows(pair: PatientPair, split: str, image_root: Path, repo_root: Path):
    for target_eye, fellow_eye in (("left", "right"), ("right", "left")):
        target_grade = getattr(pair, f"{target_eye}_grade")
        fellow_grade = getattr(pair, f"{fellow_eye}_grade")
        image_id = getattr(pair, f"{target_eye}_image")
        fellow_id = getattr(pair, f"{fellow_eye}_image")
        path = _image_path(image_root, target_grade, image_id)
        fellow_path = _image_path(image_root, fellow_grade, fellow_id)
        yield {
            "patient_id": pair.patient_id,
            "image_id": image_id,
            "target_eye": target_eye,
            "fellow_eye": fellow_eye,
            "target_grade": target_grade,
            "fellow_grade": fellow_grade,
            "patient_max_grade": pair.patient_max_grade,
            "gap": pair.gap,
            "gap_bucket": pair.gap_bucket,
            "split": split,
            "image_path": path.relative_to(repo_root).as_posix(),
            "fellow_path": fellow_path.relative_to(repo_root).as_posix(),
        }


def _split_summary(patient_rows: list[dict[str, object]], eye_rows: list[dict[str, object]]):
    return {
        "patients": len(patient_rows),
        "images": len(eye_rows),
        "left": sum(row["target_eye"] == "left" for row in eye_rows),
        "right": sum(row["target_eye"] == "right" for row in eye_rows),
        "complete_pairs": len(patient_rows),
        "grade_counts": {
            str(grade): sum(row["target_grade"] == grade for row in eye_rows)
            for grade in range(5)
        },
        "patient_max_grade_counts": {
            str(grade): sum(row["patient_max_grade"] == grade for row in patient_rows)
            for grade in range(5)
        },
        "gap_patient_counts": {
            "gap0": sum(row["gap"] == 0 for row in patient_rows),
            "gap1": sum(row["gap"] == 1 for row in patient_rows),
            "gap_ge2": sum(row["gap"] >= 2 for row in patient_rows),
        },
    }


def _audit_rows(summaries, patient_overlap: int, image_overlap: int):
    rows = []
    for split, summary in summaries.items():
        for metric in ("patients", "images", "left", "right", "complete_pairs"):
            rows.append({"split": split, "metric": metric, "group": "all", "value": summary[metric]})
        for grade, count in summary["grade_counts"].items():
            rows.append({"split": split, "metric": "grade_images", "group": grade, "value": count})
        for grade, count in summary["patient_max_grade_counts"].items():
            rows.append({"split": split, "metric": "patient_max_grade", "group": grade, "value": count})
        for gap, count in summary["gap_patient_counts"].items():
            rows.append({"split": split, "metric": "gap_patients", "group": gap, "value": count})
    rows.extend(
        [
            {"split": "cross_split", "metric": "patient_overlap", "group": "all", "value": patient_overlap},
            {"split": "cross_split", "metric": "image_overlap", "group": "all", "value": image_overlap},
        ]
    )
    return rows


def prepare_quick20(
    labels_path: Path,
    image_root: Path,
    repo_root: Path,
    seed: int = 42,
) -> dict[str, object]:
    labels_path, image_root, repo_root = map(Path, (labels_path, image_root, repo_root))
    repo_root = repo_root.resolve()
    labels_path = labels_path.resolve()
    image_root = image_root.resolve() if image_root.is_absolute() else (repo_root / image_root).resolve()
    pairs, actual_grade_counts = read_train_labels(labels_path)
    assignment, targets = assign_patient_splits(pairs, seed=seed)
    pair_by_id = {pair.patient_id: pair for pair in pairs}

    missing_images = []
    for pair in pairs:
        for side in ("left", "right"):
            path = _image_path(image_root, getattr(pair, f"{side}_grade"), getattr(pair, f"{side}_image"))
            if not path.is_file():
                missing_images.append(str(path))
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} EyePACS images; first={missing_images[0]}")

    split_patient_rows = {}
    split_eye_rows = {}
    for split in SPLIT_ORDER:
        selected = [pair_by_id[patient] for patient, value in assignment.items() if value == split]
        selected.sort(key=lambda pair: int(pair.patient_id))
        split_patient_rows[split] = [_patient_row(pair, split) for pair in selected]
        split_eye_rows[split] = [
            row for pair in selected for row in _eye_rows(pair, split, image_root, repo_root)
        ]

    all_patients = [
        _patient_row(pair, assignment.get(pair.patient_id, "unused"))
        for pair in sorted(pairs, key=lambda item: int(item.patient_id))
    ]
    split_dir = repo_root / "data" / "splits"
    for split in SPLIT_ORDER:
        write_csv(split_dir / f"eyepacs_quick20_{split}.csv", split_eye_rows[split])
    write_csv(split_dir / "eyepacs_quick20_patients.csv", all_patients)

    summaries = {
        split: _split_summary(split_patient_rows[split], split_eye_rows[split])
        for split in SPLIT_ORDER
    }
    patient_sets = {split: {row["patient_id"] for row in rows} for split, rows in split_patient_rows.items()}
    image_sets = {split: {row["image_id"] for row in rows} for split, rows in split_eye_rows.items()}
    patient_overlap = sum(
        len(patient_sets[a] & patient_sets[b])
        for i, a in enumerate(SPLIT_ORDER)
        for b in SPLIT_ORDER[i + 1 :]
    )
    image_overlap = sum(
        len(image_sets[a] & image_sets[b])
        for i, a in enumerate(SPLIT_ORDER)
        for b in SPLIT_ORDER[i + 1 :]
    )
    failures = []
    if patient_overlap:
        failures.append(f"patient leakage={patient_overlap}")
    if image_overlap:
        failures.append(f"image leakage={image_overlap}")
    for split, summary in summaries.items():
        if summary["images"] != 2 * summary["patients"] or summary["complete_pairs"] != summary["patients"]:
            failures.append(f"{split} is not composed exclusively of complete pairs")
        absent = [grade for grade, count in summary["grade_counts"].items() if count == 0]
        if absent:
            failures.append(f"{split} lacks image grades {absent}")
    test_gap_ge2_eyes = 2 * summaries["test"]["gap_patient_counts"]["gap_ge2"]
    if test_gap_ge2_eyes < 30:
        failures.append(f"test gap>=2 support is only {test_gap_ge2_eyes} target eyes")

    summary = {
        "phase": "PHASE 1 - EyePACS data audit",
        "status": "passed" if not failures else "failed",
        "seed": seed,
        "labels_source": str(labels_path.resolve()),
        "labels_sha256": file_hash(labels_path),
        "labels_schema": ["image", "level"],
        "actual_local_total": {
            "patients": len(pairs),
            "images": 2 * len(pairs),
            "complete_pairs": len(pairs),
            "grade_counts": {str(key): value for key, value in actual_grade_counts.items()},
        },
        "requested_fractions_of_full_patients": SPLIT_FRACTIONS,
        "target_patient_counts": targets,
        "selected_patient_count": len(assignment),
        "selected_image_count": 2 * len(assignment),
        "splits": summaries,
        "patient_overlap": patient_overlap,
        "image_overlap": image_overlap,
        "test_subgroup_support": {
            "gap0_target_eyes": 2 * summaries["test"]["gap_patient_counts"]["gap0"],
            "gap1_target_eyes": 2 * summaries["test"]["gap_patient_counts"]["gap1"],
            "gap_ge2_target_eyes": test_gap_ge2_eyes,
            "qwk_minimum_target_eyes": 30,
        },
        "hard_gate_passed": not failures,
        "failures": failures,
    }
    audit_dir = repo_root / "results" / "data_audit"
    write_csv(
        audit_dir / "eyepacs_split_audit.csv",
        _audit_rows(summaries, patient_overlap, image_overlap),
    )
    save_json(audit_dir / "eyepacs_split_summary.json", summary)
    if failures:
        raise RuntimeError("EyePACS hard gate failed: " + "; ".join(failures))
    return summary
