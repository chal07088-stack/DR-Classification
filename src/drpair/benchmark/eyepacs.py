"""Deterministic patient-level EyePACS Quick20 split construction and audit."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from drpair.data.manifest import write_csv
from drpair.runtime import file_hash, save_json

SPLIT_FRACTIONS = {"train": 0.10, "val": 0.03, "test": 0.07}
SPLIT_ORDER = tuple(SPLIT_FRACTIONS)
SPLIT_VERSION = "quick20_symmetry_preserved_v2"
ACTIVE_SPLIT_FILES = {
    split: f"data/splits/eyepacs_quick20_symmetry_{split}.csv" for split in SPLIT_ORDER
}
ACTIVE_PATIENT_FILE = "data/splits/eyepacs_quick20_symmetry_patients.csv"


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
    def patient_min_grade(self) -> int:
        return min(self.left_grade, self.right_grade)

    @property
    def gap(self) -> int:
        return abs(self.left_grade - self.right_grade)

    @property
    def gap_bucket(self) -> int:
        return min(self.gap, 2)

    @property
    def symmetry_group(self) -> str:
        return "symmetric" if self.gap == 0 else "asymmetric"

    @property
    def secondary_stratum(self) -> tuple[int, int, int, int]:
        # Directional grades preserve left/right grade distributions as a
        # secondary goal without ever changing the primary symmetry quota.
        return self.gap_bucket, self.patient_max_grade, self.left_grade, self.right_grade


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


def _patient_sort_key(pair: PatientPair) -> tuple[int, int | str]:
    return (0, int(pair.patient_id)) if pair.patient_id.isdigit() else (1, pair.patient_id)


def assign_patient_splits(
    pairs: list[PatientPair],
    seed: int = 42,
    fractions: dict[str, float] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Allocate exact totals with binary symmetry as the hard primary stratum."""
    fractions = dict(fractions or SPLIT_FRACTIONS)
    split_order = tuple(fractions)
    if not split_order:
        raise ValueError("At least one split fraction is required")
    fraction_sum = sum(fractions.values())
    if any(value <= 0 for value in fractions.values()) or fraction_sum > 1 + 1e-12:
        raise ValueError("Split fractions must be positive and sum to at most one")

    targets = {split: round(len(pairs) * fractions[split]) for split in split_order}
    if abs(fraction_sum - 1) <= 1e-12:
        difference = len(pairs) - sum(targets.values())
        while difference:
            direction = 1 if difference > 0 else -1
            candidates = []
            for split in split_order:
                current = targets[split]
                proposed = current + direction
                if not 0 <= proposed <= len(pairs):
                    continue
                ideal = len(pairs) * fractions[split]
                cost = abs(proposed - ideal) - abs(current - ideal)
                candidates.append((cost, split_order.index(split), split))
            if not candidates:
                raise RuntimeError("Unable to apportion exact split totals")
            split = min(candidates)[2]
            targets[split] += direction
            difference -= direction
    symmetric_total = sum(pair.symmetry_group == "symmetric" for pair in pairs)
    ideal_symmetric = {
        split: targets[split] * symmetric_total / len(pairs) for split in split_order
    }
    symmetric_targets = {split: round(ideal_symmetric[split]) for split in split_order}
    desired_combined = round(sum(targets.values()) * symmetric_total / len(pairs))
    difference = desired_combined - sum(symmetric_targets.values())
    while difference:
        direction = 1 if difference > 0 else -1
        candidates = []
        for split in split_order:
            current = symmetric_targets[split]
            proposed = current + direction
            if not 0 <= proposed <= targets[split]:
                continue
            cost = abs(proposed - ideal_symmetric[split]) - abs(current - ideal_symmetric[split])
            candidates.append((cost, split_order.index(split), split))
        if not candidates:
            raise RuntimeError("Unable to apportion the primary symmetry quota")
        split = min(candidates)[2]
        symmetric_targets[split] += direction
        difference -= direction

    group_targets = {
        "symmetric": symmetric_targets,
        "asymmetric": {
            split: targets[split] - symmetric_targets[split] for split in split_order
        },
    }
    assignment: dict[str, str] = {}
    for group in ("symmetric", "asymmetric"):
        group_items = [pair for pair in pairs if pair.symmetry_group == group]
        secondary: dict[tuple[int, int, int, int], list[PatientPair]] = defaultdict(list)
        for pair in group_items:
            secondary[pair.secondary_stratum].append(pair)
        for items in secondary.values():
            items.sort(key=lambda pair: (_stable_order(pair, seed), _patient_sort_key(pair)))
        allocation = {
            stratum: {
                split: int(len(items) * group_targets[group][split] / len(group_items))
                for split in split_order
            }
            for stratum, items in secondary.items()
        }
        deficits = {
            split: group_targets[group][split]
            - sum(row[split] for row in allocation.values())
            for split in split_order
        }
        while any(value > 0 for value in deficits.values()):
            candidates = []
            for stratum, items in secondary.items():
                used = sum(allocation[stratum].values())
                if used >= len(items):
                    continue
                for split in split_order:
                    if deficits[split] <= 0:
                        continue
                    ideal = len(items) * group_targets[group][split] / len(group_items)
                    shortfall = ideal - allocation[stratum][split]
                    candidates.append((shortfall, len(items) - used, split, stratum))
            if not candidates:
                raise RuntimeError(f"Unable to satisfy {group} quotas: {deficits}")
            _, _, split, stratum = max(
                candidates,
                key=lambda item: (item[0], item[1], -split_order.index(item[2]), item[3]),
            )
            allocation[stratum][split] += 1
            deficits[split] -= 1
        for stratum in sorted(secondary):
            cursor = 0
            for split in split_order:
                count = allocation[stratum][split]
                for pair in secondary[stratum][cursor : cursor + count]:
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
        "symmetry_group": pair.symmetry_group,
        "patient_max_grade": pair.patient_max_grade,
        "patient_min_grade": pair.patient_min_grade,
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
            "patient_min_grade": pair.patient_min_grade,
            "symmetry_group": pair.symmetry_group,
            "gap": pair.gap,
            "gap_bucket": pair.gap_bucket,
            "split": split,
            "image_path": path.relative_to(repo_root).as_posix(),
            "fellow_path": fellow_path.relative_to(repo_root).as_posix(),
        }


def _split_summary(patient_rows: list[dict[str, object]], eye_rows: list[dict[str, object]]):
    patients = len(patient_rows)
    symmetric = sum(row["symmetry_group"] == "symmetric" for row in patient_rows)
    asymmetric = patients - symmetric
    return {
        "patients": len(patient_rows),
        "images": len(eye_rows),
        "left": sum(row["target_eye"] == "left" for row in eye_rows),
        "right": sum(row["target_eye"] == "right" for row in eye_rows),
        "complete_pairs": len(patient_rows),
        "symmetric_patients": symmetric,
        "asymmetric_patients": asymmetric,
        "symmetric_rate": symmetric / patients if patients else 0.0,
        "asymmetric_rate": asymmetric / patients if patients else 0.0,
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
        for metric in (
            "patients", "images", "left", "right", "complete_pairs",
            "symmetric_patients", "asymmetric_patients", "symmetric_rate", "asymmetric_rate",
        ):
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


def _symmetry_audit_rows(summaries: dict[str, dict[str, object]]):
    full = summaries["full"]
    rows = []
    labels = {"full": "Full EyePACS complete-pair population", "train": "Train",
              "val": "Validation", "test": "Test", "quick20": "Combined Quick20"}
    for key in ("full", "train", "val", "test", "quick20"):
        value = summaries[key]
        rows.append({
            "split": labels[key], "n_patients": value["patients"],
            "symmetric_n": value["symmetric_patients"],
            "symmetric_percent": 100 * value["symmetric_rate"],
            "asymmetric_n": value["asymmetric_patients"],
            "asymmetric_percent": 100 * value["asymmetric_rate"],
            "gap1_n": value["gap_patient_counts"]["gap1"],
            "gap_ge2_n": value["gap_patient_counts"]["gap_ge2"],
            "delta_symmetric_percentage_points": 100 * (value["symmetric_rate"] - full["symmetric_rate"]),
            "delta_asymmetric_percentage_points": 100 * (value["asymmetric_rate"] - full["asymmetric_rate"]),
        })
    return rows


def _plot_distribution(path: Path, title: str, labels: list[str], full: list[float], quick: list[float]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - width / 2, full, width, label="Full")
    axis.bar(x + width / 2, quick, width, label="Quick20")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Percent")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


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
        selected.sort(key=_patient_sort_key)
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
        write_csv(
            split_dir / f"eyepacs_quick20_symmetry_{split}.csv",
            split_eye_rows[split],
        )
    write_csv(split_dir / "eyepacs_quick20_symmetry_patients.csv", all_patients)

    split_summaries = {
        split: _split_summary(split_patient_rows[split], split_eye_rows[split])
        for split in SPLIT_ORDER
    }
    full_eye_rows = [
        row for pair in pairs for row in _eye_rows(pair, "full", image_root, repo_root)
    ]
    full_patient_rows = [_patient_row(pair, "full") for pair in pairs]
    quick_patient_rows = [row for split in SPLIT_ORDER for row in split_patient_rows[split]]
    quick_eye_rows = [row for split in SPLIT_ORDER for row in split_eye_rows[split]]
    summaries = {
        "full": _split_summary(full_patient_rows, full_eye_rows),
        **split_summaries,
        "quick20": _split_summary(quick_patient_rows, quick_eye_rows),
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
    full_symmetric_rate = summaries["full"]["symmetric_rate"]
    for split in SPLIT_ORDER:
        summary = summaries[split]
        if summary["images"] != 2 * summary["patients"] or summary["complete_pairs"] != summary["patients"]:
            failures.append(f"{split} is not composed exclusively of complete pairs")
        absent = [grade for grade, count in summary["grade_counts"].items() if count == 0]
        if absent:
            failures.append(f"{split} lacks image grades {absent}")
        # One patient is the largest unavoidable integer error in a split.
        ratio_tolerance = 1 / summary["patients"] + 1e-12
        if abs(summary["symmetric_rate"] - full_symmetric_rate) > ratio_tolerance:
            failures.append(f"{split} symmetry ratio exceeds integer-rounding tolerance")
    quick_tolerance = 1 / summaries["quick20"]["patients"] + 1e-12
    if abs(summaries["quick20"]["symmetric_rate"] - full_symmetric_rate) > quick_tolerance:
        failures.append("combined Quick20 symmetry ratio exceeds integer-rounding tolerance")
    if summaries["quick20"]["patients"] != sum(targets.values()):
        failures.append("combined Quick20 patient count does not match the apportioned target")
    test_gap_ge2_eyes = 2 * summaries["test"]["gap_patient_counts"]["gap_ge2"]
    if test_gap_ge2_eyes < 30:
        failures.append(f"test gap>=2 support is only {test_gap_ge2_eyes} target eyes")

    summary = {
        "phase": "PHASE 1 - EyePACS data audit",
        "status": "passed" if not failures else "failed",
        "split_version": SPLIT_VERSION,
        "sampling_method": "symmetry-preserving patient-level stratified sample",
        "seed": seed,
        "labels_source": str(labels_path.resolve()),
        "labels_sha256": file_hash(labels_path),
        "labels_schema": ["image", "level"],
        "actual_local_total": {
            "patients": len(pairs),
            "images": 2 * len(pairs),
            "complete_pairs": len(pairs),
            "grade_counts": {str(key): value for key, value in actual_grade_counts.items()},
            "symmetric_patients": summaries["full"]["symmetric_patients"],
            "asymmetric_patients": summaries["full"]["asymmetric_patients"],
            "symmetric_rate": summaries["full"]["symmetric_rate"],
            "asymmetric_rate": summaries["full"]["asymmetric_rate"],
            "gap_patient_counts": summaries["full"]["gap_patient_counts"],
        },
        "requested_fractions_of_full_patients": SPLIT_FRACTIONS,
        "active_split_files": {
            key: {"path": value, "sha256": file_hash(repo_root / value)}
            for key, value in {**ACTIVE_SPLIT_FILES, "patients": ACTIVE_PATIENT_FILE}.items()
        },
        "target_patient_counts": targets,
        "selected_patient_count": len(assignment),
        "selected_image_count": 2 * len(assignment),
        "splits": {key: summaries[key] for key in (*SPLIT_ORDER, "quick20")},
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
    write_csv(audit_dir / "eyepacs_full_pair_population.csv", full_patient_rows)
    save_json(
        audit_dir / "eyepacs_full_symmetry_summary.json",
        {
            "split_version": SPLIT_VERSION,
            "labels_source": str(labels_path),
            "labels_sha256": file_hash(labels_path),
            **summaries["full"],
        },
    )
    write_csv(
        audit_dir / "eyepacs_symmetry_sampling_audit_long.csv",
        _audit_rows(summaries, patient_overlap, image_overlap),
    )
    symmetry_rows = _symmetry_audit_rows(summaries)
    write_csv(audit_dir / "eyepacs_symmetry_sampling_audit.csv", symmetry_rows)
    save_json(audit_dir / "eyepacs_symmetry_sampling_summary.json", summary)

    figures = repo_root / "results" / "figures"
    full = summaries["full"]
    quick = summaries["quick20"]
    _plot_distribution(
        figures / "eyepacs_full_vs_quick20_symmetry.png",
        "EyePACS symmetry distribution: Full vs Quick20",
        ["Symmetric", "Asymmetric"],
        [100 * full["symmetric_rate"], 100 * full["asymmetric_rate"]],
        [100 * quick["symmetric_rate"], 100 * quick["asymmetric_rate"]],
    )
    gap_keys = ["gap0", "gap1", "gap_ge2"]
    _plot_distribution(
        figures / "eyepacs_full_vs_quick20_gap.png",
        "EyePACS patient gap distribution: Full vs Quick20",
        ["gap=0", "gap=1", "gap>=2"],
        [100 * full["gap_patient_counts"][key] / full["patients"] for key in gap_keys],
        [100 * quick["gap_patient_counts"][key] / quick["patients"] for key in gap_keys],
    )
    grade_keys = [str(value) for value in range(5)]
    _plot_distribution(
        figures / "eyepacs_full_vs_quick20_grade.png",
        "EyePACS target-eye grade distribution: Full vs Quick20",
        grade_keys,
        [100 * full["grade_counts"][key] / full["images"] for key in grade_keys],
        [100 * quick["grade_counts"][key] / quick["images"] for key in grade_keys],
    )

    report_lines = [
        "# PHASE 1 — Data Audit",
        "",
        f"Split version: `{SPLIT_VERSION}`",
        "",
        "EyePACS Quick20 was generated using patient-level stratified random sampling that preserves the Symmetric (`|L-R|=0`) vs Asymmetric (`|L-R|>=1`) ratio of the complete-pair EyePACS population.",
        "",
        "| Split | Patients | Symmetric N | Symmetric % | Asymmetric N | Asymmetric % | Gap=1 N | Gap>=2 N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in symmetry_rows:
        report_lines.append(
            f"| {row['split']} | {row['n_patients']} | {row['symmetric_n']} | "
            f"{row['symmetric_percent']:.4f} | {row['asymmetric_n']} | "
            f"{row['asymmetric_percent']:.4f} | {row['gap1_n']} | {row['gap_ge2_n']} |"
        )
    report_lines.extend([
        "", f"- Patient leakage: **{patient_overlap}**",
        f"- Image leakage: **{image_overlap}**",
        f"- EyePACS hard gate: **{'PASS' if not failures else 'FAIL'}**",
        "", "## Messidor-2 exact-ID audit", "",
    ])
    messidor_path = audit_dir / "messidor_pair_join_summary.json"
    if messidor_path.is_file():
        messidor = json.loads(messidor_path.read_text(encoding="utf-8"))
        report_lines.extend([
            f"- Status: **{str(messidor.get('status', 'missing')).upper()}**",
            f"- Grade rows: {messidor.get('grade_annotation_rows', 'NA')}",
            f"- Official examinations: {messidor.get('official_examinations', 'NA')}",
            f"- Complete audited pairs: {messidor.get('complete_examinations_with_both_eye_grades', 'NA')}",
            f"- Reason: {messidor.get('reason', 'Exact-ID audit completed.')}",
        ])
    else:
        report_lines.append("- Status: **MISSING**")
    report_lines.extend([
        "",
        "No model training is permitted until both EyePACS and Messidor-2 audits pass.",
        "",
    ])
    (repo_root / "results" / "PHASE1_DATA_AUDIT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    if failures:
        raise RuntimeError("EyePACS hard gate failed: " + "; ".join(failures))
    return summary


def require_phase1_hard_gate(
    repo_root: Path, *, allow_eyepacs_only_provisional: bool = False
) -> dict[str, object]:
    """Validate Phase 1, allowing an explicitly labeled EyePACS-only pilot."""
    repo_root = Path(repo_root).resolve()
    audit_root = repo_root / "results" / "data_audit"
    eye_path = audit_root / "eyepacs_symmetry_sampling_summary.json"
    messidor_path = audit_root / "messidor_pair_join_summary.json"
    if not eye_path.is_file():
        raise RuntimeError("PHASE 1 hard gate: EyePACS audit summary is missing")
    eye = json.loads(eye_path.read_text(encoding="utf-8"))
    if (
        eye.get("split_version") != SPLIT_VERSION
        or eye.get("status") != "passed"
        or not eye.get("hard_gate_passed")
    ):
        raise RuntimeError("PHASE 1 hard gate: EyePACS symmetry-preserved v2 audit is not PASS")
    active_files = eye.get("active_split_files", {})
    if set(active_files) != {*ACTIVE_SPLIT_FILES, "patients"}:
        raise RuntimeError("PHASE 1 hard gate: active split inventory is incomplete")
    for item in active_files.values():
        path = repo_root / item["path"]
        if not path.is_file() or file_hash(path) != item["sha256"]:
            raise RuntimeError(f"PHASE 1 hard gate: active split changed: {path}")
    if allow_eyepacs_only_provisional:
        return {
            "eyepacs": eye,
            "scope": "EYEPACS_ONLY_PROVISIONAL_PENDING_MESSIDOR",
        }
    if not messidor_path.is_file():
        raise RuntimeError("PHASE 1 hard gate: Messidor exact-ID audit summary is missing")
    messidor = json.loads(messidor_path.read_text(encoding="utf-8"))
    if messidor.get("status") != "passed":
        reason = messidor.get("reason", "Messidor exact-ID audit is not PASS")
        raise RuntimeError(f"PHASE 1 hard gate: {reason}")
    if not messidor.get("complete_examinations_with_both_eye_grades"):
        raise RuntimeError("PHASE 1 hard gate: Messidor has no audited complete bilateral pairs")
    messidor_split_path = audit_root / "messidor_split_summary.json"
    if not messidor_split_path.is_file():
        raise RuntimeError("PHASE 1 hard gate: Messidor in-domain split audit is missing")
    messidor_split = json.loads(messidor_split_path.read_text(encoding="utf-8"))
    if messidor_split.get("status") != "passed" or not messidor_split.get("hard_gate_passed"):
        raise RuntimeError("PHASE 1 hard gate: Messidor in-domain split audit is not PASS")
    return {"eyepacs": eye, "messidor": messidor, "messidor_split": messidor_split}
