import csv
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from drpair.runtime import file_hash, save_json

EYE = re.compile(r"^(\d+)_(left|right)\.jpeg$", re.IGNORECASE)
EXPECTED = {"train": 35126, "val": 10906, "test": 42670}
FIELDS = ["patient", "side", "label", "fellow_label", "split", "path", "fellow_path"]


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["label"] = int(row["label"])
        row["fellow_label"] = int(row["fellow_label"])
    validate_manifest(rows)
    return rows


def validate_manifest(rows, root=None, official=False):
    if not rows:
        raise ValueError("Empty manifest")
    keys, patient_splits = {}, defaultdict(set)
    for row in rows:
        key = (str(row["patient"]), row["side"])
        if key in keys:
            raise ValueError(f"Duplicate patient/eye: {key}")
        if row["side"] not in ("left", "right") or not 0 <= row["label"] <= 4:
            raise ValueError(f"Invalid eye/grade: {key}")
        if row["split"] not in EXPECTED:
            raise ValueError(f"Invalid split: {row['split']}")
        keys[key] = row
        patient_splits[key[0]].add(row["split"])
        for field in ("path", "fellow_path"):
            p = Path(row[field])
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Manifest paths must be relative: {p}")
            if root is not None and not (Path(root) / p).is_file():
                raise FileNotFoundError(Path(root) / p)
    if any(len(s) != 1 for s in patient_splits.values()):
        raise ValueError("Patient leakage across train/val/test")
    for (patient, side), row in keys.items():
        other = keys.get((patient, "right" if side == "left" else "left"))
        if other is None:
            raise ValueError(f"Unpaired patient: {patient}")
        if (
            row["fellow_path"] != other["path"]
            or row["fellow_label"] != other["label"]
            or row["split"] != other["split"]
        ):
            raise ValueError(f"Fellow-eye mismatch: {patient}")
    counts = dict(Counter(r["split"] for r in rows))
    if official and counts != EXPECTED:
        raise ValueError(f"Official counts differ: {counts} != {EXPECTED}")
    return {
        split: {
            "eyes": counts.get(split, 0),
            "patients": len({r["patient"] for r in rows if r["split"] == split}),
            "grades": dict(Counter(str(r["label"]) for r in rows if r["split"] == split)),
        }
        for split in EXPECTED
    }


def content_audit_status(repo_root=None, manifest=None):
    """Reflect the committed scripts/audit_data.py result instead of hardcoding False."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    path = root / "experimental_results" / "repository_validation" / "data_content_audit.json"
    if not path.is_file():
        return {"image_content_duplicates_checked": False}
    audit = json.loads(path.read_text(encoding="utf-8"))
    if manifest is not None and audit.get("manifest_sha256") != file_hash(manifest):
        return {"image_content_duplicates_checked": False}
    return {
        "image_content_duplicates_checked": True,
        "image_content_audit": {
            "source": path.relative_to(root).as_posix(),
            **{
                key: audit.get(key)
                for key in (
                    "images",
                    "unique_content_hashes",
                    "byte_duplicate_groups",
                    "cross_split_duplicate_groups",
                    "conflicting_label_groups",
                    "near_duplicate_images_checked",
                )
            },
        },
    }


def patient_folds(rows, folds=5, seed=42):
    patients = defaultdict(list)
    for row in rows:
        patients[str(row["patient"])].append(row["label"])
    ids = sorted(patients, key=int)
    if len(ids) < folds:
        raise ValueError("Insufficient patients for folds")
    strata = np.array([max(patients[p]) for p in ids])
    # Rare grades share a stratum; the original grades remain unchanged.
    counts = Counter(strata)
    if any(n < folds for n in counts.values()):
        warnings.warn(
            f"Merging rare patient-grade strata with fewer than {folds} patients: {dict(counts)}",
            RuntimeWarning,
            stacklevel=2,
        )
    strata = np.array([x if counts[x] >= folds else -1 for x in strata])
    if np.any(np.array(list(Counter(strata).values())) < folds):
        warnings.warn(
            f"Patient-fold stratification unavailable after merging; using unstratified "
            f"{folds}-fold assignment over {len(ids)} patients.",
            RuntimeWarning,
            stacklevel=2,
        )
        strata[:] = 0
    assignment = {}
    for fold, (_, held) in enumerate(
        StratifiedKFold(folds, shuffle=True, random_state=seed).split(ids, strata)
    ):
        assignment.update({ids[i]: fold for i in held})
    return np.array([assignment[str(r["patient"])] for r in rows], dtype=np.int64)


def read_fold_assignment(path, rows, split, seed=42, folds=5):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["split"] == split]
    assignment = {}
    for row in records:
        key = (str(row["patient"]), row["side"])
        if key in assignment or int(row["seed"]) != seed:
            raise ValueError("Duplicate fold identity or wrong seed")
        assignment[key] = int(row["fold"])
    expected = {(str(r["patient"]), r["side"]) for r in rows}
    if set(assignment) != expected:
        raise ValueError("Fold manifest does not cover the requested split exactly")
    result = np.array([assignment[(str(r["patient"]), r["side"])] for r in rows])
    patient_to_folds = defaultdict(set)
    for row, fold in zip(rows, result):
        patient_to_folds[str(row["patient"])].add(int(fold))
    if set(result) != set(range(folds)) or any(len(v) != 1 for v in patient_to_folds.values()):
        raise ValueError("Invalid folds or patient leakage between folds")
    return result


def prepare(root, output, fold_output, audit_output, folds=5, seed=42, official=True):
    root = Path(root)
    by_key = {}
    for split in EXPECTED:
        for path in sorted((root / split).glob("*/*.jpeg")):
            match = EYE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Invalid EyePACS filename: {path}")
            patient, side = match.groups()
            key = (patient, side.lower())
            if key in by_key:
                raise ValueError(f"Duplicate eye across directory tree: {key}")
            by_key[key] = {
                "patient": patient,
                "side": side.lower(),
                "label": int(path.parent.name),
                "split": split,
                "path": path.relative_to(root).as_posix(),
            }
    rows = []
    for (patient, side), row in sorted(by_key.items(), key=lambda p: (int(p[0][0]), p[0][1])):
        fellow = by_key.get((patient, "right" if side == "left" else "left"))
        if fellow is None:
            raise ValueError(f"Missing fellow: {patient}")
        rows.append({**row, "fellow_label": fellow["label"], "fellow_path": fellow["path"]})
    summary = validate_manifest(rows, root=root, official=official)
    write_csv(output, rows, FIELDS)
    assignments = []
    for split in ("train", "val"):
        subset = [r for r in rows if r["split"] == split]
        ids = patient_folds(subset, folds, seed)
        assignments.extend(
            {
                "patient": r["patient"],
                "side": r["side"],
                "split": split,
                "fold": int(f),
                "seed": seed,
            }
            for r, f in zip(subset, ids)
        )
    write_csv(fold_output, assignments)
    save_json(
        audit_output,
        {
            "status": "passed",
            "counts": summary,
            "patient_disjoint": True,
            "complete_pairs": True,
            "folds": folds,
            "seed": seed,
            **content_audit_status(manifest=output),
        },
    )
    return summary
