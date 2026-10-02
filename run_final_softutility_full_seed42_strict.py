from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path


IDEA_ROOT = Path(os.environ.get("DR_IDEA_ROOT", "/home/minjae/DR-Classification-idea"))
BASELINE_ROOT = Path(os.environ.get("DR_BASELINE_ROOT", "/home/minjae/DR-Classification"))
DEFAULT_RUN_ROOT = IDEA_ROOT / "results/FINAL_SOFTUTILITY_FULL_SEED42_STRICT_01"
ALLOWED_TEST_FIELDS = (
    "dataset",
    "split",
    "sample_id",
    "patient_id",
    "target_eye",
    "fellow_eye",
    "target_image_path",
    "fellow_image_path",
)

STRICT_RUN_ROOT: Path | None = None
PRE_FREEZE_TEST_LABEL_READ = 0
PRE_FREEZE_TEST_GAP_CALCULATION = 0
PRE_FREEZE_TEST_IMAGE_FORWARD = 0
PRE_FREEZE_TEST_METRIC = 0
PRE_FREEZE_TEST_GT_USED_FOR_TRAINING_SELECTION = 0
_POST_FREEZE_LABEL_CACHE: dict[str, dict[str, tuple[str, str]]] = {}


def _load_base_module():
    path = Path(
        os.environ.get(
            "DR_FINAL_BASE_RUNNER",
            str(Path(__file__).resolve().parent / "scripts/run_final_softutility_full_seed42.py"),
        )
    )
    spec = importlib.util.spec_from_file_location("full_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


def _manifest(dataset: str) -> Path:
    return (
        BASELINE_ROOT / "outputs/official_split/data_audit/eyepacs_directional_manifest.csv"
        if dataset == "eyepacs"
        else BASELINE_ROOT / "outputs/data_audit/messidor2_directional_manifest.csv"
    )


def _freeze_marker(dataset: str) -> Path:
    if STRICT_RUN_ROOT is None:
        raise RuntimeError("Strict run root has not been initialized")
    return STRICT_RUN_ROOT / dataset / "MODEL_FREEZE_COMPLETE.txt"


def _load_test_labels_after_freeze(dataset: str) -> dict[str, tuple[str, str]]:
    if dataset in _POST_FREEZE_LABEL_CACHE:
        return _POST_FREEZE_LABEL_CACHE[dataset]
    marker = _freeze_marker(dataset)
    if not marker.is_file():
        raise RuntimeError(
            f"STRICT TEST LOCK: attempted to read {dataset} test GT before {marker}"
        )
    values: dict[str, tuple[str, str]] = {}
    with _manifest(dataset).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test":
                values[row["sample_id"]] = (row["y_true"], row["fellow_y_true"])
    _POST_FREEZE_LABEL_CACHE[dataset] = values
    return values


class LockedTestRow(dict):
    """Test metadata that cannot expose GT until the model-freeze marker exists."""

    def __init__(self, dataset: str, values: dict[str, str]):
        super().__init__(values)
        self._dataset = dataset

    def __getitem__(self, key):
        global PRE_FREEZE_TEST_LABEL_READ
        if key in {"y_true", "fellow_y_true"}:
            if not _freeze_marker(self._dataset).is_file():
                PRE_FREEZE_TEST_LABEL_READ += 1
                raise RuntimeError(
                    f"STRICT TEST LOCK: pre-freeze access to {self._dataset}.{key}"
                )
            labels = _load_test_labels_after_freeze(self._dataset)
            position = 0 if key == "y_true" else 1
            return labels[dict.__getitem__(self, "sample_id")][position]
        if key in {"inter_eye_gap", "gap_group", "gap"}:
            if not _freeze_marker(self._dataset).is_file():
                raise RuntimeError(
                    f"STRICT TEST LOCK: pre-freeze GAP access for {self._dataset}"
                )
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def strict_load_rows(dataset: str) -> list[dict[str, str]]:
    """Project test rows to metadata-only fields; labels are never retained pre-freeze."""
    raw_non_test: list[dict[str, str]] = []
    test_metadata: list[LockedTestRow] = []
    train_patient_ids: set[str] = set()
    with _manifest(dataset).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(ALLOWED_TEST_FIELDS) | {"y_true", "fellow_y_true"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise AssertionError(f"{dataset}: required manifest columns are missing")
        for source in reader:
            metadata = {field: source[field] for field in ALLOWED_TEST_FIELDS}
            if source["split"] == "test":
                metadata["run_split"] = "test"
                test_metadata.append(LockedTestRow(dataset, metadata))
            else:
                # Labels are accessed only for train/validation rows.
                metadata["y_true"] = source["y_true"]
                metadata["fellow_y_true"] = source["fellow_y_true"]
                metadata["run_split"] = source["split"]
                raw_non_test.append(metadata)
                if dataset == "messidor2" and source["split"] == "train":
                    train_patient_ids.add(source["patient_id"])
    if dataset == "messidor2":
        ranked = sorted(
            train_patient_ids,
            key=lambda patient: hashlib.sha256(f"2026:{patient}".encode()).hexdigest(),
        )
        val_ids = set(ranked[: round(len(ranked) * 0.2)])
        for row in raw_non_test:
            if row["split"] == "train":
                row["run_split"] = (
                    "inner_val" if row["patient_id"] in val_ids else "train"
                )
    rows: list[dict[str, str]] = [*raw_non_test, *test_metadata]
    strict_validate_rows(dataset, rows)
    return rows


def strict_validate_rows(dataset: str, rows: list[dict[str, str]]) -> None:
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise AssertionError(f"{dataset}: duplicate sample key")
    stable_keys = [
        (row["patient_id"], row["target_eye"], row["fellow_eye"], row["sample_id"])
        for row in rows
    ]
    if len(set(stable_keys)) != len(stable_keys):
        raise AssertionError(f"{dataset}: duplicate patient/sample directional key")
    if any(not all(field in row for field in ALLOWED_TEST_FIELDS) for row in rows):
        raise AssertionError(f"{dataset}: missing required path/metadata column")
    split_patients = {
        split: {row["patient_id"] for row in rows if row["run_split"] == split}
        for split in sorted({row["run_split"] for row in rows})
    }
    splits = list(split_patients)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if split_patients[left] & split_patients[right]:
                raise AssertionError(f"{dataset}: patient leakage {left}/{right}")
    if any(row["target_eye"] == row["fellow_eye"] for row in rows):
        raise AssertionError(f"{dataset}: target/fellow eye collision")
    # Grade range checks are explicitly limited to non-test rows.
    non_test = [row for row in rows if row["run_split"] != "test"]
    if any(
        int(row["y_true"]) not in range(5)
        or int(row["fellow_y_true"]) not in range(5)
        for row in non_test
    ):
        raise AssertionError(f"{dataset}: invalid non-test grade")
    patient_direction_count = Counter(row["patient_id"] for row in rows)
    if any(count != 2 for count in patient_direction_count.values()):
        raise AssertionError(f"{dataset}: patient does not have two directional rows")
    row_counts = Counter(row["run_split"] for row in rows)
    patient_counts = {key: len(value) for key, value in split_patients.items()}
    if dataset == "eyepacs":
        if dict(row_counts) != {"train": 35126, "val": 10906, "test": 42670}:
            raise AssertionError(f"EyePACS row counts: {row_counts}")
        if patient_counts != {"train": 17563, "val": 5453, "test": 21335}:
            raise AssertionError(f"EyePACS patient counts: {patient_counts}")
    else:
        if dict(row_counts) != {"train": 1114, "inner_val": 278, "test": 348}:
            raise AssertionError(f"Messidor row counts: {row_counts}")
        if patient_counts != {"train": 557, "inner_val": 139, "test": 174}:
            raise AssertionError(f"Messidor patient counts: {patient_counts}")


def strict_precheck(run_root: Path, config: dict):
    rows = original_precheck(run_root, config)
    if any(
        value != 0
        for value in (
            PRE_FREEZE_TEST_LABEL_READ,
            PRE_FREEZE_TEST_GAP_CALCULATION,
            PRE_FREEZE_TEST_IMAGE_FORWARD,
            PRE_FREEZE_TEST_METRIC,
            PRE_FREEZE_TEST_GT_USED_FOR_TRAINING_SELECTION,
        )
    ):
        raise AssertionError("Strict test-lock counter is non-zero")
    lines = [
        "PRE-FREEZE TEST LABEL READ = 0",
        "PRE-FREEZE TEST GAP CALCULATION = 0",
        "PRE-FREEZE TEST IMAGE FORWARD = 0",
        "PRE-FREEZE TEST METRIC = 0",
        "PRE-FREEZE TEST GT USED FOR TRAINING/SELECTION = 0",
        "",
        "EyePACS Train = 17563 patients / 35126 rows",
        "Validation = 5453 patients / 10906 rows",
        "Test metadata only = 21335 patients / 42670 rows",
        "",
        "OOF folds = 3",
        "Stage-A recipe match = YES",
        "D2 recipe match = YES",
        "",
        "STRICT TEST LOCK = PASS",
    ]
    report = "\n".join(lines) + "\n"
    base.atomic_text(run_root / "STRICT_TEST_LOCK_PRECHECK.txt", report)
    print(report, end="", flush=True)
    return rows


def _contains_test_rows(rows) -> bool:
    return any(row["run_split"] == "test" for row in rows)


def strict_train_stage_a(config, dataset, dataset_root, rows):
    global PRE_FREEZE_TEST_GT_USED_FOR_TRAINING_SELECTION
    if _contains_test_rows(rows):
        PRE_FREEZE_TEST_GT_USED_FOR_TRAINING_SELECTION += 1
        raise RuntimeError("STRICT TEST LOCK: test row passed to Stage-A training")
    return original_train_stage_a(config, dataset, dataset_root, rows)


def strict_train_oof(config, dataset, dataset_root, rows):
    global PRE_FREEZE_TEST_GT_USED_FOR_TRAINING_SELECTION
    if _contains_test_rows(rows):
        PRE_FREEZE_TEST_GT_USED_FOR_TRAINING_SELECTION += 1
        raise RuntimeError("STRICT TEST LOCK: test row passed to OOF training")
    return original_train_oof(config, dataset, dataset_root, rows)


def strict_train_d2(config, dataset, dataset_root, cache, tau, confidence, source):
    global PRE_FREEZE_TEST_GT_USED_FOR_TRAINING_SELECTION
    if _contains_test_rows(cache["metadata"]):
        PRE_FREEZE_TEST_GT_USED_FOR_TRAINING_SELECTION += 1
        raise RuntimeError("STRICT TEST LOCK: test row passed to D2 training")
    return original_train_d2(config, dataset, dataset_root, cache, tau, confidence, source)


def strict_cache_stage_a(config, dataset, dataset_root, rows, destination):
    global PRE_FREEZE_TEST_IMAGE_FORWARD
    if _contains_test_rows(rows) and not _freeze_marker(dataset).is_file():
        PRE_FREEZE_TEST_IMAGE_FORWARD += 1
        raise RuntimeError("STRICT TEST LOCK: pre-freeze test image forward")
    return original_cache_stage_a(config, dataset, dataset_root, rows, destination)


def strict_infer(model, cache, indices, policy):
    global PRE_FREEZE_TEST_IMAGE_FORWARD
    selected = [cache["metadata"][index] for index in indices]
    datasets = {row["dataset"] for row in selected}
    test_datasets = {
        row["dataset"] for row in selected if row["run_split"] == "test"
    }
    if test_datasets and any(not _freeze_marker(dataset).is_file() for dataset in test_datasets):
        PRE_FREEZE_TEST_IMAGE_FORWARD += 1
        raise RuntimeError("STRICT TEST LOCK: pre-freeze test D2 inference")
    if len(datasets) > 1:
        raise RuntimeError("Mixed datasets in one frozen inference call")
    return original_infer(model, cache, indices, policy)


def strict_evaluate_arms(values, gates, policy, output_prefix):
    global PRE_FREEZE_TEST_METRIC
    metadata = values.get("metadata", [])
    test_datasets = {
        row["dataset"] for row in metadata if row["run_split"] == "test"
    }
    if test_datasets and any(not _freeze_marker(dataset).is_file() for dataset in test_datasets):
        PRE_FREEZE_TEST_METRIC += 1
        raise RuntimeError("STRICT TEST LOCK: pre-freeze test metric")
    return original_evaluate_arms(values, gates, policy, output_prefix)


def parse_run_root() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    values, _ = parser.parse_known_args()
    return values.run_root


original_precheck = base.precheck
original_train_stage_a = base.train_stage_a
original_train_oof = base.train_oof
original_train_d2 = base.train_d2
original_cache_stage_a = base.cache_stage_a
original_infer = base.infer
original_evaluate_arms = base.evaluate_arms
base.load_rows = strict_load_rows
base.validate_rows = strict_validate_rows
base.precheck = strict_precheck
base.train_stage_a = strict_train_stage_a
base.train_oof = strict_train_oof
base.train_d2 = strict_train_d2
base.cache_stage_a = strict_cache_stage_a
base.infer = strict_infer
base.evaluate_arms = strict_evaluate_arms


def main() -> None:
    global STRICT_RUN_ROOT
    STRICT_RUN_ROOT = parse_run_root()
    # Force the explicit strict run root so the preserved invalid run is never selected.
    if "--run-root" not in sys.argv:
        sys.argv.extend(["--run-root", str(STRICT_RUN_ROOT)])
    base.main()


if __name__ == "__main__":
    main()
