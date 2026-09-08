"""Check official label provenance and exact-byte duplicates using the DVC index."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from drpair.data.manifest import read_manifest, write_csv
from drpair.runtime import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    rows = read_manifest(repo / "data" / "splits" / "eyepacs.csv")
    labels = {}
    for source, train in [(args.train_labels, True), (args.solution, False)]:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            for record in csv.DictReader(stream):
                name = record["image"]
                split = "train" if train else {"Public": "val", "Private": "test"}[record["Usage"]]
                if name in labels:
                    raise ValueError(f"Repeated source identity: {name}")
                labels[name] = (split, int(record["level"]))
    mismatches = [
        r["path"] for r in rows if labels.get(Path(r["path"]).stem) != (r["split"], r["label"])
    ]
    if mismatches or len(labels) != len(rows):
        raise ValueError(f"Official source label mismatch: {len(mismatches)}")
    metadata = yaml.safe_load((repo / "data" / "eyepacs.dvc").read_text())
    oid = metadata["outs"][0]["md5"]
    tree = json.loads((args.store / "files" / "md5" / oid[:2] / oid[2:]).read_text())
    by_path = {r["path"]: r for r in rows}
    groups = defaultdict(list)
    for item in tree:
        groups[item["md5"]].append(by_path[item["relpath"].replace("\\", "/")])
    duplicates = [g for g in groups.values() if len(g) > 1]
    duplicate_rows = []
    for i, group in enumerate(duplicates):
        for row in group:
            duplicate_rows.append(
                {
                    "group": i,
                    "path": row["path"],
                    "split": row["split"],
                    "label": row["label"],
                    "patient": row["patient"],
                    "cross_split": len({r["split"] for r in group}) > 1,
                    "label_conflict": len({r["label"] for r in group}) > 1,
                }
            )
    summary = {
        "source_label_match": True,
        "images": len(rows),
        "unique_content_hashes": len(groups),
        "byte_duplicate_groups": len(duplicates),
        "byte_duplicate_images": sum(map(len, duplicates)),
        "cross_split_duplicate_groups": sum(len({r["split"] for r in g}) > 1 for g in duplicates),
        "conflicting_label_groups": sum(len({r["label"] for r in g}) > 1 for g in duplicates),
        "near_duplicate_images_checked": False,
        "dvc_tree_md5": oid,
    }
    out = repo / "experimental_results" / "repository_validation"
    save_json(out / "data_content_audit.json", summary)
    if duplicate_rows:
        write_csv(out / "exact_duplicates.csv", duplicate_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
