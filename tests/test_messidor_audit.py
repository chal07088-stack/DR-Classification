import csv

from drpair.benchmark.messidor import (
    audit_messidor_pairing,
    normalize_image_id,
    prepare_messidor_splits,
)


def _write(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_normalize_image_id_is_conservative():
    assert normalize_image_id(r"folder\IM0001.JPG") == "im0001"
    assert normalize_image_id("  A/2000_PP.png ") == "2000_pp"


def test_exact_id_join_counts_complete_pairs(tmp_path):
    pairing = tmp_path / "pairs.csv"
    grades = tmp_path / "grades.csv"
    output = tmp_path / "out"
    pair_rows = [
        {"exam": str(index), "left": f"L{index}.JPG", "right": f"R{index}.JPG"}
        for index in range(874)
    ]
    grade_rows = []
    for index in range(874):
        for side in ("L", "R"):
            grade_rows.append(
                {
                    "image_id": f"{side}{index}.jpg",
                    "adjudicated_dr_grade": str(index % 5),
                    "adjudicated_dme": "0",
                    "adjudicated_gradable": "1",
                }
            )
    _write(pairing, ["exam", "left", "right"], pair_rows)
    _write(
        grades,
        ["image_id", "adjudicated_dr_grade", "adjudicated_dme", "adjudicated_gradable"],
        grade_rows,
    )
    result = audit_messidor_pairing(pairing, grades, output, "left", "right", "exam")
    assert result["status"] == "passed"
    assert result["official_examinations"] == 874
    assert result["exact_matched_images"] == 1748
    assert result["complete_examinations_with_both_eye_grades"] == 874
    assert result["unmatched_official_images"] == 0


def test_exact_id_join_accepts_semicolon_pairing_and_trimmed_headers(tmp_path):
    pairing = tmp_path / "official_pairs.csv"
    grades = tmp_path / "grades.csv"
    output = tmp_path / "out"
    pair_rows = [f"L{index}.png;R{index}.png" for index in range(874)]
    pairing.write_text("left;right \n" + "\n".join(pair_rows) + "\n", encoding="utf-8")
    grade_rows = []
    for index in range(874):
        for side in ("L", "R"):
            grade_rows.append(
                {
                    "image_id": f"{side}{index}.png",
                    "adjudicated_dr_grade": str(index % 5),
                    "adjudicated_gradable": "1",
                }
            )
    _write(grades, ["image_id", "adjudicated_dr_grade", "adjudicated_gradable"], grade_rows)
    result = audit_messidor_pairing(pairing, grades, output, "left", "right")
    assert result["status"] == "passed"
    assert result["complete_examinations_with_both_eye_grades"] == 874


def test_exact_id_join_checks_local_images_with_extension_normalization(tmp_path):
    pairing = tmp_path / "pairs.csv"
    grades = tmp_path / "grades.csv"
    images = tmp_path / "images"
    images.mkdir()
    output = tmp_path / "out"
    _write(
        pairing,
        ["left", "right"],
        [{"left": f"L{index}.png", "right": f"R{index}.png"} for index in range(874)],
    )
    grade_rows = []
    for index in range(874):
        for side in ("L", "R"):
            (images / f"{side}{index}.JPG").write_bytes(b"image")
            grade_rows.append(
                {
                    "image_id": f"{side}{index}.png",
                    "adjudicated_dr_grade": "0",
                    "adjudicated_gradable": "1",
                }
            )
    _write(grades, ["image_id", "adjudicated_dr_grade", "adjudicated_gradable"], grade_rows)
    result = audit_messidor_pairing(pairing, grades, output, "left", "right", image_root=images)
    assert result["status"] == "passed"
    assert result["local_image_count"] == 1748
    assert result["unmatched_local_image_filenames"] == []


def test_complete_pair_split_is_patient_disjoint_and_uses_all_pairs(tmp_path):
    pairs = tmp_path / "messidor_complete_pairs.csv"
    images = tmp_path / "images"
    images.mkdir()
    rows = ["examination_id,left_image_id,right_image_id,left_grade,right_grade,gap"]
    for index in range(20):
        left, right = f"left_{index}.jpg", f"right_{index}.jpg"
        (images / left).write_bytes(b"left")
        (images / right).write_bytes(b"right")
        rows.append(
            f"exam-{index},{left},{right},{index % 5},{(index + index % 3) % 5},{index % 3}"
        )
    pairs.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = prepare_messidor_splits(pairs, images, tmp_path)
    assert result["status"] == "passed"
    assert result["target_patient_counts"] == {"train": 14, "val": 3, "test": 3}
    assert result["patient_overlap"] == result["image_overlap"] == 0
    assert sum(
        item["patients"] for key, item in result["splits"].items() if key != "full"
    ) == 20
