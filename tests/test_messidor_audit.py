import csv

from drpair.benchmark.messidor import audit_messidor_pairing, normalize_image_id


def _write(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_normalize_image_id_is_conservative():
    assert normalize_image_id(r"folder\IM0001.JPG") == "im0001.jpg"
    assert normalize_image_id("  A/2000_PP.png ") == "2000_pp.png"


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

