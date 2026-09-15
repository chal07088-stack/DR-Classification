# PHASE 1 — Data Audit

## Status

- EyePACS Quick20: **PASSED**
- Messidor-2 exact-ID bilateral join: **BLOCKED — missing official ADCIS pairing spreadsheet and licensed images**
- Model training performed in this phase: **No**

## EyePACS source and fixed split

The actual local `trainLabels.csv` contains 35,126 images from 17,563 patients. Every
patient has exactly one left and one right image. The source SHA-256 is
`1dd6600d01dd0ff1e42ce840ffa501ae32b346745ff4c4f9296e62a2e9632f88`.

The patient-level split was fixed with seed 42 using `(patient_max_grade, gap_bucket)`
strata before any benchmark model result was observed.

| Split | Patients | Images | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Gap 0 patients | Gap 1 patients | Gap >=2 patients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1,756 | 3,512 | 2,583 | 243 | 528 | 88 | 70 | 1,533 | 148 | 75 |
| Validation | 527 | 1,054 | 774 | 73 | 160 | 26 | 21 | 460 | 45 | 22 |
| Test | 1,229 | 2,458 | 1,807 | 170 | 371 | 61 | 49 | 1,072 | 104 | 53 |

Patient overlap is 0 and image overlap is 0. All selected patients retain complete
left/right pairs. The test split contains 106 target eyes with gap >=2, above the
predefined 30-target-eye QWK stability threshold.

Machine-readable evidence is in `results/data_audit/eyepacs_split_summary.json` and
`results/data_audit/eyepacs_split_audit.csv`.

## Messidor-2

The third-party adjudicated Kaggle file was downloaded and audited independently. It
contains 1,748 unique image IDs, 1,744 gradable five-class DR grades, and four ungradable
images. The gradable distribution is 1,017 / 270 / 347 / 75 / 35 for grades 0–4.

The official ADCIS release states that Messidor-2 contains 874 examinations and 1,748
images and supplies a separate pairing spreadsheet, but download requires registration.
That official spreadsheet and the licensed images are not present locally. Consequently,
no complete-pair count has been assumed. The Messidor stage remains blocked while the
EyePACS benchmark may continue.

