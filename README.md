# DR-Classification

Diabetic retinopathy 5-class severity grading project.

## Repository Layout

```text
DR-Classification/
├── data/                 # Dataset notes, split files, and future DVC-tracked data pointers
├── docs/                 # Papers, experiment logs, and research notes
├── metrics/              # Tables, figures, prediction CSVs, and evaluation summaries
├── scripts/              # Data preparation, training, evaluation, and server scripts
├── tests/                # Unit and smoke tests
└── yardstick/            # Comparison protocols and baseline definitions
```

## Active Comparator Papers

The external comparison papers are stored in `docs/papers/comparators/`.

- DiffGeo-AOR
- CLIP-DR
- ABiD
- PLDG
- DG-ADR

## Experiment Logging

Write daily work logs in `docs/experiment_logs/`.
Use one Markdown file per day, for example `2026-09-02_setup.md`.
