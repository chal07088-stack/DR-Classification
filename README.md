# DR-Classification

Diabetic retinopathy 5-class severity grading experiments.

The current repository version is organized around the PHASE 1 paired-eye
EyePACS experiment for boundary-wise adaptive bilateral fusion.

## Repository Layout

```text
DR-Classification/
├── configs/                  # Experiment configuration files
├── data/                     # Dataset notes and DVC-ready placeholders
├── docs/                     # Papers, reports, and experiment logs
├── experimental_results/     # Reproducible experiment output records
├── metrics/                  # Curated final metric tables and figures
├── scripts/                  # Run scripts grouped by experiment phase
├── src/                      # Python source package
├── tests/                    # Smoke/unit test placeholders
└── yardstick/                # Comparison protocols and baseline definitions
```

## Active PHASE 1

PHASE 1 tests whether paired-eye fusion helps on average but harms target-eye
grading in asymmetric patients, and whether a boundary-wise gate can selectively
roll back harmful ordinal decisions.

Fixed PHASE 1 scope:

- ConvNeXt-S only
- training seed 42 only
- EyePACS official paired-eye split only
- 5-fold patient-level OOF
- single expert, paired expert, Teacher A/B, boundary-wise gate, A-G ablation

The code is in `src/kimeyepacs/`.

## Main Commands

The scripts assume the local conda environment used during development:

```powershell
.\scripts\phase1\run_vram_probe.ps1
.\scripts\phase1\run_phase1_smoke.ps1
.\scripts\phase1\run_phase1_seed42.ps1
```

Edit `configs/phase1_eyepacs_convnext_s.yaml` if the EyePACS data path or
output path differs on another machine.

## Data

Raw image data is not committed to GitHub.

Expected local data root:

```text
C:/Users/use/Desktop/pair/eyepacs_official_split
  train/0..4/*.jpeg
  val/0..4/*.jpeg
  test/0..4/*.jpeg
```

Left and right eyes from the same patient must remain in the same split or fold.

## Results

Raw run outputs are written under:

```text
experimental_results/ours_proposed/
```

Large checkpoints, raw images, and heavy prediction dumps should not be
committed directly. Use DVC or record the local/server path in
`docs/experiment_logs/`.

## Comparator Papers

External comparison papers are stored in `docs/papers/comparators/`.

- DiffGeo-AOR
- CLIP-DR
- ABiD
- PLDG
- DG-ADR
