# Existing artifact inventory — 2026-09-18

This inventory records the pre-cleanup state and separates historical work from the next full-data
experiment.

## Cleanup status

On 2026-09-18, all files below `results/` and `experimental_results/` were deleted at the user's
request while preserving their complete directory structure. The cleanup removed 925 files
(10.81 GiB) and preserved 106 subdirectories. Of those files, 443 are Git-tracked and remain
recoverable from repository history until the deletion is committed; the local-only files are not
recoverable through Git. The historical sections below describe the pre-cleanup contents.

## Data sources to preserve

| Dataset/source | Location | Verified contents | Status |
|---|---|---:|---|
| EyePACS working tree | `data/eyepacs/` | 88,702 images, 91,275.93 MiB | Keep; DVC-managed/hard-linked |
| EyePACS official split source | `C:/Users/use/Desktop/pair/eyepacs_official_split/` | 88,703 files | Keep as read-only source |
| EyePACS raw source | `C:/Users/use/Desktop/pair/eyepacs_official_raw/` | 53,588 files | Keep as read-only source |
| EyePACS DVC store | `C:/Users/use/Desktop/pair/dvc-store/` | 79,413 files | Keep; do not reorganize manually |
| Messidor-2 images | `C:/Users/use/Desktop/messidor/IMAGES/` | 1,748 images | Keep as read-only source |
| Messidor-2 labels/pairing | `C:/Users/use/Desktop/messidor/` | `messidor_data.csv`, `messidor-2.csv` | Keep |

The EyePACS working tree is backed by hard links. Its apparent size must not be interpreted as
additional physical storage, and image contents must not be modified in place.

## Audited populations to preserve

- EyePACS: 44,351 complete bilateral patients / 88,702 eyes across the existing official
  train, validation and test partitions. The partitions are patient-disjoint.
- EyePACS official train subset: 17,563 complete bilateral patients / 35,126 eyes.
- Messidor-2: 870 complete graded bilateral examinations / 1,740 eyes. Four of 874 official
  examinations are incomplete because one eye has no valid grade.
- Canonical audit files are under `data/splits/`, `results/data_audit/`, and
  `experimental_results/repository_validation/`.

## GitHub-tracked historical results

The following are historical artifacts, not the output location for the new full-data study.

### `experimental_results/`

- `phase1_convnext_s_seed42_staged_v1/`: earlier ConvNeXt single/paired staged experiment.
- `phase2_convnext_s_seed123_replication_v1/`: earlier seed-123 replication.
- `phase2_convnext_s_seed128_replication_v1/`: earlier seed-128 replication.
- `phase2_multiseed_v1/`: aggregate multi-seed summaries.
- `design_power/`, `gpu_preflight/`, `repository_validation/`: design and validation artifacts.

### `results/`

- `runs/phase2*`, `runs/phase3*`, `runs/phase4*`: QUICK/PILOT benchmark outputs.
- `runs/messidor_pilot/`, `runs/joint_eyepacs_messidor_pilot/`: prior pilot outputs.
- `messidor_full_internal_pilot_v13_seed42/`: proposed V0–V5 internal pilot, including the
  corrected V4 additions.
- `analysis/`, `figures/`, `tables/`: summaries derived from historical runs.

These results may be cited as development history only. They must not be mixed with the locked
full-v1 tables or used to select full-v1 hyperparameters after viewing its test results.

## Local-only historical artifacts

The working tree currently also contains untracked outputs, including:

- `experimental_results/phase2b_huang_matched_v1/`
- `results/messidor_complete_loss_ablation_seed42/`
- `results/messidor_v2_corrected_cf_only_seed42/`
- corrected diagnostics and figures inside `results/messidor_full_internal_pilot_v13_seed42/`
- `results/runs/phase3/` and result ZIP archives
- top-level `.test-*`, `.test_tmp_*`, `tmp/`, `output/`
- standalone `results/*.stdout.log` and `results/*.stderr.log`

The first four groups are research history and should be archived before any deletion. The test
directories, temporary directories, duplicated ZIPs and loose logs are cleanup candidates, but
must be deleted only after an explicit path-by-path approval.

## Official-source repositories

Local, Git-ignored clones exist under `third_party/`:

- `Huang-2023`
- `DRGCNN`
- `D-ODR`
- `AOR-DR`
- `CLIP-DR` (historical; excluded from full-v1)

Each included method must record its exact upstream URL and commit in the full-v1 provenance.

Current source audit:

| Method | Upstream commit | Local state |
|---|---|---|
| Huang | `e6cd6b49a27612166f7e60f68bf511ac5030140b` | clean |
| DRGCNN | `bf80220a5786bf52678519f381b742827447402f` | clean |
| D-ODR | `697dc509625b395d08dbe0804430fd903270fbf5` | clean |
| AOR-DR | `4b67c6892e0035829ec3719a8a15e3371654a5a4` | source unchanged; local `__pycache__` only |

## New-study namespace

All new runs go only under:

```text
experimental_results/full_v1/
  manifests/
  eyepacs_in_domain/<method>/seed_<seed>/
  messidor_external/<method>/seed_<seed>/
  messidor_in_domain/<method>/seed_<seed>/
  joint_multisource/<method>/seed_<seed>/
  aggregate/
```

Do not write new full runs under `results/runs/`, and do not overwrite any historical output.
Checkpoints, predictions and extracted features remain Git-ignored and should be stored with DVC;
small configs, metrics, logs, provenance and reports may be committed to Git.
