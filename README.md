# Bilateral Diabetic Retinopathy Grading

Research code for **boundary-wise fellow-eye correction and adaptive fusion**.

Retinal images from the two eyes contain related information, but disease severity can differ between eyes. This project investigates how fellow-eye information can improve five-class diabetic retinopathy grading, and when that fusion can harm an otherwise correct prediction.

## Research focus

- Compare target-only classification, simple bilateral fusion, and learned correction.
- Study boundary-wise correction, utility–strength modeling, and soft gating.
- Evaluate overall ordinal grading performance alongside prediction-level benefit and harm.
- Track patient-level data separation, feature generation, and experiment provenance.

**Status:** Ongoing research. This repository is an experiment snapshot, not a statement that every later local experiment or result has been released. No performance ranking or clinical-use claim is made here.

## Code guide

- [Model implementation](idea_work/ours/) — Stage-A and bilateral correction experiments.
- [Gate components](src/drpair/gates/) — Feature construction, networks, and fitting.
- [Evaluation utilities](src/drpair/evaluation/) — Ordinal metrics, reporting, and statistical analysis.
- [Experiment scripts](scripts/) — Preparation, training, and evaluation entry points.
- [Tests](tests/) — Unit tests and implementation checks.
- [Configurations](configs/) — Protocol and baseline settings.

## Experiment snapshot and execution

The instructions below describe the committed seed-42 workflow. Running it requires separately obtained datasets and local assets; no training was run to prepare this README.

Reproducible diabetic-retinopathy experiments for the final seed-42 study.
The repository contains two execution tracks:

1. **OUR IDEA**: Target-anchored soft-utility fusion on EyePACS and Messidor-2.
2. **Paper baselines**: Huang Single/PFF, DR-GCNN Single/BFFN, and D-ODR.

Large datasets, caches, pretrained weights, checkpoints, and run logs are local
artifacts and are intentionally excluded from Git.

## Repository layout

```text
.dvc/            DVC configuration
configs/         Final baseline and protocol configuration
data/            Dataset metadata and local data mounts (images are ignored)
docs/            Final protocol and research documentation
google/          Local D-ODR ViT weights (ignored)
idea_work/       OUR IDEA implementation and configuration
outputs/         Reproducible manifests and generated baseline artifacts
scripts/         Data preparation, training, evaluation, and local launchers
src/drpair/      Shared Python package
tests/           Unit and implementation-alignment tests
```

## Fixed experiment scope

- Seed: `42`
- Datasets: EyePACS and Messidor-2
- OUR IDEA effective batch: physical `16` x accumulation `4` = `64`
- DataLoader workers: `8`
- Baselines:
  - Huang Single: effective batch `16`
  - Huang PFF: effective batch `64`
  - DR-GCNN Single: effective batch `16`
  - DR-GCNN BFFN: effective batch `64`
  - D-ODR: batch `128`

The OUR IDEA pipeline applies a strict test lock: official test labels and model
inference are unavailable until Stage-A, D2, and SoftGate are frozen and the
`MODEL_FREEZE_COMPLETE.txt` marker exists.

## Local setup

```powershell
py -3.12 -m venv .venv-local-gpu
.\.venv-local-gpu\Scripts\python.exe -m pip install -e .
```

Required local-only assets:

- `data/eyepacs/`
- `data/messidor2/`
- `data/cache/`
- `google/vit-base-patch16-224/pytorch_model.bin`

## OUR IDEA

Precheck only:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
.\.venv-local-gpu\Scripts\python.exe run_final_softutility_full_seed42_strict.py `
  --run-root idea_work/results/FINAL_SOFTUTILITY_FULL_SEED42_LOCAL `
  --precheck-only
```

Full local run (detached worker):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_idea_full_worker.ps1
```

The current production result directory is under `idea_work/results/` and is
ignored by Git so checkpoints and logs cannot be pushed accidentally.

## Baselines

Prepare caches and run the real-image smoke suite:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_cache_preparation.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_local_smoke_queue.ps1
```

Run final seed-42 baselines after smoke and freeze checks pass:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_full_baselines_seed42.ps1
```

## Validation

```powershell
.\.venv-local-gpu\Scripts\python.exe -m pytest
```

Before committing, verify that no dataset, checkpoint, cache, or result bundle is
staged:

```powershell
git status --short
git check-ignore data/eyepacs data/messidor2 data/cache google idea_work/results
```

## Git workflow

```powershell
git add -A
git status
git commit -m "Organize final seed-42 experiment"
git pull --rebase origin main
git push origin main
```

Do not run `git pull --rebase` while the working tree has uncommitted changes.
