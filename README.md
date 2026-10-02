# DR-Classification

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
