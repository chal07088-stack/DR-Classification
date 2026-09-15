# Quick benchmark work log — 2026-09-15

## PHASE 0

- Repository confirmed at `C:\Users\use\Desktop\DR-Classification` on `main`.
- Origin confirmed as `https://github.com/chal07088-stack/DR-Classification.git`.
- Pre-existing untracked figures, guide, Phase 2B outputs, and PPT were recorded and preserved.
- Python 3.12.14 project environment, PyTorch 2.11.0+cu128, and RTX 5070 Ti verified.
- Official comparator repositories inspected at fixed commits:
  - Huang: `e6cd6b49a27612166f7e60f68bf511ac5030140b`
  - DRGCNN: `bf80220a5786bf52678519f381b742827447402f`
  - CLIP-DR: `ca7d88c44f13d0e77bea4b0c2381b51deb75d3c1`
  - AOR-DR: `4b67c6892e0035829ec3719a8a15e3371654a5a4`

## PHASE 1

- Implemented deterministic `(patient_max_grade, gap_bucket)` patient stratification.
- Generated the 10% / 3% / 7% EyePACS patient split with seed 42.
- EyePACS hard gate passed: patient leakage 0, image leakage 0, all grades present.
- Downloaded and hashed the Kaggle adjudicated Messidor-2 grade CSV.
- Messidor exact-ID join marked blocked because official ADCIS pairing metadata is absent.
- No training was started before the EyePACS hard gate passed.

## Open reproducibility issue

The local DVC tree index exists, but 9,290 of 88,702 EyePACS content blobs are absent
from the configured local cache. Workspace images are present. Cache repair must be
completed before final handoff to another machine.

## PHASE 2 — started

- Precomputed all 7,024 selected images using the Huang repository's black-border/FOV
  crop logic at 512 x 512; raw images were not modified.
- Added isolated B0 ConvNeXt-S, B1-S Huang single, and B1-P frozen-encoder PFF configs.
- The controlled Huang pair shares preprocessing, ResNet-50 encoder initialization,
  scalar MSE objective, target-eye cohort, and capped validation-selection policy.
- B0 and B1-S GPU forward/backward smoke tests passed. A smoke-only undefined-QWK
  checkpoint edge case was detected and fixed before full training.
- Full order is fixed as B0, B1-S, then B1-P. B1-P cannot start without a completed
  validation-selected B1-S checkpoint.
