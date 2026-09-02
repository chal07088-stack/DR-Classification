# Unified-Setting Comparison

## Purpose

This report compares compatible methods under the project's common hyperparameter protocol to reduce training-setting differences.

## Common Protocol

- Dataset: APTOS2019 original 3,662 images
- Split: stratified 60:20:20
- Data split seed: 42
- Input: 384 x 384 when compatible
- Preprocessing: same project preprocessing
- Offline augmentation: none
- Physical batch: 2
- Gradient accumulation: 8
- Effective batch: 16
- Epochs: 60
- Min epochs: 15
- Patience: 15
- Min save epoch: 10
- Backbone LR: 3e-5
- Head LR: 1e-4
- Weight decay: 1e-4
- Warmup: none
- Scheduler: CosineAnnealingLR
- AMP: on
- Gradient clipping: 1.0
- EMA: 0.999
- Training seeds: 42, 123, 128
- Model selection: validation-based
- Metrics: QWK, Macro-F1, Grade 3 F1, MAE, Accuracy

## Result Table

| Method | QWK | Macro-F1 | Grade 3 F1 | MAE | Accuracy |
|---|---:|---:|---:|---:|---:|
| DiffGeo-AOR | TBD | TBD | TBD | TBD | TBD |
| CLIP-DR | TBD | TBD | TBD | TBD | TBD |
| ABiD | TBD | TBD | TBD | TBD | TBD |
| PLDG | TBD | TBD | TBD | TBD | TBD |
| DG-ADR | TBD | TBD | TBD | TBD | TBD |
| Ours-Proposed | TBD | TBD | TBD | TBD | TBD |

## Compatibility Notes

If a method cannot support one of the common settings without changing its core implementation, document the exception here.
