# Official-Setting Comparison

## Purpose

This report compares external official-code methods on the same APTOS2019 split while preserving each paper's official hyperparameters whenever possible.

## Fixed Across Methods

- Dataset: APTOS2019 original 3,662 images
- Split: stratified 60:20:20
- Data split seed: 42
- Training seeds: 42, 123, 128
- Model selection: validation-based
- Metrics: QWK, Macro-F1, Grade 3 F1, MAE, Accuracy

## Method-Specific

- Backbone, input size, optimizer, learning rate, scheduler, augmentation, and pretraining follow each official repository whenever compatible with the dataset.

## Result Table

| Method | QWK | Macro-F1 | Grade 3 F1 | MAE | Accuracy |
|---|---:|---:|---:|---:|---:|
| DiffGeo-AOR | TBD | TBD | TBD | TBD | TBD |
| CLIP-DR | TBD | TBD | TBD | TBD | TBD |
| ABiD | TBD | TBD | TBD | TBD | TBD |
| PLDG | TBD | TBD | TBD | TBD | TBD |
| DG-ADR | TBD | TBD | TBD | TBD | TBD |
| Ours-Proposed | TBD | TBD | TBD | TBD | TBD |
