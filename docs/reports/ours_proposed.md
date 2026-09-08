# Ours Proposed

## Purpose

This report records the proposed boundary-wise adaptive bilateral fusion method
for paired-eye diabetic retinopathy grading.

## Current PHASE 1 Protocol

- Dataset: EyePACS official paired-eye split
- Output: target-eye 5-class DR severity grade
- Backbone: ConvNeXt-S
- Seed: 42
- OOF: 5-fold patient-level split
- Model selection: validation-based
- Metrics: QWK, macro-F1, Grade 3 F1 when available, MAE, accuracy, gap-based subgroup metrics, and boundary metrics

## Main Comparison Arms

| Arm | Method | Purpose |
|---|---|---|
| A | Single-eye ordinal expert | Target-eye-only baseline |
| B | Paired ordinal expert | Standard target + fellow-eye fusion |
| C | Fixed 50:50 fusion | Expert complementarity check |
| D | Global asymmetry gate | Patient-level rollback baseline |
| E | Boundary-wise end-to-end gate | Boundary-local learned gate |
| F | Loss-Guided OOF gate | Local probability-loss teacher |
| G | Counterfactual OOF gate | Final ordinal-benefit teacher |

## Result Table

| Method | Overall QWK | Gap >= 1 QWK | Gap >= 2 QWK | Macro-F1 | MAE | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| A. Single-eye | TBD | TBD | TBD | TBD | TBD | TBD |
| B. Paired fusion | TBD | TBD | TBD | TBD | TBD | TBD |
| C. Fixed 50:50 | TBD | TBD | TBD | TBD | TBD | TBD |
| D. Global gate | TBD | TBD | TBD | TBD | TBD | TBD |
| E. Boundary-wise gate | TBD | TBD | TBD | TBD | TBD | TBD |
| F. Loss-Guided OOF gate | TBD | TBD | TBD | TBD | TBD | TBD |
| G. Counterfactual OOF gate | TBD | TBD | TBD | TBD | TBD | TBD |

## Notes

Final claims should wait until PHASE 1 finishes and the teacher/gate analysis is
reviewed. Later robustness runs can add more seeds, backbones, and external
paired-eye validation.
