# PHASE 1 Method Summary

This repository currently centers on the PHASE 1 EyePACS paired-eye experiment.

## Scope

- Dataset: EyePACS official paired-eye split
- Backbone: ConvNeXt-S
- Seed: 42
- OOF: 5-fold patient-level out-of-fold prediction
- Output: target-eye 5-class diabetic retinopathy grade

## Main Arms

| Arm | Method | Purpose |
|---|---|---|
| A | Single-eye ordinal expert | Target-eye-only baseline |
| B | Paired ordinal expert | Standard bilateral fusion baseline |
| C | Fixed 50:50 fusion | Complementarity check |
| D | Global asymmetry gate | Patient-level rollback baseline |
| E | Boundary-wise end-to-end gate | Boundary-local gate baseline |
| F | Loss-Guided OOF gate | Local probability-loss teacher |
| G | Counterfactual OOF gate | Final ordinal-benefit teacher |

## Key Principle

The proposed method keeps paired fusion as the default prediction and rolls back
only the ordinal boundaries where the fellow-eye signal is predicted to be
harmful.

The final boundary probability is

```text
q_final,b = (1 - alpha_b) q_f,b + alpha_b q_s,b
```

where `alpha_b = 0` keeps paired fusion and `alpha_b = 1` fully returns to the
single-eye prediction for boundary `b`.
