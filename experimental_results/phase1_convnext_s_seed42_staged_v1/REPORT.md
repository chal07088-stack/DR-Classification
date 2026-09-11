# PHASE 1 result report

ConvNeXt-S, seed 42, patient-level five-fold OOF.

Proposed arm fixed before test: G_teacher_b (one-boundary Teacher B).

| Arm | Overall QWK | Gap >= 1 QWK | Gap >= 2 QWK |
| --- | --- | --- | --- |
| A_single | 0.8298 | 0.6478 | 0.5857 |
| B_paired | 0.8462 | 0.6260 | 0.5393 |
| C_fixed50 | 0.8410 | 0.6439 | 0.5791 |
| D_global_learned | 0.8483 | 0.6347 | 0.5511 |
| E_boundary_task | 0.8485 | 0.6385 | 0.5623 |
| F_teacher_a | 0.8465 | 0.6339 | 0.5571 |
| G_teacher_b | 0.8481 | 0.6416 | 0.5656 |
| global_asymmetry | 0.8456 | 0.6327 | 0.5528 |
| entropy_reliability | 0.8410 | 0.6439 | 0.5791 |

![QWK by gap](figures/qwk_by_gap.png)

Patient bootstrap intervals quantify sampling uncertainty, not variation across training seeds.
Teacher B labels are local U_b; joint deployed utility is reported separately.
The 16-mask oracle is a diagnostic discrete policy class, not a learned method or a continuous-gate upper bound.
Locality slopes are descriptive and prevalence-adjusted; they do not prove a causal mechanism.
No external comparator paper has been reproduced by the generic reliability baseline.
PHASE 1 ends here. PHASE 2/3 require separate decisions.
