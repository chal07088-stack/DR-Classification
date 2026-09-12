# PHASE 2A multi-seed replication

Locked EyePACS / ConvNeXt-S protocol. Values are mean +/- sample SD across training seeds.
Patient bootstrap intervals remain in each seed report and are not substitutes for seed SD.

| Arm | Overall QWK | Gap >= 1 QWK | Gap >= 2 QWK |
| --- | ---: | ---: | ---: |
| A_single | 0.8282 +/- 0.0024 | 0.6463 +/- 0.0043 | 0.5837 +/- 0.0065 |
| B_paired | 0.8447 +/- 0.0032 | 0.6254 +/- 0.0091 | 0.5376 +/- 0.0112 |
| C_fixed50 | 0.8386 +/- 0.0032 | 0.6408 +/- 0.0063 | 0.5722 +/- 0.0097 |
| D_global_learned | 0.8466 +/- 0.0021 | 0.6328 +/- 0.0028 | 0.5474 +/- 0.0035 |
| E_boundary_task | 0.8479 +/- 0.0024 | 0.6383 +/- 0.0037 | 0.5602 +/- 0.0051 |
| F_teacher_a | 0.8447 +/- 0.0032 | 0.6326 +/- 0.0070 | 0.5516 +/- 0.0093 |
| G_teacher_b | 0.8462 +/- 0.0029 | 0.6394 +/- 0.0058 | 0.5627 +/- 0.0092 |
| global_asymmetry | 0.8442 +/- 0.0029 | 0.6307 +/- 0.0070 | 0.5460 +/- 0.0095 |
| entropy_reliability | 0.8386 +/- 0.0032 | 0.6408 +/- 0.0063 | 0.5722 +/- 0.0097 |

Proposed arm remained fixed as `G_teacher_b` before the replication runs.
Train-expert OOF was not repeated because the deployed gates are fitted by patient-level
cross-fitting on official validation predictions. Final experts and gates were retrained per seed.
