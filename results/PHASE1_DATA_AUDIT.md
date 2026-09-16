# PHASE 1 — Data Audit

Split version: `quick20_symmetry_preserved_v2`

EyePACS Quick20 was generated using patient-level stratified random sampling that preserves the Symmetric (`|L-R|=0`) vs Asymmetric (`|L-R|>=1`) ratio of the complete-pair EyePACS population.

| Split | Patients | Symmetric N | Symmetric % | Asymmetric N | Asymmetric % | Gap=1 N | Gap>=2 N |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full EyePACS complete-pair population | 17563 | 15323 | 87.2459 | 2240 | 12.7541 | 1485 | 755 |
| Train | 1756 | 1532 | 87.2437 | 224 | 12.7563 | 150 | 74 |
| Validation | 527 | 460 | 87.2865 | 67 | 12.7135 | 45 | 22 |
| Test | 1229 | 1072 | 87.2254 | 157 | 12.7746 | 104 | 53 |
| Combined Quick20 | 3512 | 3064 | 87.2437 | 448 | 12.7563 | 299 | 149 |

- Patient leakage: **0**
- Image leakage: **0**
- EyePACS hard gate: **PASS**

## Messidor-2 exact-ID audit

- Status: **BLOCKED**
- Grade rows: 1748
- Official examinations: None
- Complete audited pairs: None
- Reason: Official ADCIS pairing spreadsheet and licensed image archive are not present locally. No pair count was inferred or fabricated.

No model training is permitted until both EyePACS and Messidor-2 audits pass.
