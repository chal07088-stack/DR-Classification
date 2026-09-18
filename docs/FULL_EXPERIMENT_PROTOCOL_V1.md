# Full experiment protocol v1

Status: design freeze candidate. No full-v1 training has started.

## Scope

The study compares four published approaches only:

1. Huang et al. 2023 paired feature fusion (bilateral).
2. DRGCNN 2024 bilateral feature fusion (bilateral).
3. D-ODR (single-eye ordinal regression).
4. AOR-DR 2025 autoregressive ordinal diffusion (single-eye).

The earlier ConvNeXt/proposed V0–V5 experiments and CLIP-DR are excluded from the primary
full-v1 comparison. Single-eye and bilateral methods share data partitions and evaluation rules,
but retain their paper-specific model, loss, optimizer and preprocessing unless an adaptation is
required for the available data or current software stack.

## Locked data populations

### EyePACS full

| Partition | Bilateral patients | Eyes |
|---|---:|---:|
| Train | 17,563 | 35,126 |
| Validation | 5,453 | 10,906 |
| Test | 21,335 | 42,670 |

The existing audit reports patient-disjoint partitions, complete pairs, unique byte hashes and no
cross-split exact duplicates. Near-duplicate detection remains a documented limitation and should
be completed before the final confirmatory run if feasible.

### Messidor-2 complete graded pairs

| Partition | Bilateral examinations | Eyes |
|---|---:|---:|
| Train | 609 | 1,218 |
| Validation | 131 | 262 |
| Test | 130 | 260 |
| Full external set | 870 | 1,740 |

Only exact-ID matched, gradable eyes are used. The four incomplete official examinations are
excluded from bilateral analyses. The preserved canonical complete-pair manifest is
`data/splits/messidor_patients.csv`; deleted historical audit outputs are not runtime inputs.

## Evaluation tracks and required order

1. **EyePACS in-domain:** train on EyePACS train, select on EyePACS validation, evaluate EyePACS
   test once after the method/seed configuration is frozen.
2. **Messidor external transfer:** use the EyePACS-selected checkpoint without Messidor fitting or
   threshold tuning; evaluate all 870 complete Messidor examinations. This track must run before
   any Messidor in-domain tuning.
3. **Messidor in-domain:** train on the fixed 609-patient split, select on 131 validation patients,
   evaluate the 130-patient test split once.
4. **Joint multi-source:** train on EyePACS train plus Messidor train using dataset-balanced
   sampling; select from separately reported validation metrics using the predeclared unweighted
   mean of the two domain QWK values; evaluate each domain's test partition separately.

The joint setting is a **protocol-adapted multi-source experiment** unless the exact paper already
defines it. It must never be described as an original-paper reproduction.

## Sample construction

- A complete bilateral patient produces two directional rows: left target/right fellow and right
  target/left fellow.
- Huang and DRGCNN receive both target and same-patient fellow images.
- D-ODR and AOR-DR receive only the target image; the fellow image is retained only for matched
  patient-level analysis.
- Random image-level splitting is forbidden. All sampling, folds and bootstrap resampling are
  patient/examination clustered.
- Left/right mirroring or swapping is allowed only when semantically correct and must be identical
  across comparable arms.

## Seeds and model selection

- Required seeds: 42, 123 and 128.
- Each method is trained independently at every seed and track.
- Best epoch is selected only by validation QWK. Exact ties use lower validation MAE, then the
  earlier epoch.
- Test predictions are generated once from the selected checkpoint.
- No test-set hyperparameter changes, checkpoint reselection or threshold fitting are allowed.
- Huang's single-eye encoder stage and paired fusion stage are both part of the seed-specific run.

## Reproduction labels

Every run must be classified in its provenance as one of:

- `official_reproduction`: official code/data protocol can be followed exactly.
- `compatibility_adapted_reproduction`: only software/runtime compatibility changed.
- `protocol_adapted_reproduction`: data, split, joint-training or schedule differs materially.

All deviations must be listed before test inference. Matched-backbone variants and ensembles, if
added later, are secondary experiments and cannot replace each method's primary official-style arm.

## Metrics

Primary metric:

- Per-eye quadratic weighted kappa (QWK), reported per seed and as mean ± sample SD across seeds.

Secondary metrics:

- MAE, exact accuracy and macro F1.
- Per-grade support, recall and confusion matrix.
- Sensitivity/specificity for referable DR (grade >= 2) and severe/vision-threatening threshold
  (grade >= 3), with the exact naming kept consistent with the source label definition.
- Patient-clustered 95% bootstrap confidence intervals.
- Bilateral subgroup results for grade gap 0, 1 and >=2.
- Symmetric versus asymmetric patient results.

Benefit/harm relative to a target-only prediction is reported only for methods that have a valid,
matched target-only counterpart. It is diagnostic, not a replacement for the primary QWK endpoint.

## Joint-training balance

- Each optimizer step uses equal expected contribution from EyePACS and Messidor train data.
- Dataset balancing is separate from any paper-specific class balancing.
- Validation and test metrics are always computed per domain; pooled sample-count-weighted metrics
  are supplementary only.
- Batch composition, oversampling rate and repeated Messidor exposure are logged explicitly.

## Execution contract

The server root is `/home/minjae/DR-Classification`, using NVIDIA GPU 0. Long jobs run in a named
`tmux` session and write line-buffered console output so epoch, validation and finish markers are
visible with `tmux attach`.

One orchestrator owns the full state machine:

```text
DATA_AUDIT -> ENV_SMOKE -> METHOD_SMOKE -> EYEPACS_TRAIN
-> MESSIDOR_EXTERNAL -> MESSIDOR_IN_DOMAIN -> JOINT_MULTISOURCE
-> AGGREGATE -> COMPLETE
```

- A successful stage writes `COMPLETED.json` atomically and automatically launches the next stage.
- A failed stage writes `FAILED.json`, preserves the command/error/last checkpoint, and stops.
- Restart resumes from the first incomplete stage; completed stages are never silently rerun.
- Every epoch prints method, dataset, seed, epoch/max_epoch, train loss, validation metrics, elapsed
  time and ETA. Every run prints an explicit `FINISHED` or `FAILED` marker.
- Logs live beside each run and are also linked from `experimental_results/full_v1/aggregate/STATUS.md`.

## Artifact contract

Each method/seed directory must contain:

```text
config.lock.json
provenance.json
environment.txt
train.log
history.csv
best_checkpoint.json
metrics_val.json
metrics_test.json
predictions_val.parquet
predictions_test.parquet
confusion_matrix.csv
COMPLETED.json or FAILED.json
```

Large checkpoints, predictions and features are DVC artifacts. Git contains only the small audit,
configuration, metric and report files. Historical pilot directories are read-only inputs and must
never be overwritten by full-v1.

## Pre-launch gates

Full training starts only after all of the following pass:

1. Dataset path, count, exact-ID, label-range and patient-overlap audits.
2. Upstream commit and local patch audit for all four methods.
3. Fresh environment export and GPU smoke forward/backward for every method.
4. One short end-to-end smoke run that saves and reloads a checkpoint and predictions.
5. Configuration hashes and the full-v1 manifest are frozen before test access.
