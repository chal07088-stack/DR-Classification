# Experiment Plan

## PHASE 1A: Final Experts and Mandatory Review

EyePACS official train 35,126 eyes; validation 10,906 eyes; test 42,670 eyes.
ConvNeXt-S, ordered-CORAL, RGB 512px, seed 42. Single and paired experts are fully trainable.

1. Validate official patient manifests and the data audit.
2. Complete train-internal single and paired epoch selection, followed by full-train refits.
   This is up to four fits, not a 10% data pilot. Reuse completed matching checkpoints.
3. Save final validation predictions. Do not infer test images in this stage.
4. On validation, cross-fit raw/Platt/isotonic comparison and training-selected specificity
   operating points with five patient folds. These are not five expert training folds.
5. Report overall/gap performance, boundary support, clinical misses at grade >=2 and >=3,
   patient-clustered intervals, binary masks and the finite 625-policy alpha grid.
6. Report raw-feature logistic predictability of local counterfactual benefits and descriptive
   target-grade-adjusted shrinkage associations. Preserve inputs, fitted parameters and predictions.
7. Write diagnostics/phase1a/REPORT.md and exit with awaiting_review. No automatic next stage.

Binary/grid policies use labels and optimize per-eye squared grade error, not population QWK.
Neither is a continuous-gate or QWK upper bound. Sparse threshold events do not exclude calibration
or ranking signal. Do not use a universal +0.005 cutoff. Failure of an exploratory logistic probe
is not proof that no learnable gate exists.

All validation diagnostics are exploratory and conditional on one trained expert pair.
Confidence intervals do not include expert training seeds or refitting calibration in each bootstrap.
The user reviews support, effect size, uncertainty and calibration before deciding on 1B.

## PHASE 1B: OOF, Teachers and Gate Comparison After Review

Explicit command: python scripts/run_phase1.py --stage 1b --reviewed-1a

- Reuse final experts. Generate five patient-fold expert OOF pairs on official train.
- Preserve Teacher A and one-boundary Teacher B targets from expert OOF for diagnostics.
- Fit and cross-fit gates on official validation predictions; final experts never used that split
  for epoch selection. Fit final gates on all validation and evaluate the fixed arms on test.
- Keep G_teacher_b as proposed before test inspection. A-G comparisons are implemented;
  added calibrated gate and hard-switch comparator protocols must be frozen before test access.
- Teacher B replaces one coordinate, projects and decodes, and measures squared-error recovery.
  Joint deployed gate utility remains a separate evaluation; the oracle never supplies its labels.
- Report QWK, clinical boundaries, alpha distributions/collapse, and local-versus-joint effects.

Before 1B test access, record the primary endpoint, operating point, overall safety criterion,
comparison family and any calibration changes. Do not choose a favorable metric after test results.
No numerical noninferiority margin has yet been agreed; report estimated differences and intervals
without claiming formal noninferiority.

## PHASE 2: Reproducibility, Comparators and Mechanism

Requires review of PHASE 1B. These are plans, not implementations or completed experiments.

- Repeat final experts and validation-crossfit gates at seeds 42, 123, 128 under one locked protocol.
  The current gate fits on validation, so repeating train expert OOF for all seeds is optional
  unless needed by the final claims. Folds are not independent training seeds.
- Reproduce Huang et al.'s official two-stage paired feature fusion; distinguish the original
  training protocol from controlled adaptations. Report single-model and ensemble results separately.
- Compare simple prediction-based hard switching, global gates, boundary task-loss, Teacher A/B,
  and calibrated counterparts with identical patient partitions and operating-point selection rules.
- Validate whether improvement survives calibration and threshold tuning. Fit calibration,
  thresholds and gates only on training portions of each cross-fitting split; nest when needed.
- Extend association plots with controlled fellow-input interventions; distinguish model response
  from clinical causality and account for image quality, label uncertainty and out-of-distribution pairs.
- Assess risk-coverage/AURC after defining a deployable rejection score; alpha is intervention
  strength, not automatically confidence. Include baseline scores and uncertainty intervals.
- Predefine confirmatory hypotheses and BH/Holm families; disclose exploratory and negative results.

### PHASE 2B-1 locked Huang matched-backbone comparator

The first comparator run after the three-seed review is a controlled adaptation of Huang et al.'s
paired feature fusion (PFF), not a claim to reproduce the paper's reported ResNet-50 number.

- Reuse each seed's completed single-eye ConvNeXt-S encoder and keep every encoder parameter frozen.
- Concatenate ordered target/fellow features and adapt the official fusion dimensions proportionally:
  `2d -> d -> max-pool/2 -> d/4 -> max-pool/2 -> d/8 -> 1`.
- Match the published/official second-stage optimizer: scalar grade MSE, SGD with momentum 0.9 and
  Nesterov, weight decay 0.0005, constant learning rate 0.02, batch 64, exactly 20 epochs.
- Select the checkpoint by official-validation QWK. Access test predictions once after all 20
  validation evaluations; do not tune the architecture, decoder or optimizer on test.
- Decode with the official half-integer thresholds and report overall, Gap >= 1 and Gap >= 2 QWK
  separately for seeds 42, 123 and 128, followed by mean and sample SD.
- Record the inspected official-code commit and frozen checkpoint SHA-256 values. Keep the original
  ResNet-50 protocol as a separate future arm so its training changes are not conflated with the
  matched-backbone comparison.

Calibration/threshold robustness and intervention-penalty analyses follow this comparator and must
be frozen separately before execution. They are not launched automatically by the Huang runner.

## PHASE 3: External Data and Encoder Generalization

Requires separate authorization after the earlier results are reviewed.

- Messidor-2 offers official image pairing. Audit third-party per-eye grade provenance and licensing
  before external ordinal validation; report complete-pair and grade support counts.
- ODIR-5K offers bilateral images and diagnostic labels, not automatically trustworthy 0-4 DR grades.
  Audit per-eye keywords/labels. Binary disease generalization is a different task and must be labeled so.
- Add a matched-protocol ResNet-50 experiment, then EfficientNet if justified by resources.
  Huang's original ResNet experiment alone does not isolate backbone effects from training changes.

## Contribution and Publication Boundaries

Check the reference pair project's publication/submission status, contributors and overlapping results.
Do not present its harm finding as a new independent discovery. The proposed incremental claim is
boundary-specific diagnosis and mitigation, conditional on evidence. Author contributions and citations
must be agreed before submission. Neither a venue nor novelty is guaranteed by this experiment list.

Primary references: [Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/),
[Messidor-2](https://design.adcis.net/en/third-party/messidor2/),
[ODIR-5K](https://odir2019.grand-challenge.org/dataset/).
