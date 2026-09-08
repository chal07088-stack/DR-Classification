# Experiment Plan

## PHASE 1

EyePACS official split: 35,126 train / 10,906 validation / 42,670 test eyes.
ConvNeXt-S, ordered-CORAL, seed 42, five patient folds.

1. Validate identities, paired labels, split disjointness, and official counts.
2. Fit single and paired experts under the same optimization and RGB protocol.
3. Generate complete patient-level train OOF predictions. Select epochs inside each
   training fold, then refit that fold; never select with outer holdout labels.
4. Generate Teacher A and one-boundary Teacher B diagnostics from expert OOF dumps.
5. Fit and cross-fit gates on official validation predictions from final experts.
   Official validation was not used to choose expert checkpoints.
6. Compare fixed A-G arms plus internal scalar-asymmetry and entropy baselines.
7. Report overall, gap 0/1/>=1/>=2, boundary metrics, alpha distributions,
   clustered bootstrap intervals, and local teacher versus joint deployed utility.
8. Produce the PHASE 1 report and stop for review.

Current: source/data/environment reconstruction. No new research training launched.
G_teacher_b is the proposed arm fixed before test inspection.

## PHASE 2 (After Review)

Repeat the locked protocol at training seeds 42, 123, 128 with five patient folds.
Report seed mean and sample SD, effect sign across seeds, and patient bootstrap CI
separately; folds are not independent training seeds.

Add external fusion comparator implementations only after inspecting their official
code and fixing the adaptation protocol. Candidates in the supplied comparator PDF:
Multimodal Dynamics, uncertainty suppression, and bilateral medical fusion.
The current generic entropy baseline is not an official reproduction of any paper.

Pre-register threshold/calibration sensitivity, the overall noninferiority margin,
boundary hypothesis families, and confirmatory versus exploratory analyses before
additional test evaluation. Locality regression in PHASE 1 is descriptive.
Within-patient boundary permutation needs a defensible exchangeability null;
unrestricted shuffling of nonexchangeable severity boundaries is not confirmatory.

## PHASE 3 (After Protocol/Paper Review)

External paired datasets and additional encoder families (ResNet, EfficientNet, ViT).
Verify subject keys, per-eye severity label provenance, and boundary event counts first.
ODIR-5K diagnostic labels alone are not automatically a 0-4 DR grading ground truth;
the reference project's derived labels require a separate audit.

Assess boundary-level power before strong external claims. Report sparse or negative
findings. Fuzzy gates, CORN, additional domains, and expensive robustness are optional
extensions with their own explicit experiment decisions.

The two-paper suggestion in the PDFs is a possible publication strategy, not an
automatic execution instruction or an assertion that two contributions are established.
