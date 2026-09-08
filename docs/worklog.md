# Work Log

## 2026-09-08: PDF-based reconstruction

Replaced the earlier `src/kimeyepacs` layout and empty placeholders with runnable
modules for data, models, training, teachers, gates, evaluation, and experiment stages.
Preserved historical PDF references and added the four supplied research PDFs.

Teacher B was explicitly corrected to the user's one-boundary counterfactual U_b
definition. Joint effect and 16-mask oracle remain diagnostic calculations.
Ordered-CORAL experts use a shared scalar score with ordered cutpoints. PAVA remains
necessary after boundary-dependent mixing. Paired encoders are trained end to end.

Fixed evaluated-EMA checkpoint capture, tail accumulation, patient/fellow validation,
fold-local model selection, worker seeds, and portable configuration.
The new code has no compatibility promise for the prior grayscale/frozen-paired run.

Dependencies are locked with uv. Full EyePACS is registered in a local DVC object
store at the user's request, retaining the original data paths via hardlinks.
This local store is not yet a remote server accessible to collaborators.

The v3 gate validation protocol is now distinct from expert OOF. Official validation
is reserved for patient cross-fitted gate fitting; experts select epochs within
their own training subsets and refit before predictions. This incurs 12 pilot jobs
in addition to 12 retained experts and is documented before any future launch.

Validation records are in `experimental_results/repository_validation/`.
User explicitly requested no experiment execution; no real EyePACS training started.

## 2026-09-08: Authorized GPU launch preparation

The user subsequently authorized PHASE 1 training and asked to stop any previous
training. Windows process inspection found no existing Python training process.
The active implementation remains ConvNeXt-S, seed 42, five patient folds only.

Real 512px RGB probes covered single batch 8/16 and paired batch 8, followed by
100 consecutive forward/backward iterations for the selected single/paired batches.
The probes include AdamW state, EMA updates, EMA evaluation, and CPU snapshot;
their temporary learned weights are discarded and never reused for PHASE 1.
Single B16: peak allocated 8.769 GiB, about 79.7 target eyes/s.
Paired B8: peak allocated 8.747 GiB, about 40.4 target eyes/s.
These are short hardware measurements, not performance results or full-run ETAs.

Selected B16 x 2 / B8 x 4 with effective batch 32, BF16, channels-last, six
DataLoader workers, and a 13 GiB PyTorch allocator limit. Deterministic settings,
seeds, learning rates, model/teacher definitions, and split assignments are unchanged.
An explicit CUDA device-index issue found in the first preflight was fixed and
covered with a regression test. All 13 CPU tests pass.

Added a one-shot PHASE 1 runner with a duplicate-run lock and failure status,
per-minute in-job progress/VRAM logs, and Windows system-sleep prevention only
for the lifetime of this process. No scheduled automation or proactive messages
were created. Training errors stop the queue rather than silently changing batches.
The expert -> Teacher A/B/gate -> report queue ends after PHASE 1.

## 2026-09-08: Review corrections and clean restart

Stopped the authorized run during final_single_selection epoch 6 and removed only
its `experimental_results/phase1_convnext_s_seed42_pdfv3` directory as requested.
The data, source, GPU preflight records, and repository verification are preserved.

Renamed pilot_fraction to inner_validation_fraction without shrinking the data or
reducing the epoch budget. The protocol still requires up to 24 expert fits.
Added per-epoch boundary event/recall/ROC-AUC/AP reports and actual epoch data-wait,
throughput, and validation timing. Both final selections now precede refits/OOF;
a B3-silence check can halt the remaining queue for review without automatic
threshold/weight tuning. Local U_b, oracle masks, and joint effects remain distinct.

Run-level provenance already prevented config changes. Added checkpoint-internal
config/source/manifest/training-row/validation-row fingerprints to also reject
copied, incompatible, and unmarked legacy checkpoint files.
Gate inputs now explicitly reload official validation rather than inheriting the
last diagnostics-loop variables. Rare stratum merging/fallback emits warnings.

The exact-byte audit was already complete, but prepare's old false flag remained.
Re-ran label/content-index checks and linked the split audit to the matching report.
All 88,702 content hashes remain unique; near-duplicate detection is not claimed.
No tracked .gitkeep files remain in this repository. Documented the shared-score
proportional-odds constraint and the limits of the internal ordinal baseline.
