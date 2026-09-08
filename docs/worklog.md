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
