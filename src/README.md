# Source Package

`src/kimeyepacs/` contains the modular PHASE 1 implementation.

Main modules:

- `data.py`: paired-eye EyePACS dataset loading and transforms
- `models.py`: single-eye and paired ordinal experts
- `engine.py`: training and prediction loops
- `teachers.py`: Loss-Guided and Counterfactual Ordinal teacher construction
- `gates.py`: boundary-wise gate and adaptive mixing
- `metrics.py`: QWK, F1, MAE, accuracy, subgroup metrics
- `run_phase1.py`: full PHASE 1 pipeline
- `probe_vram.py`: RTX memory probe helper
