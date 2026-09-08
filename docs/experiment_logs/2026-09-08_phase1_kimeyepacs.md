# Worklog

## 2026-09-04

- Confirmed local GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB VRAM.
- Confirmed PyTorch environment: `torch 2.11.0+cu128`, CUDA available, BF16 supported.
- Downloaded and verified official EyePACS test labels and images.
- Built official split:
  - train: 35,126 images
  - val: 10,906 images
  - test: 42,670 images
- Created the `kimeyepacs` experiment workspace.
- Added RTX 5070 Ti batch/precision defaults and modular training code.
- Smoke test showed Windows pagefile pressure with 8 DataLoader workers
  (`WinError 1455` while workers import CUDA Torch DLLs). PHASE 1 uses
  `num_workers=2`; smoke uses `num_workers=0`.
