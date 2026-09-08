# Configs

`phase1_eyepacs_convnext_s.yaml` is the active PHASE 1 configuration.

Important fields:

- `data_root`: local EyePACS official split path
- `out_root`: repository-local experiment output root
- `model.arch`: ConvNeXt-S backbone
- `model.image_size`: 512
- `optimization`: batch size, accumulation, epochs, learning rates, EMA, and gate settings

Raw data paths are intentionally machine-local and should be edited when the
repository is cloned on another workstation or server.
