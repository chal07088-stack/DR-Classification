$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = "C:\Users\use\anaconda3\envs\tf_dr\python.exe"
$env:PYTHONPATH = Join-Path $Root "src"
& $Python -m kimeyepacs.probe_vram `
  --config (Join-Path $Root "configs\phase1_eyepacs_convnext_s.yaml") `
  --out (Join-Path $Root "experimental_results\ours_proposed\vram_probe.json") `
  --steps 25
