$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = "C:\Users\use\anaconda3\envs\tf_dr\python.exe"
$env:PYTHONPATH = Join-Path $Root "src"
Set-Location $Root
& $Python -m kimeyepacs.run_phase1 `
  --config (Join-Path $Root "configs\phase1_eyepacs_convnext_s.yaml") `
  --seed 42 `
  --run-name phase1_smoke `
  --folds 2 `
  --image-size 128 `
  --epochs 1 `
  --paired-epochs 1 `
  --gate-epochs 2 `
  --max-train-patients 40 `
  --max-val-patients 20 `
  --max-test-patients 20 `
  --pretrained false `
  --num-workers 0
