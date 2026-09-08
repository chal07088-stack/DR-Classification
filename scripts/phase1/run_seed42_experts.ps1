$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = "C:\Users\use\anaconda3\envs\tf_dr\python.exe"
$env:PYTHONPATH = Join-Path $Root "src"
Set-Location $Root
& $Python -m kimeyepacs.run_experts `
  --config (Join-Path $Root "configs\phase1_eyepacs_convnext_s.yaml") `
  --seed 42 `
  --run-name seed42_convnext_s_experts
