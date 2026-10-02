param(
    [string]$Repo = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)),
    [string]$RunRoot = ""
)

$ErrorActionPreference = 'Stop'
if (-not $RunRoot) {
    $RunRoot = Join-Path $Repo 'idea_work\results\FINAL_SOFTUTILITY_FULL_SEED42_LOCAL_03'
}
$Python = Join-Path $Repo '.venv-local-gpu\Scripts\python.exe'
$IdeaRoot = Join-Path $Repo 'idea_work'
$StrictRunner = Join-Path $Repo 'run_final_softutility_full_seed42_strict.py'
$BaseRunner = Join-Path $Repo 'scripts\run_final_softutility_full_seed42.py'

Set-Location $Repo
$env:CUDA_VISIBLE_DEVICES = '0'
$env:DR_PHYSICAL_GPU = '0'
$env:DR_EXPECTED_GPU_NAME = 'RTX 5070 Ti'
$env:DR_BASELINE_ROOT = $Repo
$env:DR_IDEA_ROOT = $IdeaRoot
$env:DR_IDEA_OUTPUT = $RunRoot
$env:DR_FINAL_BASE_RUNNER = $BaseRunner
$env:PYTHONPATH = $IdeaRoot

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$PID | Set-Content -Path (Join-Path $RunRoot 'worker_pid.txt') -Encoding ascii
try {
    & $Python -u $StrictRunner --run-root $RunRoot
    if ($LASTEXITCODE -ne 0) {
        throw "OUR IDEA runner exited with code $LASTEXITCODE"
    }
    "COMPLETE $(Get-Date -Format o)" | Set-Content -Path (Join-Path $RunRoot 'worker_status.txt')
}
catch {
    "FAILED $(Get-Date -Format o) $($_.Exception.Message)" |
        Set-Content -Path (Join-Path $RunRoot 'worker_status.txt')
    throw
}
