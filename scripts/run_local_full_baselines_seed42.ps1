param(
    [string]$Repo = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)),
    [int]$Workers = 8,
    [int]$CacheWorkers = 24
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv-local-gpu\Scripts\python.exe'
$LogDir = Join-Path $Repo 'outputs\local_full_seed42\logs'
$MasterLog = Join-Path $LogDir 'master.log'
$Methods = @('huang_single','huang_pff','drgcnn_single','drgcnn_bffn','d_odr')
$Datasets = @('eyepacs','messidor2')

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$env:PYTHONPATH = Join-Path $Repo 'src'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:DR_NUM_WORKERS = [string]$Workers
$env:DR_EVAL_NUM_WORKERS = [string]$Workers
$env:DR_BFFN_NUM_WORKERS = [string]$Workers

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python -u @Arguments 2>&1 | Tee-Object -FilePath $MasterLog -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($code -ne 0) {
        throw "Python command failed with exit code ${code}: $($Arguments -join ' ')"
    }
}

"[$(Get-Date -Format o)] PHASE 1 TRAIN/VALIDATION ONLY; OFFICIAL TEST LOCKED" |
    Tee-Object -FilePath $MasterLog -Append
foreach ($dataset in $Datasets) {
    foreach ($method in $Methods) {
        $trained = Join-Path $Repo "outputs\status\${dataset}_${method}_seed42.trained"
        if (Test-Path $trained) {
            "SKIP existing training freeze candidate $dataset/$method" |
                Tee-Object -FilePath $MasterLog -Append
            continue
        }
        Invoke-CheckedPython @(
            'scripts\train_full_baseline.py',
            '--dataset', $dataset,
            '--method', $method,
            '--seed', '42'
        )
    }
}

"[$(Get-Date -Format o)] PHASE 2 GLOBAL CHECKPOINT FREEZE" |
    Tee-Object -FilePath $MasterLog -Append
Invoke-CheckedPython @('scripts\freeze_full_baseline_checkpoints.py', '--seed', '42')

"[$(Get-Date -Format o)] PHASE 3 POST-FREEZE TEST CACHE" |
    Tee-Object -FilePath $MasterLog -Append
Invoke-CheckedPython @(
    'scripts\prepare_drgcnn_cache.py', '--dataset', 'eyepacs',
    '--workers', [string]$CacheWorkers, '--splits', 'test'
)
Invoke-CheckedPython @(
    'scripts\prepare_drgcnn_cache.py', '--dataset', 'messidor2',
    '--workers', [string]$CacheWorkers, '--splits', 'test'
)

"[$(Get-Date -Format o)] PHASE 4 ONE-SHOT OFFICIAL TEST" |
    Tee-Object -FilePath $MasterLog -Append
foreach ($dataset in $Datasets) {
    foreach ($method in $Methods) {
        $done = Join-Path $Repo "outputs\status\${dataset}_${method}_seed42.done"
        if (Test-Path $done) {
            "SKIP completed one-shot test $dataset/$method" |
                Tee-Object -FilePath $MasterLog -Append
            continue
        }
        Invoke-CheckedPython @(
            'scripts\evaluate_frozen_full_baseline.py',
            '--dataset', $dataset,
            '--method', $method,
            '--seed', '42'
        )
    }
}

"[$(Get-Date -Format o)] FULL BASELINE SEED42 COMPLETE" |
    Tee-Object -FilePath $MasterLog -Append
