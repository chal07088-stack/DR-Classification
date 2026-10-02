param(
    [string]$Repo = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)),
    [ValidateSet('all','eyepacs','messidor2')]
    [string]$DatasetFilter = 'all'
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv-local-gpu\Scripts\python.exe'
$LogDir = Join-Path $Repo 'outputs\local_preparation'
$CacheLog = Join-Path $LogDir 'cache_preparation.log'
$Log = Join-Path $LogDir 'smoke_queue.log'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$env:PYTHONPATH = Join-Path $Repo 'src'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:DR_NUM_WORKERS = '8'
$env:DR_EVAL_NUM_WORKERS = '8'
$env:DR_BFFN_NUM_WORKERS = '8'

"[$(Get-Date -Format o)] Waiting for train/validation cache preparation" |
    Tee-Object -FilePath $Log -Append
while ($true) {
    $ready = (Test-Path 'outputs\data_audit\drgcnn_eyepacs_cache.json') -and
             (Test-Path 'outputs\data_audit\drgcnn_messidor2_cache.json') -and
             ((Get-Content $CacheLog -Raw -ErrorAction SilentlyContinue) -match 'LOCAL CACHE PREPARATION COMPLETE')
    if ($ready) { break }
    $eye = (Get-ChildItem 'data\cache\drgcnn\eyepacs' -File -Recurse -ErrorAction SilentlyContinue).Count
    $mess = (Get-ChildItem 'data\cache\drgcnn\messidor2' -File -Recurse -ErrorAction SilentlyContinue).Count
    "[$(Get-Date -Format HH:mm:ss)] cache files EyePACS=$eye Messidor2=$mess" |
        Tee-Object -FilePath $Log -Append
    Start-Sleep -Seconds 30
}

$methods = @('huang_single','huang_pff','drgcnn_single','drgcnn_bffn','d_odr')
$datasets = if ($DatasetFilter -eq 'all') { @('eyepacs','messidor2') } else { @($DatasetFilter) }
foreach ($dataset in $datasets) {
    foreach ($method in $methods) {
        "[$(Get-Date -Format o)] SMOKE START $dataset/$method" |
            Tee-Object -FilePath $Log -Append
        # Windows PowerShell 5 wraps any native stderr line (including benign
        # Python warnings) as an ErrorRecord.  Keep stderr in the transcript,
        # but decide success exclusively from the native process exit code.
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $Python -u scripts\train_full_baseline.py --dataset $dataset --method $method --seed 42 --smoke 2>&1 |
            Tee-Object -FilePath $Log -Append
        $nativeExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedErrorActionPreference
        if ($nativeExitCode -ne 0) {
            throw "Smoke failed: $dataset/$method (exit $nativeExitCode)"
        }
    }
}

"[$(Get-Date -Format o)] ALL LOCAL SMOKE TESTS COMPLETE" |
    Tee-Object -FilePath $Log -Append
