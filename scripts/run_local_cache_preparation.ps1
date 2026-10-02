param(
    [string]$Repo = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)),
    [int]$CacheWorkers = 24
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv-local-gpu\Scripts\python.exe'
$LogDir = Join-Path $Repo 'outputs\local_preparation'
$Log = Join-Path $LogDir 'cache_preparation.log'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$env:PYTHONPATH = Join-Path $Repo 'src'

"[$(Get-Date -Format o)] LOCAL PREPARATION START" | Tee-Object -FilePath $Log
"Repo: $Repo" | Tee-Object -FilePath $Log -Append
"Python: $Python" | Tee-Object -FilePath $Log -Append
"Workers: $CacheWorkers" | Tee-Object -FilePath $Log -Append

& $Python -u scripts\prepare_drgcnn_cache.py --dataset eyepacs --workers $CacheWorkers --splits train val 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "EyePACS cache failed with exit code $LASTEXITCODE" }

& $Python -u scripts\prepare_drgcnn_cache.py --dataset messidor2 --workers $CacheWorkers --splits train 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "Messidor2 cache failed with exit code $LASTEXITCODE" }

"[$(Get-Date -Format o)] LOCAL CACHE PREPARATION COMPLETE" |
    Tee-Object -FilePath $Log -Append
