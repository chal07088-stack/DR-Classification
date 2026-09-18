$ErrorActionPreference = 'Stop'

$server = 'minjae@210.94.194.144'
$port = '18181'
$key = Join-Path $env:USERPROFILE '.ssh\id_ed25519_dodr'
$knownHosts = Join-Path $env:USERPROFILE '.ssh\known_hosts_dodr'
$repoRoot = Split-Path -Parent $PSScriptRoot
$eyeParent = Join-Path $repoRoot 'data'
$messidorParent = 'C:\Users\use\Desktop\messidor'
$remoteRoot = '/home/minjae/DR-Classification'
$sshArgs = @('-p', $port, '-i', $key, '-o', "UserKnownHostsFile=$knownHosts", '-o', 'BatchMode=yes', $server)

function Send-TarTree {
    param(
        [Parameter(Mandatory)] [string] $SourceParent,
        [Parameter(Mandatory)] [string[]] $Items,
        [Parameter(Mandatory)] [string] $RemoteDestination,
        [Parameter(Mandatory)] [string] $Label
    )
    Write-Output "[$(Get-Date -Format o)] START $Label"
    & ssh.exe @sshArgs "mkdir -p '$RemoteDestination'"
    if ($LASTEXITCODE -ne 0) { throw "Failed to create $RemoteDestination" }
    & tar.exe -cf - -C $SourceParent @Items | & ssh.exe @sshArgs "tar -xf - -C '$RemoteDestination'"
    if ($LASTEXITCODE -ne 0) { throw "$Label transfer failed with exit $LASTEXITCODE" }
    Write-Output "[$(Get-Date -Format o)] FINISHED $Label"
}

Send-TarTree -SourceParent $eyeParent -Items @('eyepacs') -RemoteDestination "$remoteRoot/data" -Label 'EyePACS'
Send-TarTree -SourceParent $messidorParent -Items @('IMAGES', 'messidor_data.csv', 'messidor-2.csv', 'messidor_readme.txt') -RemoteDestination "$remoteRoot/data/messidor2" -Label 'Messidor-2'

& ssh.exe @sshArgs "touch '$remoteRoot/data/.transfer_complete'"
if ($LASTEXITCODE -ne 0) { throw 'Could not write transfer completion marker' }
Write-Output "[$(Get-Date -Format o)] ALL DATA TRANSFERS FINISHED"
