$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class QuickBenchmarkPhase3Power {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]1
[QuickBenchmarkPhase3Power]::SetThreadExecutionState($continuous -bor $systemRequired) | Out-Null
try {
    & .venv\Scripts\python.exe scripts\run_quick_phase3.py
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 3 runner exited with code $LASTEXITCODE"
    }
}
finally {
    [QuickBenchmarkPhase3Power]::SetThreadExecutionState($continuous) | Out-Null
}
