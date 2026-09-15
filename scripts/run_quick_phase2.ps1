$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class QuickBenchmarkPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]1
[QuickBenchmarkPower]::SetThreadExecutionState($continuous -bor $systemRequired) | Out-Null
try {
    & .venv\Scripts\python.exe scripts\run_quick_phase2.py
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2 runner exited with code $LASTEXITCODE"
    }
}
finally {
    [QuickBenchmarkPower]::SetThreadExecutionState($continuous) | Out-Null
}
