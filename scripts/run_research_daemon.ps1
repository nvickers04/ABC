<#
.SYNOPSIS
  Start the research daemon only (never __main__.py / Grok trader).

.DESCRIPTION
  Use this as the Program/script for Windows Task Scheduler or Docker host scripts
  so a typo cannot launch the full trading agent.

.EXAMPLE
  .\scripts\run_research_daemon.ps1
  .\scripts\run_research_daemon.ps1 -Verbose
#>
[CmdletBinding()]
param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $python = $venvPy
}
else {
    $python = "python"
}

$pyArgs = @((Join-Path $RepoRoot "research_daemon.py"))
if ($Verbose) {
    $pyArgs += "--verbose"
}

Write-Host "Starting research daemon only: $python $($pyArgs -join ' ')" -ForegroundColor Cyan
& $python @pyArgs
