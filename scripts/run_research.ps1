<#
.SYNOPSIS
  Start the research host process only (never __main__.py / Grok trader).

.DESCRIPTION
  Runs ``python -m research`` from the repo root. Use for Task Scheduler or
  manual starts so a typo cannot launch the trading agent.

.PARAMETER Verbose
  Pass ``--verbose`` to the research host (DEBUG logging).

.PARAMETER NoEvolution
  Pass ``--no-evolution`` (scoring only, no template-evolution thread).

.EXAMPLE
  .\scripts\run_research.ps1
  .\scripts\run_research.ps1 -Verbose
  .\scripts\run_research.ps1 -Verbose -NoEvolution
#>
[CmdletBinding()]
param(
    [switch]$Verbose,
    [switch]$NoEvolution
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

$pyArgs = @("-m", "research")
if ($Verbose) {
    $pyArgs += "--verbose"
}
if ($NoEvolution) {
    $pyArgs += "--no-evolution"
}

Write-Host "Starting research host: $python $($pyArgs -join ' ')" -ForegroundColor Cyan
& $python @pyArgs
