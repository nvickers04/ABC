# One-click deploy wrapper (repo root or infra/).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$DeployPy = Join-Path $PSScriptRoot "deploy.py"
Set-Location $Root
python $DeployPy @args
exit $LASTEXITCODE
