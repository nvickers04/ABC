param(
    # Repo root (folder that contains logs\, memory\, etc.). Defaults to parent of this script.
    [string]$RepoRoot = "",
    [switch]$NoTail
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

Set-Location -LiteralPath $RepoRoot

function Get-PythonExe {
    foreach ($cmd in @(
            @{ Name = "py"; Args = @("-3") },
            @{ Name = "python"; Args = @() },
            @{ Name = "python3"; Args = @() }
        )) {
        $exe = Get-Command $cmd.Name -ErrorAction SilentlyContinue
        if ($exe) {
            return @{ Exe = $exe.Source; PrefixArgs = $cmd.Args }
        }
    }
    return $null
}

Write-Host "== Research host process (python -m research) =="
$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match '^python(3(\.\d+)?)?\.exe$' -and
        $_.CommandLine -and
        ($_.CommandLine -match '-m\s+research\b')
    }
if (-not $procs) {
    Write-Host "(none found - process may use a different Python executable name)"
}
else {
    $procs | Select-Object ProcessId, CommandLine | Format-Table -AutoSize
}

Write-Host ""
Write-Host "== Latest heartbeat (daemon_heartbeat_ts) =="
$py = Get-PythonExe
if (-not $py) {
    throw "Could not find Python on PATH. Install Python or use 'py' launcher, then retry."
}
$heartbeatCode = @'
import memory
db = memory.get_db()
r = db.execute("SELECT value FROM research_config WHERE key = ?", ("daemon_heartbeat_ts",)).fetchone()
print(r["value"] if r else "no heartbeat")
'@
if ($py.PrefixArgs.Count -gt 0) {
    $heartbeatCode | & $py.Exe @($py.PrefixArgs + @("-")) | Write-Host
}
else {
    $heartbeatCode | & $py.Exe - | Write-Host
}

Write-Host ""
if (-not $NoTail) {
    $logPath = Join-Path $RepoRoot "logs\research.log"
    if (-not (Test-Path -LiteralPath $logPath)) {
        throw "Log file not found: $logPath"
    }
    Write-Host "== Live log tail (Ctrl+C to stop) =="
    Get-Content -LiteralPath $logPath -Wait -Tail 80
}
