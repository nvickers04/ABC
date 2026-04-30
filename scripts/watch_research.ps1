param(
    [string]$RepoRoot = "C:\Users\nvick\Documents\GitHub\ABC",
    [switch]$NoTail
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

Write-Host "== Research daemon process =="
wmic process where "name='python3.11.exe' and commandline like '%research_daemon.py%'" get ProcessId,CommandLine

Write-Host ""
Write-Host "== Latest heartbeat (daemon_heartbeat_ts) =="
$heartbeatCode = @'
import memory
db = memory.get_db()
r = db.execute("SELECT value FROM research_config WHERE key = ?", ("daemon_heartbeat_ts",)).fetchone()
print(r["value"] if r else "no heartbeat")
'@
$heartbeatCode | python -

Write-Host ""
if (-not $NoTail) {
    Write-Host "== Live log tail (Ctrl+C to stop) =="
    Get-Content ".\logs\research_daemon.log" -Wait -Tail 80
}
