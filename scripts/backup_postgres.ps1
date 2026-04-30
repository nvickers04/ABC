param(
    [string]$RepoRoot = "C:\Users\nvick\Documents\GitHub\ABC",
    [int]$RetentionDays = 14
)

$ErrorActionPreference = "Stop"

function Get-EnvMap {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Missing env file: $Path"
    }
    $map = @{}
    foreach ($line in Get-Content $Path) {
        $s = $line.Trim()
        if ($s -eq "" -or $s.StartsWith("#")) { continue }
        $parts = $s -split "=", 2
        if ($parts.Count -ne 2) { continue }
        $map[$parts[0].Trim()] = $parts[1].Trim()
    }
    return $map
}

$infraEnvPath = Join-Path $RepoRoot "infra\postgres\.env"
$backupDir = Join-Path $RepoRoot "backups\postgres"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

$envMap = Get-EnvMap -Path $infraEnvPath
$required = @("POSTGRES_DB", "POSTGRES_SUPERUSER", "POSTGRES_SUPERPASS")
foreach ($k in $required) {
    if (-not $envMap.ContainsKey($k) -or [string]::IsNullOrWhiteSpace($envMap[$k])) {
        throw "Required key missing in infra env: $k"
    }
}

$db = $envMap["POSTGRES_DB"]
$superUser = $envMap["POSTGRES_SUPERUSER"]
$superPass = $envMap["POSTGRES_SUPERPASS"]
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$fileName = "${db}_${timestamp}.dump"
$hostPath = Join-Path $backupDir $fileName
$containerPath = "/tmp/$fileName"

Write-Host "Creating backup for database '$db'..."

docker exec -e "PGPASSWORD=$superPass" abc-postgres pg_dump -U $superUser -d $db -Fc -f $containerPath
if ($LASTEXITCODE -ne 0) {
    throw "pg_dump failed with exit code $LASTEXITCODE"
}

docker cp "abc-postgres:$containerPath" $hostPath
if ($LASTEXITCODE -ne 0) {
    throw "docker cp failed with exit code $LASTEXITCODE"
}

docker exec abc-postgres rm -f $containerPath | Out-Null

Write-Host "Backup written: $hostPath"

# Retention cleanup
$cutoff = (Get-Date).AddDays(-$RetentionDays)
Get-ChildItem -Path $backupDir -File -Filter "*.dump" |
    Where-Object { $_.LastWriteTime -lt $cutoff } |
    ForEach-Object {
        Write-Host "Removing old backup: $($_.FullName)"
        Remove-Item $_.FullName -Force
    }

Write-Host "Backup complete."
