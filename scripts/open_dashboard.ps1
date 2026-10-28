param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$url = 'http://127.0.0.1:8765/'
function Test-Gg49Dashboard {
    try {
        $rows = @(Invoke-RestMethod -Uri ($url + 'api/runs') -TimeoutSec 3)
        $ids = @($rows | ForEach-Object { $_.id })
        return $null -ne $rows
    } catch { return $false }
}
try {
    if (-not (Test-Path -LiteralPath 'D:\mms_experiments\gg49-lanesegnet-lab_sj\runs\pipeline-status.json')) {
        throw 'Connect the D: drive containing the saved GG49 experiments, then retry.'
    }
    if (-not (Test-Gg49Dashboard)) {
        & wsl.exe -d Ubuntu -- mountpoint -q /mnt/d
        if ($LASTEXITCODE -ne 0) {
            & wsl.exe -d Ubuntu -u root -- mkdir -p /mnt/d
            if ($LASTEXITCODE -ne 0) { throw 'Cannot prepare the WSL mount point.' }
            & wsl.exe -d Ubuntu -u root -- mount -t drvfs D: /mnt/d -o metadata,uid=1000,gid=1000
            if ($LASTEXITCODE -ne 0) { throw 'Cannot mount D: in WSL.' }
        }
        & wsl.exe -d Ubuntu -- bash /home/rjstns23/gg49-lanesegnet-lab_sj/scripts/ensure_dashboard.sh
        if ($LASTEXITCODE -ne 0) { throw 'Dashboard startup failed; see D:\mms_experiments\gg49-lanesegnet-lab_sj\logs\dashboard.log' }
        $ready = $false
        for ($attempt = 0; $attempt -lt 15; $attempt++) {
            if (Test-Gg49Dashboard) { $ready = $true; break }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) { throw 'Dashboard started but the Windows localhost connection is not ready.' }
    }
    Write-Host ('GG49 dashboard ready: ' + $url)
    if (-not $NoBrowser) { Start-Process $url }
} catch {
    Write-Error $_
    exit 1
}
