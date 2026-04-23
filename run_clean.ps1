$ErrorActionPreference = "Stop"

$ports = @(8000, 8501)
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Stop-ProcessOnPort {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "Port $Port is free."
        return
    }

    $targetProcessIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($TargetPID in $targetProcessIds) {
        if ($TargetPID -and $TargetPID -ne 0) {
            $proc = Get-Process -Id $TargetPID -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "Stopping PID $TargetPID ($($proc.ProcessName)) on port $Port"
                Stop-Process -Id $TargetPID -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

foreach ($port in $ports) {
    Stop-ProcessOnPort -Port $port
}

Set-Location $projectRoot

$fastApiArgs = @(
    "-m", "uvicorn",
    "app.main:app",
    "--host", "127.0.0.1",
    "--port", "8000",
    "--reload"
)

$streamlitArgs = @(
    "run", "Home.py",
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.address", "127.0.0.1"
)

Write-Host "Starting FastAPI on 127.0.0.1:8000"
Start-Process -FilePath "python" -ArgumentList $fastApiArgs -WorkingDirectory $projectRoot

Write-Host "Starting Streamlit on 127.0.0.1:8501"
Start-Process -FilePath "streamlit" -ArgumentList $streamlitArgs -WorkingDirectory $projectRoot

Write-Host "Safe start complete."
