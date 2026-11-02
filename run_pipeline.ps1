# Runs the whole pipeline: installs the locked environment with uv, then hands
# over to scripts/run_pipeline.py. Options are passed through, e.g.
#   .\run_pipeline.ps1 --list
#   .\run_pipeline.ps1 --from forecast
# If scripts are blocked on this machine:
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Install it with:"
    Write-Host "    winget install --id=astral-sh.uv -e"
    Write-Host "then open a new terminal and run this script again."
    exit 1
}

Write-Host "Setting up the Python environment (uv sync)..."
uv sync --frozen --extra dev --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run --frozen python scripts/run_pipeline.py @args
exit $LASTEXITCODE
