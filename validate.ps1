param(
    [string]$EnvUrl = "http://localhost:7860"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host "ClinicalTrialEnv Pre-Submission Checks"
Write-Host "========================================"
Write-Host ""

Write-Host "1. Running test suite..."
uv run --python .venv\Scripts\python.exe pytest tests -q

Write-Host ""
Write-Host "2. Running OpenEnv validator..."
uv run --python .venv\Scripts\python.exe openenv validate

Write-Host ""
Write-Host "3. Checking local /reset endpoint..."
$body = "{}"
$response = Invoke-RestMethod -Uri "$EnvUrl/reset" -Method Post -ContentType "application/json" -Body $body
if (-not $response.session_id) {
    throw "Expected session_id in /reset response."
}
Write-Host "Local API check passed."

Write-Host ""
Write-Host "4. Checking Docker daemon..."
docker info | Out-Null
Write-Host "Docker daemon reachable."

Write-Host ""
Write-Host "5. Building Docker image..."
docker build -t clinicaltrial-env .

Write-Host ""
Write-Host "All local pre-submission checks passed."

