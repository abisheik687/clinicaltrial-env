param(
    [string]$RepoUrl = "https://github.com/abisheik687/clinicaltrial-env.git",
    [string]$Flavor = "t4-small",
    [string]$Timeout = "2h",
    [int]$SignalSteps = 100,
    [int]$LongSteps = 500
)

$ErrorActionPreference = "Stop"

hf auth whoami | Out-Null

$bash = @"
set -euo pipefail
git clone $RepoUrl /workspace/ClinicalTrialEnv
cd /workspace/ClinicalTrialEnv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-training.txt
python training/hf_job_entrypoint.py --signal-steps $SignalSteps --long-steps $LongSteps --max-runtime-minutes 110
"@

hf jobs run python:3.11 bash -lc $bash --flavor $Flavor --timeout $Timeout --secrets HF_TOKEN --detach
