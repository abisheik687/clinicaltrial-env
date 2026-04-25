#!/bin/bash
set -euo pipefail

echo "[hf-run] starting"
which python3
python3 --version

echo "[hf-run] cloning repo"
git clone https://github.com/abisheik687/clinicaltrial-env.git /workspace/ClinicalTrialEnv || true

cd /workspace/ClinicalTrialEnv

echo "[hf-run] installing dependencies"
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt -r requirements-training.txt

echo "[hf-run] starting training"
python3 training/hf_job_entrypoint.py \
  --signal-steps 100 \
  --long-steps 500 \
  --max-runtime-minutes 110
