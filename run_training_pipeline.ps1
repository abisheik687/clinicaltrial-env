$ErrorActionPreference = "Stop"

Write-Host "Starting API server in background..."
$env:PYTHONPATH = "."
$serverProcess = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", "7860" -PassThru -WindowStyle Hidden

Write-Host "Waiting 5 seconds for server to boot..."
Start-Sleep -Seconds 5

try {
    Write-Host "1. Running baseline evaluation..."
    .\.venv\Scripts\python.exe training\evaluate_models.py --policy local_model --model-name Qwen/Qwen2.5-0.5B-Instruct --task-ids task3 --num-seeds 3 --seed-start 200 --output artifacts\eval\base_model_task3_eval.json

    Write-Host "2. Running compact action-policy training..."
    .\.venv\Scripts\python.exe training\train_task3_policy_gradient.py --seed-start 200 --train-seed-count 50 --eval-seed-start 200 --eval-num-seeds 50 --train-steps 15 --batch-size 50 --learning-rate 0.3 --log-every 1

    Write-Host "3. Validating compact action-policy artifacts..."
    .\.venv\Scripts\python.exe training\validate_training_outputs.py --mode action_policy --train-log artifacts\phase1_pg\train_log_history.json --trained-eval artifacts\eval\policy_gradient_task3_eval.json

    Write-Host "4. Checking artifact integrity..."
    .\.venv\Scripts\python.exe training\check_artifact_integrity.py

    Write-Host "Pipeline completed successfully."
} finally {
    Write-Host "Stopping API server..."
    Stop-Process -Id $serverProcess.Id -Force
}
