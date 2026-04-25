#!/usr/bin/env python3
"""HF Jobs entrypoint: start env, train in staged passes, validate artifacts."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_URL = "http://127.0.0.1:7860"


def log(message: str) -> None:
    print(f"[hf-job] {message}", flush=True)


def wait_for_health(timeout_seconds: int) -> None:
    log("waiting for /health")
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            response = httpx.get(f"{ENV_URL}/health", timeout=5.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                log("/health == ok")
                return
            last_error = f"status={response.status_code} body={response.text[:200]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"FastAPI health check never reached ok: {last_error}")


def remaining_seconds(start_time: float, max_runtime_minutes: int, cushion_seconds: int = 60) -> int:
    elapsed = time.time() - start_time
    remaining = int(max_runtime_minutes * 60 - elapsed - cushion_seconds)
    if remaining <= 0:
        raise TimeoutError("max_runtime_guard reached before next command.")
    return remaining


def run_command(command: Sequence[str], start_time: float, max_runtime_minutes: int) -> None:
    timeout = remaining_seconds(start_time, max_runtime_minutes)
    log("running: " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, timeout=timeout)


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("ENABLE_INTERMEDIATE_SHAPING", "1")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7860",
    ]
    log("starting FastAPI environment")
    return subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)


def terminate_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    log("stopping FastAPI environment")
    if os.name == "nt":
        process.terminate()
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged ClinicalTrialEnv HF training job.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--signal-steps", type=int, default=100)
    parser.add_argument("--long-steps", type=int, default=500)
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--sft-warmstart-epochs", type=int, default=50)
    parser.add_argument("--max-runtime-minutes", type=int, default=110)
    parser.add_argument("--skip-long-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    os.environ.setdefault("ENABLE_INTERMEDIATE_SHAPING", "1")

    server = start_server()
    try:
        wait_for_health(timeout_seconds=120)
        common_eval = [
            sys.executable,
            "training/evaluate_models.py",
            "--policy",
            "local_model",
            "--env-url",
            ENV_URL,
            "--task-ids",
            "task3",
            "--seed-start",
            "200",
            "--num-seeds",
            "5",
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        run_command([sys.executable, "training/verify_task3_anchor.py", "--env-url", ENV_URL], started, args.max_runtime_minutes)
        run_command(
            [*common_eval, "--model-name", args.model, "--output", "artifacts/eval/base_model_task3_eval.json"],
            started,
            args.max_runtime_minutes,
        )

        signal_dir = "artifacts/phase1_grpo_signal"
        run_command(
            [
                sys.executable,
                "training/grpo_phase1.py",
                "--env-url",
                ENV_URL,
                "--task-id",
                "task3",
                "--model",
                args.model,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--num-episodes",
                str(args.num_episodes),
                "--num-generations",
                str(args.num_generations),
                "--sft-warmstart-epochs",
                str(args.sft_warmstart_epochs),
                "--max-steps",
                str(args.signal_steps),
                "--output-dir",
                signal_dir,
                "--collect-debug-rollouts",
            ],
            started,
            args.max_runtime_minutes,
        )
        run_command(
            [
                *common_eval,
                "--model-name",
                f"{signal_dir}/model",
                "--output",
                "artifacts/eval/trained_task3_eval_signal.json",
            ],
            started,
            args.max_runtime_minutes,
        )
        run_command(
            [
                sys.executable,
                "training/generate_before_after_trace.py",
                "--baseline-eval",
                "artifacts/eval/base_model_task3_eval.json",
                "--trained-eval",
                "artifacts/eval/trained_task3_eval_signal.json",
                "--output",
                "artifacts/eval/before_after_trajectories.json",
            ],
            started,
            args.max_runtime_minutes,
        )
        run_command(
            [
                sys.executable,
                "training/validate_training_outputs.py",
                "--train-log",
                f"{signal_dir}/train_log_history.json",
                "--rollout-debug",
                f"{signal_dir}/rollout_debug.json",
                "--baseline-eval",
                "artifacts/eval/base_model_task3_eval.json",
                "--trained-eval",
                "artifacts/eval/trained_task3_eval_signal.json",
            ],
            started,
            args.max_runtime_minutes,
        )
        log("100-step signal pass passed hard validation gates")

        final_dir = signal_dir if args.skip_long_run else "artifacts/phase1_grpo"
        final_eval = "artifacts/eval/trained_task3_eval_signal.json"
        if not args.skip_long_run:
            run_command(
                [
                    sys.executable,
                    "training/grpo_phase1.py",
                    "--env-url",
                    ENV_URL,
                    "--task-id",
                    "task3",
                    "--model",
                    f"{signal_dir}/model",
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--num-episodes",
                    str(args.num_episodes),
                    "--num-generations",
                    str(args.num_generations),
                    "--sft-warmstart-epochs",
                    "0",
                    "--max-steps",
                    str(args.long_steps),
                    "--output-dir",
                    final_dir,
                    "--collect-debug-rollouts",
                ],
                started,
                args.max_runtime_minutes,
            )
            final_eval = "artifacts/eval/trained_task3_eval.json"
            run_command(
                [*common_eval, "--model-name", f"{final_dir}/model", "--output", final_eval],
                started,
                args.max_runtime_minutes,
            )

        run_command(
            [
                sys.executable,
                "training/generate_before_after_trace.py",
                "--baseline-eval",
                "artifacts/eval/base_model_task3_eval.json",
                "--trained-eval",
                final_eval,
                "--output",
                "artifacts/eval/before_after_trajectories.json",
            ],
            started,
            args.max_runtime_minutes,
        )
        run_command(
            [
                sys.executable,
                "training/plot_results.py",
                "--train-log",
                f"{final_dir}/train_log_history.json",
                "--baseline-eval",
                "artifacts/eval/base_model_task3_eval.json",
                "--trained-eval",
                final_eval,
            ],
            started,
            args.max_runtime_minutes,
        )
        run_command(
            [
                sys.executable,
                "training/validate_training_outputs.py",
                "--train-log",
                f"{final_dir}/train_log_history.json",
                "--rollout-debug",
                f"{final_dir}/rollout_debug.json",
                "--baseline-eval",
                "artifacts/eval/base_model_task3_eval.json",
                "--trained-eval",
                final_eval,
            ],
            started,
            args.max_runtime_minutes,
        )
        run_command(
            [
                sys.executable,
                "training/generate_training_report.py",
                "--baseline-eval",
                "artifacts/eval/base_model_task3_eval.json",
                "--trained-eval",
                final_eval,
            ],
            started,
            args.max_runtime_minutes,
        )
        run_command([sys.executable, "training/check_artifact_integrity.py"], started, args.max_runtime_minutes)
        log("HF training job completed and artifacts passed validation")
    finally:
        terminate_server(server)


if __name__ == "__main__":
    main()
