#!/usr/bin/env python3
"""Run a short local signal-debug training loop before any HF training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.verify_task3_anchor import replay_anchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short local signal-debug loop for Task 3.")
    parser.add_argument("--env-url", default=os.environ.get("ENV_URL", "http://localhost:7860"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--base-output-dir", default="artifacts/local_signal_debug")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--grpo-epsilon", type=float, default=0.05)
    return parser.parse_args()


def _extract_reward(log_row: dict[str, Any]) -> float | None:
    for key in ("reward", "rewards/environment_reward/mean", "http_replay_reward_mean"):
        value = log_row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def analyze_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [reward for row in history if (reward := _extract_reward(row)) is not None]
    clipped = [float(row["completions/clipped_ratio"]) for row in history if "completions/clipped_ratio" in row]
    invalid_rates = [float(row["trajectory_invalid_or_unsafe_rate"]) for row in history if "trajectory_invalid_or_unsafe_rate" in row]
    reward_delta = 0.0 if len(rewards) < 2 else rewards[-1] - rewards[0]
    return {
        "num_logs": len(history),
        "reward_series": rewards,
        "reward_delta": reward_delta,
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "clipped_ratio_max": max(clipped) if clipped else None,
        "clipped_ratio_last": clipped[-1] if clipped else None,
        "invalid_or_unsafe_rate_last": invalid_rates[-1] if invalid_rates else None,
        "upward_trend": reward_delta > 0.0,
        "clipping_saturated": bool(clipped) and clipped[-1] >= 0.99,
    }


def analyze_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    if not rollouts:
        return {
            "num_rollouts": 0,
            "done_rate": None,
            "invalid_or_unsafe_rate": None,
            "end_only_reward_warning": None,
        }
    done_flags: list[bool] = []
    invalid_or_unsafe_flags: list[bool] = []
    reward_sequences: list[list[float]] = []
    for rollout in rollouts:
        final_payload = rollout.get("final_payload", {})
        done_flags.append(bool(final_payload.get("done", False)))
        reward_payload = final_payload.get("reward", {})
        info_payload = final_payload.get("info", {})
        invalid_or_unsafe_flags.append(bool(reward_payload.get("unsafe_action")) or bool(info_payload.get("invalid_action")))
        trajectory_rewards = [float(item) for item in rollout.get("reward_trace", []) if isinstance(item, (int, float))]
        if trajectory_rewards:
            reward_sequences.append(trajectory_rewards)
    end_only_reward = False
    if reward_sequences:
        end_only_reward = all(
            all(value == 0.0 for value in sequence[:-1]) and sequence[-1] != 0.0
            for sequence in reward_sequences
            if sequence
        )
    return {
        "num_rollouts": len(rollouts),
        "done_rate": sum(done_flags) / len(done_flags),
        "invalid_or_unsafe_rate": sum(invalid_or_unsafe_flags) / len(invalid_or_unsafe_flags),
        "end_only_reward_warning": end_only_reward,
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.base_output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    anchor_payload = replay_anchor(args.env_url)
    (output_dir / "anchor_verification.json").write_text(json.dumps(anchor_payload, indent=2), encoding="utf-8")
    if not anchor_payload.get("passed", False):
        raise SystemExit("Anchor gate failed. Stop and fix environment/reward path first.")

    env = dict(os.environ)
    env["LOCAL_SIGNAL_DEBUG"] = "1"
    train_output_dir = output_dir / "grpo"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "grpo_phase1.py"),
        "--env-url",
        args.env_url,
        "--task-id",
        "task3",
        "--model",
        args.model,
        "--max-steps",
        str(args.steps),
        "--num-episodes",
        str(args.num_episodes),
        "--seed-start",
        str(args.seed_start),
        "--learning-rate",
        str(args.learning_rate),
        "--grpo-epsilon",
        str(args.grpo_epsilon),
        "--sft-warmstart-epochs",
        "1",
        "--local-debug-mode",
        "--output-dir",
        str(train_output_dir),
        "--collect-debug-rollouts",
    ]
    subprocess.run(cmd, check=True, env=env)

    history_path = train_output_dir / "train_log_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    diagnostics = analyze_history(history)
    rollout_path = train_output_dir / "rollout_debug.json"
    if rollout_path.exists():
        rollouts = json.loads(rollout_path.read_text(encoding="utf-8"))
        diagnostics["rollout_checks"] = analyze_rollouts(rollouts)
    (output_dir / "signal_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    csv_path = output_dir / "reward_trace.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "reward"])
        for index, reward in enumerate(diagnostics["reward_series"]):
            writer.writerow([index, reward])

    hard_failures: list[str] = []
    if not diagnostics["upward_trend"]:
        hard_failures.append("reward trend is flat/non-increasing")
    if diagnostics["clipping_saturated"]:
        hard_failures.append("clipped_ratio remains saturated near 1.0")
    rollout_checks = diagnostics.get("rollout_checks", {})
    if rollout_checks.get("done_rate") == 0.0:
        hard_failures.append("episodes never reach done")
    if rollout_checks.get("invalid_or_unsafe_rate") == 1.0:
        hard_failures.append("all rollouts are invalid or unsafe")
    if rollout_checks.get("end_only_reward_warning"):
        hard_failures.append("reward appears sparse with mostly terminal-only signal")
    if hard_failures:
        raise SystemExit(f"Local signal debug failed: {', '.join(hard_failures)}")
    print("Local signal debug passed.")


if __name__ == "__main__":
    main()
