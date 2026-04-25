#!/usr/bin/env python3
"""Validate judge-facing RL evidence before claiming training improvement."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


REWARD_KEYS = (
    "http_replay_reward_mean",
    "reward",
    "rewards/environment_reward/mean",
    "objective/reward",
)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reward_series(log_history: list[dict[str, Any]]) -> list[float]:
    rewards: list[float] = []
    for row in log_history:
        for key in REWARD_KEYS:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                rewards.append(float(value))
                break
    return rewards


def moving_average(values: list[float], window: int = 20) -> list[float]:
    if not values:
        return []
    window = max(1, min(window, 30, len(values)))
    return [statistics.fmean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]


def rollout_stats(rollouts: list[dict[str, Any]]) -> dict[str, float]:
    if not rollouts:
        return {
            "trajectory_final_reward_std": 0.0,
            "episode_done_rate": 0.0,
            "valid_trajectory_rate": 0.0,
            "unsafe_rate": 0.0,
        }
    final_rewards: list[float] = []
    done_count = 0
    valid_count = 0
    unsafe_count = 0
    for rollout in rollouts:
        final_payload = rollout.get("final_payload", {})
        reward_payload = final_payload.get("reward", {}) if isinstance(final_payload, dict) else {}
        if rollout.get("trajectory") and not rollout.get("final_payload", {}).get("error"):
            valid_count += 1
        if final_payload.get("done"):
            done_count += 1
        if reward_payload.get("unsafe_action"):
            unsafe_count += 1
        if "reward" in rollout and isinstance(rollout["reward"], (int, float)):
            final_rewards.append(float(rollout["reward"]))
    return {
        "trajectory_final_reward_std": statistics.pstdev(final_rewards) if len(final_rewards) > 1 else 0.0,
        "episode_done_rate": done_count / len(rollouts),
        "valid_trajectory_rate": valid_count / len(rollouts),
        "unsafe_rate": unsafe_count / len(rollouts),
    }


def structural_diff(before_eval: dict[str, Any], after_eval: dict[str, Any]) -> dict[str, Any]:
    before_episode = before_eval.get("episodes", [{}])[0]
    after_episode = after_eval.get("episodes", [{}])[0]
    before_actions = [action.get("action_type") for action in before_episode.get("trajectory", [])]
    after_actions = [action.get("action_type") for action in after_episode.get("trajectory", [])]
    decision_actions = {"enroll", "exclude", "defer", "schedule_followup", "handle_safety_event"}
    before_decisions = [action for action in before_actions if action in decision_actions]
    after_decisions = [action for action in after_actions if action in decision_actions]
    return {
        "seed": before_episode.get("seed"),
        "before_actions": before_actions,
        "after_actions": after_actions,
        "before_decision_points": before_decisions,
        "after_decision_points": after_decisions,
        "length_differs": len(before_actions) != len(after_actions),
        "decision_points_differ": before_decisions != after_decisions,
        "structural_behavior_diff": len(before_actions) != len(after_actions) or before_decisions != after_decisions,
    }


def load_or_build_behavior_diff(
    before_after_path: str | Path,
    before_eval: dict[str, Any],
    after_eval: dict[str, Any],
) -> dict[str, Any]:
    path = Path(before_after_path)
    if path.exists():
        payload = load_json(path)
        if "structural_behavior_diff" in payload:
            return payload
    return structural_diff(before_eval, after_eval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final training artifacts.")
    parser.add_argument("--train-log", default="artifacts/phase1_grpo/train_log_history.json")
    parser.add_argument("--rollout-debug", default="artifacts/phase1_grpo/rollout_debug.json")
    parser.add_argument("--baseline-eval", default="artifacts/eval/base_model_task3_eval.json")
    parser.add_argument("--trained-eval", default="artifacts/eval/trained_task3_eval.json")
    parser.add_argument("--output", default="artifacts/eval/training_validation_summary.json")
    parser.add_argument("--baseline-output", default="artifacts/eval/baseline_avg_reward.json")
    parser.add_argument("--before-after", default="artifacts/eval/before_after_trajectories.json")
    parser.add_argument("--before-after-output", default="artifacts/eval/before_after_trajectory_diff.json")
    parser.add_argument("--min-reward-delta", type=float, default=0.2)
    parser.add_argument("--allow-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_log_payload = load_json(args.train_log)
    train_log = train_log_payload if isinstance(train_log_payload, list) else train_log_payload.get("log_history", [])
    baseline_eval = load_json(args.baseline_eval)
    trained_eval = load_json(args.trained_eval)
    rollouts = load_json(args.rollout_debug) if Path(args.rollout_debug).exists() else []

    rewards = reward_series(train_log)
    smoothed = moving_average(rewards, window=20)
    rollout_metrics = rollout_stats(rollouts)
    clipped_values = [
        float(row["completions/clipped_ratio"])
        for row in train_log
        if isinstance(row.get("completions/clipped_ratio"), (int, float))
    ]
    advantage_std_values = [
        float(row["advantage_std"])
        for row in train_log
        if isinstance(row.get("advantage_std"), (int, float))
    ]
    advantage_mean_values = [
        float(row["advantage_mean"])
        for row in train_log
        if isinstance(row.get("advantage_mean"), (int, float))
    ]
    diff = load_or_build_behavior_diff(args.before_after, baseline_eval, trained_eval)
    baseline_avg_reward = float(baseline_eval["aggregate"].get("mean_final_reward", 0.0))
    trained_avg_reward = float(trained_eval["aggregate"].get("mean_final_reward", 0.0))
    reward_delta = (smoothed[-1] - smoothed[0]) if len(smoothed) >= 2 else 0.0

    gates = {
        "trajectory_reward_std_positive": rollout_metrics["trajectory_final_reward_std"] > 0.0,
        "episode_done_rate_positive": rollout_metrics["episode_done_rate"] > 0.0,
        "valid_trajectory_rate_min": rollout_metrics["valid_trajectory_rate"] > 0.3,
        "clipped_ratio_below_threshold": (max(clipped_values) if clipped_values else 1.0) < 0.9,
        "reward_delta_min": reward_delta >= args.min_reward_delta,
        "unsafe_rate_zero": float(trained_eval["aggregate"].get("unsafe_rate", 1.0)) == 0.0,
        "no_fallback_used": "fallback_used_rate" in trained_eval["aggregate"]
        and float(trained_eval["aggregate"].get("fallback_used_rate", 1.0)) == 0.0,
        "structural_behavior_diff": bool(diff["structural_behavior_diff"]),
    }
    hard_failure_same = (
        baseline_eval["aggregate"].get("success_rate") == trained_eval["aggregate"].get("success_rate")
        and baseline_avg_reward == trained_avg_reward
        and not diff["structural_behavior_diff"]
    )
    payload = {
        "passed": all(gates.values()) and not hard_failure_same,
        "gates": gates,
        "hard_failure_same_success_reward_behavior": hard_failure_same,
        "reward_series": rewards,
        "reward_delta": reward_delta,
        "initial_avg_reward": smoothed[0] if smoothed else None,
        "final_avg_reward": smoothed[-1] if smoothed else None,
        "baseline_avg_reward": baseline_avg_reward,
        "trained_avg_reward": trained_avg_reward,
        "max_clipped_ratio": max(clipped_values) if clipped_values else None,
        "advantage_mean": statistics.fmean(advantage_mean_values) if advantage_mean_values else None,
        "advantage_std": statistics.fmean(advantage_std_values) if advantage_std_values else None,
        **rollout_metrics,
        "behavior_diff": diff,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.baseline_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.baseline_output).write_text(json.dumps({"baseline_avg_reward": baseline_avg_reward}, indent=2), encoding="utf-8")
    Path(args.before_after_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.before_after_output).write_text(json.dumps(diff, indent=2), encoding="utf-8")

    if not payload["passed"] and not args.allow_failed:
        failed = [name for name, passed in gates.items() if not passed]
        raise SystemExit(f"Training validation failed: {failed}; hard_failure_same={hard_failure_same}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
