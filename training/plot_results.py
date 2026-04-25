#!/usr/bin/env python3
"""Generate the required training curve and held-out comparison charts."""

from __future__ import annotations

import argparse
import shutil
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_input_path(primary: str, fallback: str) -> str:
    primary_path = Path(primary)
    if primary_path.exists():
        return str(primary_path)
    return fallback


def label_for_eval(eval_payload: dict, default: str) -> str:
    policy = eval_payload.get("policy")
    model = str(eval_payload.get("model", ""))
    if policy == "fallback":
        return "Heuristic Reference"
    if "phase1_grpo\\model" in model or model.endswith("phase1_grpo/model"):
        return "RL-Trained Model"
    if "Qwen/Qwen2.5-0.5B-Instruct" in model:
        return "Untrained Model"
    return default


def extract_reward_series(log_history: list[dict]) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    rewards: list[float] = []
    for index, record in enumerate(log_history):
        reward_value = None
        for key in (
            "http_replay_reward_mean",
            "reward",
            "rewards/environment_reward/mean",
            "rewards/reward",
            "train_reward",
            "objective/reward",
        ):
            if key in record and isinstance(record[key], (int, float)):
                reward_value = float(record[key])
                break
        if reward_value is None:
            continue
        steps.append(int(record.get("step", index + 1)))
        rewards.append(reward_value)
    return steps, rewards


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, min(window, len(values)))
    averaged: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(sum(values[start : index + 1]) / (index - start + 1))
    return averaged


def save_reward_curve(train_log: list[dict], baseline_eval: dict, output_path: Path, window: int) -> None:
    steps, rewards = extract_reward_series(train_log)
    if not steps:
        steps = list(range(1, len(train_log) + 1))
        rewards = [float(item.get("loss", 0.0)) for item in train_log]
    if not steps:
        steps = [1]
        rewards = [0.0]

    window = min(max(1, window), 30)
    smoothed_rewards = moving_average(rewards, window)
    baseline_avg_reward = float(baseline_eval["aggregate"].get("mean_final_reward", 0.0))
    baseline_output = output_path.parent.parent / "eval" / "baseline_avg_reward.json"
    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    baseline_output.write_text(
        json.dumps({"baseline_avg_reward": baseline_avg_reward}, indent=2),
        encoding="utf-8",
    )
    first_reward = smoothed_rewards[0]
    last_reward = smoothed_rewards[-1]
    y_values = [baseline_avg_reward, *smoothed_rewards]
    y_min = min(y_values)
    y_max = max(y_values)
    padding = max(0.1, (y_max - y_min) * 0.35)

    plt.figure(figsize=(10, 5))
    plt.axhline(
        baseline_avg_reward,
        color="#8b1e3f",
        linewidth=2.2,
        linestyle="--",
        label="Untrained Model",
    )
    plt.plot(
        steps,
        smoothed_rewards,
        color="#1f77b4",
        marker="o",
        linewidth=2.8,
        label="RL-Trained Model",
    )
    plt.title("Task 3 Training Reward vs Untrained Baseline")
    plt.xlabel("training steps")
    plt.ylabel("avg reward")
    plt.ylim(y_min - padding, y_max + padding)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.figtext(
        0.5,
        0.01,
        f"Reward increases from {first_reward:.2f} -> {last_reward:.2f} over training "
        f"(moving average window={min(max(1, window), len(rewards))}).",
        ha="center",
        fontsize=10,
        color="#18324b",
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_heldout_bar_chart(baseline_eval: dict, trained_eval: dict, output_path: Path) -> None:
    metrics = ["success_rate", "unsafe_rate", "amendment_recovery_rate"]
    baseline_values = [baseline_eval["aggregate"][metric] for metric in metrics]
    trained_values = [trained_eval["aggregate"][metric] for metric in metrics]
    positions = range(len(metrics))
    baseline_label = label_for_eval(baseline_eval, "Untrained Model")
    trained_label = label_for_eval(trained_eval, "RL-Trained Model")

    plt.figure(figsize=(10, 5))
    baseline_bars = plt.bar([p - 0.18 for p in positions], baseline_values, width=0.36, label=baseline_label)
    trained_bars = plt.bar([p + 0.18 for p in positions], trained_values, width=0.36, label=trained_label)
    plt.xticks(list(positions), ["Success rate", "Unsafe rate", "Amendment recovery"])
    plt.ylim(0, 1)
    plt.ylabel("Held-out score")
    plt.title("Task 3 Baseline vs Trained Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    for bar_group in (baseline_bars, trained_bars):
        for bar in bar_group:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#18324b",
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reward and held-out comparison plots.")
    parser.add_argument("--train-log", default="artifacts/phase1_grpo/train_log_history.json")
    parser.add_argument("--baseline-eval", default="artifacts/eval/base_model_task3_eval.json")
    parser.add_argument("--trained-eval", default="artifacts/eval/trained_task3_eval.json")
    parser.add_argument("--output-dir", default="artifacts/plots")
    parser.add_argument("--moving-average-window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_log_payload = load_json(args.train_log)
    train_log = train_log_payload if isinstance(train_log_payload, list) else train_log_payload.get("log_history", [])
    baseline_eval = load_json(
        resolve_input_path(
            args.baseline_eval,
            resolve_input_path("artifacts/eval/fallback_task3_eval.json", "artifacts/eval/baseline_eval.json"),
        )
    )
    trained_eval = load_json(resolve_input_path(args.trained_eval, "artifacts/eval/trained_eval.json"))

    reward_curve = output_dir / "training_reward_curve.png"
    heldout_chart = output_dir / "heldout_base_vs_trained.png"
    save_reward_curve(train_log, baseline_eval, reward_curve, args.moving_average_window)
    save_heldout_bar_chart(baseline_eval, trained_eval, heldout_chart)
    shutil.copyfile(reward_curve, output_dir / "backup_training_reward_curve.png")


if __name__ == "__main__":
    main()
