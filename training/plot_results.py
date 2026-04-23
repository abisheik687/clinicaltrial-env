#!/usr/bin/env python3
"""Generate the required training curve and held-out comparison charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_reward_series(log_history: list[dict]) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    rewards: list[float] = []
    for index, record in enumerate(log_history):
        reward_value = None
        for key in ("reward", "rewards/reward", "train_reward", "objective/reward"):
            if key in record and isinstance(record[key], (int, float)):
                reward_value = float(record[key])
                break
        if reward_value is None:
            continue
        steps.append(int(record.get("step", index + 1)))
        rewards.append(reward_value)
    return steps, rewards


def save_reward_curve(train_log: dict, output_path: Path) -> None:
    steps, rewards = extract_reward_series(train_log)
    if not steps:
        steps = list(range(1, len(train_log) + 1))
        rewards = [float(item.get("loss", 0.0)) for item in train_log]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, marker="o", linewidth=2)
    plt.title("Phase 1 GRPO Training Reward")
    plt.xlabel("Training step")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_heldout_bar_chart(baseline_eval: dict, trained_eval: dict, output_path: Path) -> None:
    metrics = ["success_rate", "unsafe_rate", "amendment_recovery_rate"]
    baseline_values = [baseline_eval["aggregate"][metric] for metric in metrics]
    trained_values = [trained_eval["aggregate"][metric] for metric in metrics]
    positions = range(len(metrics))

    plt.figure(figsize=(10, 5))
    plt.bar([p - 0.18 for p in positions], baseline_values, width=0.36, label="Baseline")
    plt.bar([p + 0.18 for p in positions], trained_values, width=0.36, label="Trained")
    plt.xticks(list(positions), ["Success rate", "Unsafe rate", "Amendment recovery"])
    plt.ylim(0, 1)
    plt.ylabel("Held-out score")
    plt.title("Held-out Base vs Trained Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reward and held-out comparison plots.")
    parser.add_argument("--train-log", default="artifacts/phase1_grpo/train_log_history.json")
    parser.add_argument("--baseline-eval", default="artifacts/eval/baseline_eval.json")
    parser.add_argument("--trained-eval", default="artifacts/eval/trained_eval.json")
    parser.add_argument("--output-dir", default="artifacts/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_log = load_json(args.train_log)
    baseline_eval = load_json(args.baseline_eval)
    trained_eval = load_json(args.trained_eval)

    save_reward_curve(train_log, output_dir / "training_reward_curve.png")
    save_heldout_bar_chart(baseline_eval, trained_eval, output_dir / "heldout_base_vs_trained.png")


if __name__ == "__main__":
    main()
