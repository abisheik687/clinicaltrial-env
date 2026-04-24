#!/usr/bin/env python3
"""Generate the required training curve and held-out comparison charts."""

from __future__ import annotations

import argparse
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
        return "Heuristic"
    if "phase1_grpo\\model" in model or model.endswith("phase1_grpo/model"):
        return "Trained"
    if "Qwen/Qwen2.5-0.5B-Instruct" in model:
        return "Base Model"
    return default


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
    baseline_label = label_for_eval(baseline_eval, "Baseline")
    trained_label = label_for_eval(trained_eval, "Trained")

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_log = load_json(args.train_log)
    baseline_eval = load_json(
        resolve_input_path(
            args.baseline_eval,
            resolve_input_path("artifacts/eval/fallback_task3_eval.json", "artifacts/eval/baseline_eval.json"),
        )
    )
    trained_eval = load_json(resolve_input_path(args.trained_eval, "artifacts/eval/trained_eval.json"))

    save_reward_curve(train_log, output_dir / "training_reward_curve.png")
    save_heldout_bar_chart(baseline_eval, trained_eval, output_dir / "heldout_base_vs_trained.png")


if __name__ == "__main__":
    main()
