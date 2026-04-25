#!/usr/bin/env python3
"""Phase 1 GRPO training using HTTP replay instead of environment_factory."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__" and os.name == "nt" and not sys.flags.utf8_mode:
    rerun = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )
    sys.exit(rerun.returncode)

from training.trajectory_helpers import (
    build_episode_prompt,
    normalize_completion_text,
    parse_trajectory_completion,
)
from training.task3_anchor import TASK3_ANCHOR_SEED, task3_anchor_completion, task3_compact_completion
from training.verify_task3_anchor import replay_anchor


ACTIVE_ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
ACTIVE_DEFAULT_TASK_ID = "task3"
ACTIVE_TIMEOUT = 60.0
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INVALID_COMPLETION_REWARD = -1.0
INVALID_TRAJECTORY_WEIGHT = 0.1


def wait_for_server(url: str, max_wait_seconds: int = 120, poll_interval_seconds: int = 5) -> None:
    """Poll GET {url}/ until the server responds or the deadline is exceeded."""
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        try:
            response = httpx.get(url + "/", timeout=5)
            if response.status_code < 500:
                return
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        time.sleep(poll_interval_seconds)
    raise RuntimeError(
        f"Environment server not reachable at {url} after {max_wait_seconds}s. "
        "Start the server (e.g. `docker-compose up`) before running training."
    )


def build_prompt_dataset(task_id: str, seed_start: int, num_episodes: int, max_actions: int, tokenizer: Any | None = None) -> Dataset:
    prompts: list[str] = []
    task_ids: list[str] = []
    seeds: list[int] = []
    with httpx.Client(base_url=ACTIVE_ENV_URL, timeout=ACTIVE_TIMEOUT) as client:
        for offset in range(num_episodes):
            seed = seed_start + offset
            response = client.post("/reset", json={"task_id": task_id, "seed": seed})
            response.raise_for_status()
            data = response.json()
            prompts.append(
                build_episode_prompt(
                    data["observation"],
                    task_id,
                    seed,
                    max_actions,
                    tokenizer=tokenizer,
                    local_debug_mode=os.environ.get("LOCAL_SIGNAL_DEBUG", "0") == "1",
                )
            )
            task_ids.append(task_id)
            seeds.append(seed)
    return Dataset.from_dict({"prompt": prompts, "task_id": task_ids, "seed": seeds})

def replay_trajectory(task_id: str, seed: int, trajectory: list[dict[str, Any]]) -> tuple[float, dict[str, Any], list[float]]:
    with httpx.Client(base_url=ACTIVE_ENV_URL, timeout=ACTIVE_TIMEOUT) as client:
        reset_response = client.post("/reset", json={"task_id": task_id, "seed": seed})
        reset_response.raise_for_status()
        reset_data = reset_response.json()
        session_id = reset_data["session_id"]
        observation = reset_data["observation"]

        reward_trace: list[float] = []
        action_records: list[dict[str, Any]] = []
        final_payload: dict[str, Any] = {
            "done": False,
            "reward": {"total_reward": 0.0, "terminal_success": False, "unsafe_action": False},
        }

        for action in trajectory:
            step_response = client.post("/step", json={"session_id": session_id, "action": action})
            step_response.raise_for_status()
            step_data = step_response.json()
            reward_trace.append(float(step_data["reward"]["total_reward"]))
            final_payload = step_data
            observation = step_data["observation"]
            action_records.append(action)
            if step_data["done"]:
                break

    return round(sum(reward_trace), 4), final_payload, reward_trace


def weighted_reward(raw_reward: float, final_payload: dict[str, Any]) -> tuple[float, float, bool]:
    reward_payload = final_payload.get("reward", {}) if isinstance(final_payload, dict) else {}
    info_payload = final_payload.get("info", {}) if isinstance(final_payload, dict) else {}
    invalid_or_unsafe = bool(reward_payload.get("unsafe_action")) or bool(info_payload.get("invalid_action"))
    weight = INVALID_TRAJECTORY_WEIGHT if invalid_or_unsafe else 1.0
    return float(raw_reward) * weight, weight, invalid_or_unsafe


def environment_reward(
    prompts: list[Any],
    completions: list[Any],
    task_id: list[str],
    seed: list[int],
    log_metric=None,
    **kwargs,
) -> list[float]:
    del prompts, kwargs
    rewards: list[float] = []
    invalid_or_unsafe_count = 0
    weight_trace: list[float] = []
    for completion, sample_task_id, sample_seed in zip(completions, task_id, seed, strict=True):
        try:
            trajectory = parse_trajectory_completion(normalize_completion_text(completion), max_actions=12)
            reward_value, final_payload, _ = replay_trajectory(sample_task_id, int(sample_seed), trajectory)
            reward_value, weight, invalid_or_unsafe = weighted_reward(reward_value, final_payload)
            weight_trace.append(weight)
            if invalid_or_unsafe:
                invalid_or_unsafe_count += 1
        except Exception:
            reward_value = INVALID_COMPLETION_REWARD
        rewards.append(float(reward_value))

    if log_metric is not None and rewards:
        mean_reward = statistics.fmean(rewards)
        reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        advantages = [reward - mean_reward for reward in rewards]
        advantage_std = statistics.pstdev(advantages) if len(advantages) > 1 else 0.0
        log_metric("http_replay_reward_mean", mean_reward)
        log_metric("trajectory_final_reward_std", reward_std)
        log_metric("advantage_mean", statistics.fmean(advantages) if advantages else 0.0)
        log_metric("advantage_std", advantage_std)
        if weight_trace:
            log_metric("trajectory_weight_mean", statistics.fmean(weight_trace))
            log_metric("trajectory_invalid_or_unsafe_rate", invalid_or_unsafe_count / len(weight_trace))
    return rewards


def collect_rollouts(
    model_name: str,
    seeds: list[int],
    task_id: str,
    max_actions: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False)
    if torch.cuda.is_available():
        model.to("cuda")
    if hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
    for field in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, field):
            setattr(model.generation_config, field, None)
    rollouts: list[dict[str, Any]] = []
    for seed_value in seeds:
        with httpx.Client(base_url=ACTIVE_ENV_URL, timeout=ACTIVE_TIMEOUT) as client:
            response = client.post("/reset", json={"task_id": task_id, "seed": int(seed_value)})
            response.raise_for_status()
            prompt = build_episode_prompt(
                response.json()["observation"],
                task_id=task_id,
                seed=int(seed_value),
                max_actions=max_actions,
                tokenizer=tokenizer,
                local_debug_mode=os.environ.get("LOCAL_SIGNAL_DEBUG", "0") == "1",
            )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_tokens = generated[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        try:
            trajectory = parse_trajectory_completion(completion, max_actions=max_actions)
            reward_value, final_payload, reward_trace = replay_trajectory(task_id, int(seed_value), trajectory)
            weighted_value, reward_weight, invalid_or_unsafe = weighted_reward(reward_value, final_payload)
        except Exception as exc:
            trajectory = []
            reward_value = INVALID_COMPLETION_REWARD
            weighted_value = INVALID_COMPLETION_REWARD
            reward_weight = 1.0
            invalid_or_unsafe = False
            reward_trace = []
            final_payload = {"error": str(exc)}
        rollouts.append(
            {
                "prompt": prompt,
                "completion": completion,
                "reward": reward_value,
                "weighted_reward": weighted_value,
                "reward_weight": reward_weight,
                "invalid_or_unsafe": invalid_or_unsafe,
                "task_id": task_id,
                "seed": int(seed_value),
                "trajectory": trajectory,
                "reward_trace": reward_trace,
                "final_payload": final_payload,
            }
        )
    return rollouts


def rollouts_to_dataset(rollouts: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": rollout["prompt"],
                "completion": rollout["completion"],
                "reward": rollout["reward"],
                "task_id": rollout["task_id"],
                "seed": rollout["seed"],
            }
            for rollout in rollouts
        ]
    )


def save_log_history(trainer: GRPOTrainer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "train_log_history.json"
    history_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")


def save_rollout_debug(output_dir: Path, rollouts: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "rollout_debug.json"
    debug_path.write_text(json.dumps(rollouts, indent=2, default=str), encoding="utf-8")


def verify_anchor_gate(env_url: str, output_dir: Path) -> None:
    """Stop training immediately if the canonical perfect path is broken."""
    payload = replay_anchor(env_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task3_anchor_verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["passed"]:
        final_reward = float(payload.get("final_reward", {}).get("total_reward", -999.0))
        violations = int(payload.get("violations", -1))
        terminal_success = bool(payload.get("final_reward", {}).get("terminal_success", False))
        raise RuntimeError(
            "Task 3 anchor trajectory failed. "
            f"terminal_success={terminal_success} reward={final_reward} violations={violations}. "
            "Stop training and fix the environment/path first."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 GRPO training against ClinicalTrialEnv task3.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-url", default=ACTIVE_ENV_URL)
    parser.add_argument("--task-id", default="task3")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--output-dir", default="artifacts/phase1_grpo")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--grpo-epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--collect-debug-rollouts", action="store_true")
    parser.add_argument("--local-debug-mode", action="store_true")
    parser.add_argument("--sft-warmstart-epochs", type=int, default=50)
    parser.add_argument("--sft-learning-rate", type=float, default=5e-6)
    return parser.parse_args()


def _anchor_warmstart_text(task_id: str, max_actions: int, tokenizer: Any) -> str:
    with httpx.Client(base_url=ACTIVE_ENV_URL, timeout=ACTIVE_TIMEOUT) as client:
        response = client.post("/reset", json={"task_id": task_id, "seed": TASK3_ANCHOR_SEED})
        response.raise_for_status()
        observation = response.json()["observation"]
    prompt = build_episode_prompt(
        observation,
        task_id=task_id,
        seed=TASK3_ANCHOR_SEED,
        max_actions=max_actions,
        tokenizer=tokenizer,
    )
    # Use compact trajectory (~200 tokens) that fits within max_new_tokens budget
    return f"{prompt}{task3_compact_completion()}"


def run_anchor_warmstart(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task_id: str,
    max_actions: int,
    epochs: int,
    learning_rate: float,
) -> None:
    if task_id != "task3" or epochs <= 0:
        return
    training_text = _anchor_warmstart_text(task_id=task_id, max_actions=max_actions, tokenizer=tokenizer)
    encoded = tokenizer(training_text, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**encoded, labels=labels)
        outputs.loss.backward()
        optimizer.step()


def main() -> None:
    global ACTIVE_ENV_URL, ACTIVE_DEFAULT_TASK_ID

    from trl import GRPOConfig, GRPOTrainer

    args = parse_args()
    ACTIVE_ENV_URL = args.env_url
    ACTIVE_DEFAULT_TASK_ID = args.task_id
    if args.local_debug_mode:
        os.environ["LOCAL_SIGNAL_DEBUG"] = "1"

    # Enable intermediate reward shaping so non-terminal steps produce signal
    os.environ.setdefault("ENABLE_INTERMEDIATE_SHAPING", "1")

    set_seed(args.seed)
    wait_for_server(ACTIVE_ENV_URL)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = build_prompt_dataset(args.task_id, args.seed_start, args.num_episodes, args.max_actions, tokenizer=tokenizer)
    output_dir = Path(args.output_dir)
    if args.task_id == "task3":
        verify_anchor_gate(ACTIVE_ENV_URL, output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=False,
    )
    if torch.cuda.is_available():
        model.to("cuda")
    run_anchor_warmstart(
        model=model,
        tokenizer=tokenizer,
        task_id=args.task_id,
        max_actions=args.max_actions,
        epochs=args.sft_warmstart_epochs,
        learning_rate=args.sft_learning_rate,
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        reward_funcs=environment_reward,
        processing_class=tokenizer,
        args=GRPOConfig(
            output_dir=str(output_dir),
            max_steps=args.max_steps,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            epsilon=args.grpo_epsilon,
            max_completion_length=args.max_new_tokens,
            logging_steps=5,
            save_steps=25,
            report_to="none",
            remove_unused_columns=False,
            use_cpu=not torch.cuda.is_available(),
            dataloader_pin_memory=False,
        ),
    )

    trainer.train()
    trainer.save_model(str(output_dir / "model"))
    save_log_history(trainer, output_dir)

    if args.collect_debug_rollouts:
        debug_seeds = list(range(args.seed_start, args.seed_start + min(args.num_episodes, 4)))
        debug_rollouts = collect_rollouts(
            model_name=str(output_dir / "model"),
            seeds=debug_seeds,
            task_id=args.task_id,
            max_actions=args.max_actions,
            max_new_tokens=args.max_new_tokens,
        )
        save_rollout_debug(output_dir, debug_rollouts)
        rollouts_to_dataset(debug_rollouts).to_json(str(output_dir / "rollout_dataset.json"))


if __name__ == "__main__":
    main()
