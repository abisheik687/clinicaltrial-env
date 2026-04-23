#!/usr/bin/env python3
"""Phase 1 GRPO training using HTTP replay instead of environment_factory."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.name == "nt" and not sys.flags.utf8_mode:
    rerun = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )
    sys.exit(rerun.returncode)

from trl import GRPOConfig, GRPOTrainer

import inference
from server.models.action import ScreeningAction


ACTIVE_ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
ACTIVE_DEFAULT_TASK_ID = "task3"
ACTIVE_TIMEOUT = 60.0
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INVALID_COMPLETION_REWARD = -1.0


def summarize_observation(observation: dict[str, Any]) -> str:
    protocol = observation["trial_protocol_summary"]
    summary = {
        "patient_id": observation["patient_id"],
        "step_number": observation["step_number"],
        "steps_remaining": observation["steps_remaining"],
        "demographics": observation["demographics"],
        "diagnosis": observation["diagnosis"],
        "lab_values": observation["lab_values"],
        "current_medications": observation["current_medications"],
        "amendment_active": protocol["amendment_active"],
        "amendment_description": protocol["amendment_description"],
        "inclusion_criteria": protocol["inclusion_criteria"],
        "exclusion_criteria": protocol["exclusion_criteria"],
        "operational_state": observation.get("operational_state"),
        "info_message": observation["info_message"],
    }
    return json.dumps(summary, indent=2, default=str)


def build_episode_prompt(observation: dict[str, Any], task_id: str, seed: int, max_actions: int) -> str:
    schema = {
        "trajectory": [
            {
                "action_type": "evaluate_criterion",
                "criterion_id": "INC-001",
                "evaluation": {
                    "criterion_id": "INC-001",
                    "verdict": "met",
                    "reasoning": "Short clinical justification",
                },
                "confidence_score": 0.8,
            },
            {
                "action_type": "ask_clarification",
                "clarification_target": "INC-003",
                "confidence_score": 0.6,
            },
            {
                "action_type": "schedule_followup",
                "followup_day": 8,
                "confidence_score": 0.9,
            },
            {
                "action_type": "handle_safety_event",
                "safety_response": "escalate",
                "confidence_score": 0.9,
            },
        ]
    }
    return (
        "You are operating inside Clinical Trial Operations Arena.\n"
        "Plan the full episode up front and return only one JSON object.\n"
        f"Task: {task_id}\n"
        f"Episode seed: {seed}\n"
        f"Use at most {max_actions} actions.\n"
        "Use workflow actions when the observation enters followup_scheduling or safety_event.\n"
        "The final action in task3 should usually be handle_safety_event after a safe enroll.\n"
        "Each action must match the environment schema exactly.\n"
        "Ask for clarification only when the criterion is clarifiable and evidence is pending or ambiguous.\n"
        "Do not include markdown or commentary.\n\n"
        "JSON schema example:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Episode observation:\n"
        f"{summarize_observation(observation)}\n"
    )


def build_prompt_dataset(task_id: str, seed_start: int, num_episodes: int, max_actions: int) -> Dataset:
    prompts: list[str] = []
    task_ids: list[str] = []
    seeds: list[int] = []
    with httpx.Client(base_url=ACTIVE_ENV_URL, timeout=ACTIVE_TIMEOUT) as client:
        for offset in range(num_episodes):
            seed = seed_start + offset
            response = client.post("/reset", json={"task_id": task_id, "seed": seed})
            response.raise_for_status()
            data = response.json()
            prompts.append(build_episode_prompt(data["observation"], task_id, seed, max_actions))
            task_ids.append(task_id)
            seeds.append(seed)
    return Dataset.from_dict({"prompt": prompts, "task_id": task_ids, "seed": seeds})


def extract_json_block(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
            return text[index : index + end]
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in model completion.")


def normalize_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def parse_trajectory_completion(completion_text: str, max_actions: int) -> list[dict[str, Any]]:
    raw_json = extract_json_block(completion_text.strip())
    parsed = json.loads(raw_json)
    if isinstance(parsed, dict):
        trajectory = parsed.get("trajectory")
    elif isinstance(parsed, list):
        trajectory = parsed
    else:
        raise ValueError("Completion must decode to a JSON object or list.")

    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Trajectory must be a non-empty list.")

    validated: list[dict[str, Any]] = []
    for action in trajectory[:max_actions]:
        validated_action = ScreeningAction.model_validate(action)
        validated.append(validated_action.model_dump(exclude_none=True))
    return validated


def replay_trajectory(task_id: str, seed: int, trajectory: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    with httpx.Client(base_url=ACTIVE_ENV_URL, timeout=ACTIVE_TIMEOUT) as client:
        reset_response = client.post("/reset", json={"task_id": task_id, "seed": seed})
        reset_response.raise_for_status()
        reset_data = reset_response.json()
        session_id = reset_data["session_id"]

        reward_trace: list[float] = []
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
            if step_data["done"]:
                break

    return round(sum(reward_trace), 4), final_payload


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
    for completion, sample_task_id, sample_seed in zip(completions, task_id, seed, strict=True):
        try:
            trajectory = parse_trajectory_completion(normalize_completion_text(completion), max_actions=12)
            reward_value, _ = replay_trajectory(sample_task_id, int(sample_seed), trajectory)
        except Exception:
            reward_value = INVALID_COMPLETION_REWARD
        rewards.append(float(reward_value))

    if log_metric is not None and rewards:
        log_metric("http_replay_reward_mean", statistics.fmean(rewards))
    return rewards


def collect_rollouts(
    model_name: str,
    seeds: list[int],
    task_id: str,
    max_actions: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline(
        task="text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    dataset = build_prompt_dataset(task_id=task_id, seed_start=min(seeds), num_episodes=len(seeds), max_actions=max_actions)
    rollouts: list[dict[str, Any]] = []
    for prompt, seed_value in zip(dataset["prompt"], dataset["seed"], strict=True):
        completion = generator(prompt, num_return_sequences=1)[0]["generated_text"]
        try:
            trajectory = parse_trajectory_completion(completion, max_actions=max_actions)
            reward_value, final_payload = replay_trajectory(task_id, int(seed_value), trajectory)
        except Exception as exc:
            trajectory = []
            reward_value = INVALID_COMPLETION_REWARD
            final_payload = {"error": str(exc)}
        rollouts.append(
            {
                "prompt": prompt,
                "completion": completion,
                "reward": reward_value,
                "task_id": task_id,
                "seed": int(seed_value),
                "trajectory": trajectory,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 GRPO training against ClinicalTrialEnv task3.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-url", default=ACTIVE_ENV_URL)
    parser.add_argument("--task-id", default="task3")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=32)
    parser.add_argument("--output-dir", default="artifacts/phase1_grpo")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--collect-debug-rollouts", action="store_true")
    return parser.parse_args()


def main() -> None:
    global ACTIVE_ENV_URL, ACTIVE_DEFAULT_TASK_ID

    args = parse_args()
    ACTIVE_ENV_URL = args.env_url
    ACTIVE_DEFAULT_TASK_ID = args.task_id

    set_seed(args.seed)
    dataset = build_prompt_dataset(args.task_id, args.seed_start, args.num_episodes, args.max_actions)
    output_dir = Path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=False,
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
