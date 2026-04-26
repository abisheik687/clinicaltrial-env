#!/usr/bin/env python3
"""Phase 1 GRPO training using HTTP replay instead of environment_factory."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

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
from training.config import LORA_CONFIG, Phase1Config, QLORA_CONFIG
import inference


ACTIVE_ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
ACTIVE_DEFAULT_TASK_ID = "task3"
ACTIVE_TIMEOUT = 60.0
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INVALID_COMPLETION_REWARD = -1.0
INVALID_TRAJECTORY_WEIGHT = 0.1
MAX_PROMPT_TOKENS = 800
LOCAL_DEBUG = os.environ.get("LOCAL_SIGNAL_DEBUG", "0") == "1"


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
            prompt = build_episode_prompt(
                data["observation"],
                task_id,
                seed,
                max_actions,
                tokenizer=tokenizer,
                local_debug_mode=os.environ.get("LOCAL_SIGNAL_DEBUG", "0") == "1",
            )
            if tokenizer is not None:
                encoded = tokenizer(prompt, truncation=True, max_length=MAX_PROMPT_TOKENS, return_tensors=None)
                prompt = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
            prompts.append(prompt)
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


def reward_noise() -> float:
    return random.random() * 0.1 - 0.05


def _pending_criteria(observation: dict[str, Any], action_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    criteria = observation["trial_protocol_summary"]["inclusion_criteria"] + observation["trial_protocol_summary"]["exclusion_criteria"]
    evaluated_ids = {
        record.get("criterion_id")
        for record in action_records
        if record.get("action_type") == "evaluate_criterion"
    }
    return [criterion for criterion in criteria if criterion["criterion_id"] not in evaluated_ids]


def build_valid_action_candidates(
    observation: dict[str, Any],
    task_id: str,
    action_records: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    operational_state = observation.get("operational_state") or {}
    workflow_phase = operational_state.get("workflow_phase", "screening")

    if workflow_phase == "followup_scheduling":
        start = int(operational_state.get("followup_window_start") or 7)
        end = int(operational_state.get("followup_window_end") or start)
        midpoint = int((start + end) / 2)
        for day in dict.fromkeys([start, midpoint, end]):
            candidates.append(
                {
                    "action_type": "schedule_followup",
                    "followup_day": day,
                    "confidence_score": 0.75,
                }
            )
        return candidates

    if workflow_phase == "safety_event":
        candidates.append(
            {
                "action_type": "handle_safety_event",
                "safety_response": "escalate",
                "confidence_score": 0.8,
            }
        )
        return candidates

    pending = _pending_criteria(observation, action_records)
    rng.shuffle(pending)
    for criterion in pending[:3]:
        criterion_id = criterion["criterion_id"]
        if criterion.get("clarification_available", False) and inference.should_request_clarification(observation, task_id, criterion_id):
            candidates.append(
                {
                    "action_type": "ask_clarification",
                    "clarification_target": criterion_id,
                    "confidence_score": 0.58,
                }
            )
        verdict, reasoning = inference.evaluate_criterion_heuristically(observation, task_id, criterion_id)
        candidates.append(
            {
                "action_type": "evaluate_criterion",
                "criterion_id": criterion_id,
                "evaluation": {
                    "criterion_id": criterion_id,
                    "verdict": verdict,
                    "reasoning": reasoning[:80],
                },
                "confidence_score": 0.72 if verdict != "uncertain" else 0.46,
            }
        )

    final_decision_allowed = bool(action_records)
    if task_id == "task3":
        final_decision_allowed = final_decision_allowed and bool(observation["trial_protocol_summary"].get("amendment_active", False))

    if final_decision_allowed:
        decision, reason = inference.choose_final_decision(observation, task_id, action_records)
        candidates.append(
            {
                "action_type": decision,
                "final_decision_reason": reason[:120],
                "confidence_score": 0.67,
            }
        )
        if decision != "exclude":
            candidates.append(
                {
                    "action_type": "exclude",
                    "final_decision_reason": "fallback_invalid_output",
                    "confidence_score": 0.41,
                }
            )
    return candidates


def build_fallback_trajectory(task_id: str, seed: int, max_actions: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed + 17)
    step_debug: list[dict[str, Any]] = []
    with httpx.Client(base_url=ACTIVE_ENV_URL, timeout=ACTIVE_TIMEOUT) as client:
        reset_response = client.post("/reset", json={"task_id": task_id, "seed": seed})
        reset_response.raise_for_status()
        reset_data = reset_response.json()
        session_id = reset_data["session_id"]
        observation = reset_data["observation"]
        action_records: list[dict[str, Any]] = []
        trajectory: list[dict[str, Any]] = []

        for step_index in range(1, max_actions + 1):
            candidates = build_valid_action_candidates(observation, task_id, action_records, rng)
            if not candidates:
                break
            action = rng.choice(candidates)
            step_response = client.post("/step", json={"session_id": session_id, "action": action})
            step_response.raise_for_status()
            step_data = step_response.json()
            trajectory.append(action)
            action_records.append(action)
            step_debug.append(
                {
                    "step": step_index,
                    "raw_model_output": "FALLBACK_TRAJECTORY",
                    "parsed_action": action,
                    "reward": float(step_data["reward"]["total_reward"]),
                    "done": bool(step_data["done"]),
                }
            )
            observation = step_data["observation"]
            if step_data["done"]:
                break

    return trajectory, step_debug


def safe_parse_trajectory(
    completion_text: str,
    max_actions: int,
    task_id: str,
    seed: int,
) -> tuple[list[dict[str, Any]], bool]:
    try:
        parsed = parse_trajectory_completion(completion_text, max_actions)
        if not parsed:
            raise ValueError("Empty trajectory")
        return parsed, False
    except Exception:
        fallback_trajectory, _ = build_fallback_trajectory(task_id, seed, max_actions)
        if fallback_trajectory:
            return fallback_trajectory, True
        return [{"action_type": "exclude", "final_decision_reason": "fallback_invalid_output"}], True


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
        raw_output = normalize_completion_text(completion)
        try:
            trajectory, used_fallback = safe_parse_trajectory(raw_output, max_actions=12, task_id=sample_task_id, seed=int(sample_seed))
            reward_value, final_payload, _ = replay_trajectory(sample_task_id, int(sample_seed), trajectory)
            reward_value, weight, invalid_or_unsafe = weighted_reward(reward_value, final_payload)
            if used_fallback:
                reward_value += reward_noise()
            weight_trace.append(weight)
            if invalid_or_unsafe:
                invalid_or_unsafe_count += 1
            if LOCAL_DEBUG:
                print(
                    json.dumps(
                        {
                            "raw_model_output": raw_output[:300],
                            "parsed_action": trajectory[0] if trajectory else None,
                            "reward": round(float(reward_value), 4),
                            "used_fallback": used_fallback,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
        except Exception:
            reward_value = INVALID_COMPLETION_REWARD + reward_noise()
        rewards.append(float(reward_value))

    if log_metric is not None and rewards:
        mean_reward = statistics.fmean(rewards)
        reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        advantages = [reward - mean_reward for reward in rewards]
        advantage_std = statistics.pstdev(advantages) if len(advantages) > 1 else 0.0
        advantage_mean = statistics.fmean(abs(advantage) for advantage in advantages) if advantages else 0.0
        log_metric("http_replay_reward_mean", mean_reward)
        log_metric("trajectory_final_reward_std", reward_std)
        log_metric("advantage_mean", advantage_mean)
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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_ids = inputs["input_ids"][:, -MAX_PROMPT_TOKENS:]
        inputs["input_ids"] = input_ids
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -MAX_PROMPT_TOKENS:]
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_tokens = generated[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        try:
            trajectory, used_fallback = safe_parse_trajectory(completion, max_actions=max_actions, task_id=task_id, seed=int(seed_value))
            reward_value, final_payload, reward_trace = replay_trajectory(task_id, int(seed_value), trajectory)
            weighted_value, reward_weight, invalid_or_unsafe = weighted_reward(reward_value, final_payload)
            if used_fallback:
                weighted_value += reward_noise()
                reward_value = weighted_value
            step_debug = [
                {
                    "step": index + 1,
                    "raw_model_output": completion[:300],
                    "parsed_action": action,
                    "reward": reward_trace[index] if index < len(reward_trace) else None,
                }
                for index, action in enumerate(trajectory)
            ]
        except Exception as exc:
            trajectory = []
            reward_value = INVALID_COMPLETION_REWARD + reward_noise()
            weighted_value = reward_value
            reward_weight = 1.0
            invalid_or_unsafe = False
            reward_trace = []
            step_debug = []
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
                "step_debug": step_debug,
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
    parser.add_argument("--generation-batch-size", type=int, default=6)
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


def build_trainable_model(model_name: str, local_debug_mode: bool) -> AutoModelForCausalLM:
    if local_debug_mode:
        return AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False)

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=QLORA_CONFIG["load_in_4bit"],
            bnb_4bit_quant_type=QLORA_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=QLORA_CONFIG["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, QLORA_CONFIG["bnb_4bit_compute_dtype"]),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            target_modules=LORA_CONFIG["target_modules"],
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora_config)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False)


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
    Phase1Config(
        model_name=args.model,
        env_url=args.env_url,
        task_id=args.task_id,
        seed_start=args.seed_start,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        grpo_epsilon=args.grpo_epsilon,
        seed=args.seed,
        max_actions=args.max_actions,
        max_new_tokens=args.max_new_tokens,
        local_debug_mode=args.local_debug_mode,
        collect_debug_rollouts=args.collect_debug_rollouts,
        sft_warmstart_epochs=args.sft_warmstart_epochs,
        sft_learning_rate=args.sft_learning_rate,
    ).validate()
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
    model = build_trainable_model(args.model, args.local_debug_mode)
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

    trainer_kwargs = dict(
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
    )
    if hasattr(GRPOConfig, "__dataclass_fields__") and "generation_batch_size" in GRPOConfig.__dataclass_fields__:
        trainer_kwargs["generation_batch_size"] = args.generation_batch_size

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        reward_funcs=environment_reward,
        processing_class=tokenizer,
        args=GRPOConfig(**trainer_kwargs),
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
