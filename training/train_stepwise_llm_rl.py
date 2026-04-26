#!/usr/bin/env python3
"""Stepwise causal-LM policy training with environment rewards.

This replaces the broken whole-trajectory GRPO path with strict JSON action
generation, online reward propagation, and policy-gradient updates over full
episodes. The environment remains unchanged; only the LLM policy loop changes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference
from clinicaltrial_env.action import ScreeningAction
from server.environment.env import ClinicalTrialEnv
from training.stepwise_action_policy import build_candidate_actions, build_generation_prompt, compact_action_json, summarize_action_history


TASKS = {
    "task1": {"task_id": "task1", "seed": 42, "max_steps": 8},
    "task2": {"task_id": "task2", "seed": 43, "max_steps": 14},
    "task3": {"task_id": "task3", "seed": 44, "max_steps": 20},
}


@dataclass
class StepTransition:
    step: int
    task_id: str
    seed: int
    prompt: str
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_text: str
    action: dict[str, Any] | None
    reward: float
    done: bool
    invalid: bool
    success: bool
    unsafe: bool


@dataclass
class EpisodeResult:
    task_id: str
    seed: int
    total_reward: float
    success: bool
    unsafe: bool
    transitions: list[StepTransition]
    action_log: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stepwise LLM policy against ClinicalTrialEnv.")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--output-dir", default="artifacts/stepwise_llm_rl")
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument("--eval-seeds", type=int, default=6)
    parser.add_argument("--eval-start-seed", type=int, default=5000)
    parser.add_argument("--sft-seeds", type=int, default=12)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-ids", default="task1,task2,task3")
    return parser.parse_args()


def requested_tasks(task_ids: str) -> list[str]:
    selected = [task_id.strip() for task_id in task_ids.split(",") if task_id.strip()]
    return [task_id for task_id in selected if task_id in TASKS]


def load_model_and_tokenizer(model_name: str, use_lora: bool) -> tuple[Any, Any, torch.device]:
    tokenizer_source = model_name
    if model_name == "tiny-debug":
        tokenizer_source = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if model_name == "tiny-debug":
        from transformers import GPT2Config, GPT2LMHeadModel

        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_positions=4096,
                n_ctx=4096,
                n_embd=96,
                n_layer=2,
                n_head=2,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False)
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                task_type=TaskType.CAUSAL_LM,
                target_modules="all-linear",
            )
            model = get_peft_model(model, lora_config)
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    return model, tokenizer, device


def response_logprob(
    model: Any,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if response_ids.numel() == 0:
        return torch.tensor(0.0, device=device)
    full_ids = torch.cat([prompt_ids.to(device), response_ids.to(device)], dim=0)
    labels = full_ids.clone()
    labels[: prompt_ids.shape[0]] = -100
    outputs = model(
        input_ids=full_ids.unsqueeze(0),
        attention_mask=torch.ones_like(full_ids).unsqueeze(0),
        labels=labels.unsqueeze(0),
    )
    token_count = max(int((labels != -100).sum().item()), 1)
    return -outputs.loss * token_count


def score_candidates_batch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    candidates: list[dict[str, Any]],
) -> list[tuple[torch.Tensor, torch.Tensor, float, dict[str, Any]]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    if getattr(getattr(model, "config", None), "n_embd", 9999) <= 128:
        scored: list[tuple[torch.Tensor, torch.Tensor, float, dict[str, Any]]] = []
        for candidate in candidates:
            response_ids = tokenizer(compact_action_json(candidate), add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            score = response_logprob(model, prompt_ids, response_ids, device)
            scored.append((prompt_ids.detach().cpu(), response_ids.detach().cpu(), float(score.detach().cpu().item()), candidate))
        return scored
    response_batches = [
        tokenizer(compact_action_json(candidate), add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        for candidate in candidates
    ]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    full_sequences: list[torch.Tensor] = []
    label_sequences: list[torch.Tensor] = []
    max_len = 0
    for response_ids in response_batches:
        full_ids = torch.cat([prompt_ids, response_ids], dim=0)
        labels = full_ids.clone()
        labels[: prompt_ids.shape[0]] = -100
        full_sequences.append(full_ids)
        label_sequences.append(labels)
        max_len = max(max_len, int(full_ids.shape[0]))

    padded_inputs: list[torch.Tensor] = []
    padded_labels: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    for full_ids, labels in zip(full_sequences, label_sequences, strict=True):
        pad_len = max_len - int(full_ids.shape[0])
        padded_inputs.append(torch.cat([full_ids, torch.full((pad_len,), pad_id, device=device, dtype=full_ids.dtype)]))
        padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100, device=device, dtype=labels.dtype)]))
        attention_masks.append(torch.cat([torch.ones_like(full_ids), torch.zeros(pad_len, device=device, dtype=full_ids.dtype)]))

    batch_inputs = torch.stack(padded_inputs, dim=0)
    batch_labels = torch.stack(padded_labels, dim=0)
    batch_masks = torch.stack(attention_masks, dim=0)
    outputs = model(input_ids=batch_inputs, attention_mask=batch_masks)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = batch_labels[:, 1:]
    per_token_loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_logits.size(0), -1)
    scores = -per_token_loss.sum(dim=1)
    return [
        (prompt_ids.detach().cpu(), response_ids.detach().cpu(), float(score.detach().cpu().item()), candidate)
        for response_ids, score, candidate in zip(response_batches, scores, candidates, strict=True)
    ]


def choose_candidate_action(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    candidates: list[dict[str, Any]],
    *,
    do_sample: bool,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]:
    scored = score_candidates_batch(model, tokenizer, device, prompt, candidates)
    score_tensor = torch.tensor([item[2] for item in scored], dtype=torch.float32)
    if do_sample:
        probabilities = torch.softmax(score_tensor / max(temperature, 1e-4), dim=0)
        index = int(torch.multinomial(probabilities, 1).item())
    else:
        index = int(torch.argmax(score_tensor).item())
    prompt_ids, response_ids, _, candidate = scored[index]
    return prompt_ids, response_ids, compact_action_json(candidate), candidate


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    running = 0.0
    returns: list[float] = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def format_episode_metrics(episodes: list[EpisodeResult]) -> dict[str, float]:
    if not episodes:
        return {
            "success_rate": 0.0,
            "unsafe_rate": 0.0,
            "mean_reward": 0.0,
        }
    return {
        "success_rate": round(sum(1.0 for item in episodes if item.success) / len(episodes), 4),
        "unsafe_rate": round(sum(1.0 for item in episodes if item.unsafe) / len(episodes), 4),
        "mean_reward": round(sum(item.total_reward for item in episodes) / len(episodes), 4),
    }


def collect_supervised_examples(task_ids: list[str], per_task_seeds: int) -> list[tuple[str, str]]:
    env = ClinicalTrialEnv()
    examples: list[tuple[str, str]] = []
    for task_id in task_ids:
        max_steps = TASKS[task_id]["max_steps"]
        for seed in range(100, 100 + per_task_seeds):
            observation_model, session_id, _ = env.reset(task_id, seed)
            observation = observation_model.model_dump()
            history: list[str] = []
            action_records: list[dict[str, Any]] = []
            reward = 0.0
            for step in range(1, max_steps + 1):
                prompt = build_generation_prompt(
                    tokenizer=object(),
                    observation=observation,
                    reward=reward,
                    history=history,
                    step=step,
                    task_id=task_id,
                )
                action = inference.build_fallback_action(observation, task_id, action_records)
                examples.append((prompt, compact_action_json(action)))
                next_observation, reward_model, done, _ = env.step(session_id, ScreeningAction.model_validate(action))
                reward = float(reward_model.total_reward)
                history.append(summarize_action_history(action))
                action_records.append(action)
                observation = next_observation.model_dump()
                if done:
                    break
    return examples


def run_supervised_warmstart(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    examples: list[tuple[str, str]],
    epochs: int,
) -> None:
    if epochs <= 0 or not examples:
        return
    model.train()
    for _ in range(epochs):
        random.shuffle(examples)
        for prompt, target in examples:
            prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            full_ids = torch.cat([prompt_ids, target_ids], dim=0)
            labels = full_ids.clone()
            labels[: prompt_ids.shape[0]] = -100
            outputs = model(
                input_ids=full_ids.unsqueeze(0),
                attention_mask=torch.ones_like(full_ids).unsqueeze(0),
                labels=labels.unsqueeze(0),
            )
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def run_episode(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    env: ClinicalTrialEnv,
    task_id: str,
    seed: int,
    *,
    do_sample: bool,
    temperature: float,
) -> EpisodeResult:
    max_steps = TASKS[task_id]["max_steps"]
    observation_model, session_id, _ = env.reset(task_id, seed)
    observation = observation_model.model_dump()
    history: list[str] = []
    action_records: list[dict[str, Any]] = []
    transitions: list[StepTransition] = []
    action_log: list[dict[str, Any]] = []
    total_reward = 0.0
    last_reward = 0.0
    success = False
    unsafe = False

    for step in range(1, max_steps + 1):
        prompt = build_generation_prompt(
            tokenizer=tokenizer,
            observation=observation,
            reward=last_reward,
            history=history,
            step=step,
            task_id=task_id,
        )
        candidates = build_candidate_actions(observation, task_id, action_records)
        if not candidates:
            raise RuntimeError(f"No candidate actions available for {task_id} step {step}")
        prompt_ids, response_ids, response_text, parsed_action = choose_candidate_action(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            candidates=candidates,
            do_sample=do_sample,
            temperature=temperature,
        )
        invalid = False
        reward_value = 0.0
        done = False
        try:
            next_observation, reward_model, done, _ = env.step(session_id, ScreeningAction.model_validate(parsed_action))
            reward_value = float(reward_model.total_reward)
            success = bool(reward_model.terminal_success)
            unsafe = unsafe or bool(reward_model.unsafe_action)
            observation = next_observation.model_dump()
            action_records.append(parsed_action)
            history.append(summarize_action_history(parsed_action))
        except Exception:
            invalid = True
            reward_value = -1.0
            history.append(f"INVALID::{response_text[:160]}")

        total_reward += reward_value
        last_reward = reward_value
        action_log.append(
            {
                "step": step,
                "raw_output": response_text,
                "action": parsed_action if parsed_action is not None else "INVALID_ACTION",
                "reward": reward_value,
                "invalid": invalid,
            }
        )
        transitions.append(
            StepTransition(
                step=step,
                task_id=task_id,
                seed=seed,
                prompt=prompt,
                prompt_ids=prompt_ids.cpu(),
                response_ids=response_ids.cpu(),
                response_text=response_text,
                action=parsed_action,
                reward=reward_value,
                done=done,
                invalid=invalid,
                success=success,
                unsafe=unsafe,
            )
        )
        if done:
            break

    return EpisodeResult(
        task_id=task_id,
        seed=seed,
        total_reward=round(total_reward, 4),
        success=success,
        unsafe=unsafe,
        transitions=transitions,
        action_log=action_log,
    )


def update_from_episode(
    model: Any,
    optimizer: torch.optim.Optimizer,
    episode: EpisodeResult,
    device: torch.device,
    gamma: float,
    max_grad_norm: float,
) -> float:
    rewards = [transition.reward for transition in episode.transitions]
    returns = discounted_returns(rewards, gamma)
    if not returns:
        return 0.0
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device).clamp(min=-2.0, max=2.0)

    losses: list[torch.Tensor] = []
    for transition, episode_return in zip(episode.transitions, returns_tensor, strict=True):
        logprob = response_logprob(model, transition.prompt_ids, transition.response_ids, device)
        losses.append(-logprob * episode_return)

    if not losses:
        return 0.0
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu().item())


def evaluate_model(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    task_ids: list[str],
    num_seeds: int,
    start_seed: int,
) -> dict[str, Any]:
    env = ClinicalTrialEnv()
    episodes: list[EpisodeResult] = []
    model.eval()
    with torch.no_grad():
        for task_id in task_ids:
            for seed in range(start_seed, start_seed + num_seeds):
                episodes.append(
                    run_episode(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        env=env,
                        task_id=task_id,
                        seed=seed,
                        do_sample=False,
                        temperature=0.7,
                    )
                )
    model.train()
    aggregate = format_episode_metrics(episodes)
    return {
        "aggregate": aggregate,
        "episodes": [
            {
                "task_id": episode.task_id,
                "seed": episode.seed,
                "total_reward": episode.total_reward,
                "success": episode.success,
                "unsafe": episode.unsafe,
                "action_log": episode.action_log,
            }
            for episode in episodes
        ],
    }


def training_seed_for_index(index: int, task_id: str) -> int:
    task_offset = {"task1": 1000, "task2": 2000, "task3": 3000}[task_id]
    return task_offset + index


def reward_series(episodes: list[EpisodeResult]) -> list[float]:
    return [episode.total_reward for episode in episodes]


def sample_trajectory_diff(baseline_eval: dict[str, Any], trained_eval: dict[str, Any]) -> dict[str, Any]:
    baseline_map = {(item["task_id"], item["seed"]): item for item in baseline_eval["episodes"]}
    for trained in trained_eval["episodes"]:
        key = (trained["task_id"], trained["seed"])
        baseline = baseline_map.get(key)
        if baseline is None:
            continue
        if baseline["action_log"] != trained["action_log"]:
            return {
                "task_id": trained["task_id"],
                "seed": trained["seed"],
                "baseline_actions": baseline["action_log"][:6],
                "trained_actions": trained["action_log"][:6],
            }
    return {"task_id": None, "seed": None, "baseline_actions": [], "trained_actions": []}


def training_window_metrics(episodes: list[EpisodeResult], window: int = 10) -> dict[str, float]:
    if not episodes:
        return {
            "reward_start": 0.0,
            "reward_end": 0.0,
            "success_rate_start": 0.0,
            "success_rate_end": 0.0,
        }
    start_slice = episodes[: min(window, len(episodes))]
    end_slice = episodes[-min(window, len(episodes)) :]
    return {
        "reward_start": round(statistics.fmean(item.total_reward for item in start_slice), 4),
        "reward_end": round(statistics.fmean(item.total_reward for item in end_slice), 4),
        "success_rate_start": round(sum(1.0 for item in start_slice if item.success) / len(start_slice), 4),
        "success_rate_end": round(sum(1.0 for item in end_slice if item.success) / len(end_slice), 4),
    }


def sample_training_trajectory_diff(episodes: list[EpisodeResult]) -> dict[str, Any]:
    if len(episodes) < 2:
        return {"task_id": None, "seed_start": None, "seed_end": None, "start_actions": [], "end_actions": []}
    first = episodes[0]
    last = episodes[-1]
    return {
        "task_id": last.task_id,
        "seed_start": first.seed,
        "seed_end": last.seed,
        "start_actions": first.action_log[:6],
        "end_actions": last.action_log[:6],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    task_ids = requested_tasks(args.task_ids)
    if not task_ids:
        raise SystemExit("No valid task ids selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args.model_name, use_lora=args.use_lora)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    baseline_eval = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        task_ids=task_ids,
        num_seeds=args.eval_seeds,
        start_seed=args.eval_start_seed,
    )

    warmstart_examples = collect_supervised_examples(task_ids, per_task_seeds=args.sft_seeds)
    run_supervised_warmstart(model, tokenizer, device, optimizer, warmstart_examples, epochs=args.sft_epochs)

    training_history: list[dict[str, Any]] = []
    training_episodes: list[EpisodeResult] = []
    for episode_index in range(args.train_episodes):
        task_id = task_ids[episode_index % len(task_ids)]
        seed = training_seed_for_index(episode_index, task_id)
        env = ClinicalTrialEnv()
        episode = run_episode(
            model=model,
            tokenizer=tokenizer,
            device=device,
            env=env,
            task_id=task_id,
            seed=seed,
            do_sample=True,
            temperature=args.temperature,
        )
        loss = update_from_episode(
            model=model,
            optimizer=optimizer,
            episode=episode,
            device=device,
            gamma=args.gamma,
            max_grad_norm=args.max_grad_norm,
        )
        training_episodes.append(episode)
        recent_rewards = reward_series(training_episodes[-10:])
        reward_std = statistics.pstdev(recent_rewards) if len(recent_rewards) > 1 else 0.0
        training_history.append(
            {
                "episode": episode_index,
                "task_id": task_id,
                "seed": seed,
                "reward": episode.total_reward,
                "success": episode.success,
                "unsafe": episode.unsafe,
                "loss": round(loss, 6),
                "reward_std_recent10": round(reward_std, 6),
            }
        )
        if episode_index == 0 or (episode_index + 1) % args.log_every == 0:
            print(
                f"[TRAIN] episode={episode_index + 1} task={task_id} reward={episode.total_reward:.4f} "
                f"success={episode.success} unsafe={episode.unsafe} std_recent10={reward_std:.4f}",
                flush=True,
            )

    all_rewards = reward_series(training_episodes)
    reward_std = statistics.pstdev(all_rewards) if len(all_rewards) > 1 else 0.0
    if reward_std == 0.0:
        raise ValueError("No learning signal — rewards are constant")

    trained_eval = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        task_ids=task_ids,
        num_seeds=args.eval_seeds,
        start_seed=args.eval_start_seed,
    )

    window_metrics = training_window_metrics(training_episodes)
    trajectory_diff = sample_trajectory_diff(baseline_eval, trained_eval)
    training_diff = sample_training_trajectory_diff(training_episodes)

    summary = {
        "model_name": args.model_name,
        "task_ids": task_ids,
        "reward_std": round(reward_std, 6),
        "reward_start": window_metrics["reward_start"],
        "reward_end": window_metrics["reward_end"],
        "success_rate_before": window_metrics["success_rate_start"],
        "success_rate_after": window_metrics["success_rate_end"],
        "mean_reward_before": baseline_eval["aggregate"]["mean_reward"],
        "mean_reward_after": trained_eval["aggregate"]["mean_reward"],
        "heldout_success_rate_before": baseline_eval["aggregate"]["success_rate"],
        "heldout_success_rate_after": trained_eval["aggregate"]["success_rate"],
        "unsafe_rate_before": baseline_eval["aggregate"]["unsafe_rate"],
        "unsafe_rate_after": trained_eval["aggregate"]["unsafe_rate"],
        "sample_trajectory_diff": trajectory_diff,
        "training_trajectory_diff": training_diff,
    }

    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")
    (output_dir / "baseline_eval.json").write_text(json.dumps(baseline_eval, indent=2), encoding="utf-8")
    (output_dir / "trained_eval.json").write_text(json.dumps(trained_eval, indent=2), encoding="utf-8")
    (output_dir / "train_history.json").write_text(json.dumps(training_history, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
