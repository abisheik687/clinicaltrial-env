#!/usr/bin/env python3
"""Held-out evaluation for baseline or trained models against ClinicalTrialEnv."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference
from training.stepwise_action_policy import build_candidate_actions, build_generation_prompt, compact_action_json, summarize_action_history


class FallbackOnlyClient:
    """Sentinel client used to force the heuristic fallback policy."""


class LocalModelClient:
    """Evaluate local checkpoints with strict stepwise JSON action generation."""

    def __init__(self, model_name: str, max_new_tokens: int = 384, do_sample: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False)
        if torch.cuda.is_available():
            self.model.to("cuda")
        if hasattr(self.model.generation_config, "max_length"):
            self.model.generation_config.max_length = None
        self.invalid_patients: set[str] = set()

    def get_action(
        self,
        observation: dict[str, Any],
        reward: float,
        history: list[str],
        step: int,
        task_id: str,
        action_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        patient_id = observation["patient_id"]
        try:
            prompt = build_generation_prompt(
                tokenizer=self.tokenizer,
                observation=observation,
                reward=reward,
                history=history,
                step=step,
                task_id=task_id,
            )
            candidates = build_candidate_actions(observation, task_id, action_records)
            if not candidates:
                raise ValueError("No valid candidates")
            chosen = self._choose_candidate(prompt, candidates)
            return inference.stabilize_action(chosen, observation, task_id, action_records)
        except Exception:
            pass
        self.invalid_patients.add(patient_id)
        return {
            "action_type": "INVALID_ACTION",
            "raw_output": "invalid_or_unparseable_model_action",
            "confidence_score": 0.0,
        }

    def _candidate_logprob(self, prompt: str, candidate: dict[str, Any]) -> float:
        return self._candidate_logprobs(prompt, [candidate])[0]

    def _candidate_logprobs(self, prompt: str, candidates: list[dict[str, Any]]) -> list[float]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.model.device)
        if getattr(getattr(self.model, "config", None), "n_embd", 9999) <= 128:
            scores: list[float] = []
            for candidate in candidates:
                candidate_ids = self.tokenizer(compact_action_json(candidate), add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.model.device)
                full_ids = self.torch.cat([prompt_ids, candidate_ids], dim=0)
                labels = full_ids.clone()
                labels[: prompt_ids.shape[0]] = -100
                with self.torch.no_grad():
                    outputs = self.model(
                        input_ids=full_ids.unsqueeze(0),
                        attention_mask=self.torch.ones_like(full_ids).unsqueeze(0),
                        labels=labels.unsqueeze(0),
                    )
                token_count = max(int((labels != -100).sum().item()), 1)
                scores.append(float((-outputs.loss * token_count).cpu().item()))
            return scores
        candidate_batches = [
            self.tokenizer(compact_action_json(candidate), add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.model.device)
            for candidate in candidates
        ]
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        full_sequences: list[Any] = []
        label_sequences: list[Any] = []
        max_len = 0
        for candidate_ids in candidate_batches:
            full_ids = self.torch.cat([prompt_ids, candidate_ids], dim=0)
            labels = full_ids.clone()
            labels[: prompt_ids.shape[0]] = -100
            full_sequences.append(full_ids)
            label_sequences.append(labels)
            max_len = max(max_len, int(full_ids.shape[0]))

        padded_inputs = []
        padded_labels = []
        padded_masks = []
        for full_ids, labels in zip(full_sequences, label_sequences, strict=True):
            pad_len = max_len - int(full_ids.shape[0])
            padded_inputs.append(self.torch.cat([full_ids, self.torch.full((pad_len,), pad_id, device=self.model.device, dtype=full_ids.dtype)]))
            padded_labels.append(self.torch.cat([labels, self.torch.full((pad_len,), -100, device=self.model.device, dtype=labels.dtype)]))
            padded_masks.append(self.torch.cat([self.torch.ones_like(full_ids), self.torch.zeros(pad_len, device=self.model.device, dtype=full_ids.dtype)]))

        batch_inputs = self.torch.stack(padded_inputs, dim=0)
        batch_labels = self.torch.stack(padded_labels, dim=0)
        batch_masks = self.torch.stack(padded_masks, dim=0)
        with self.torch.no_grad():
            outputs = self.model(input_ids=batch_inputs, attention_mask=batch_masks)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = batch_labels[:, 1:]
        per_token_loss = self.torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_logits.size(0), -1)
        return [float(value.cpu().item()) for value in (-per_token_loss.sum(dim=1))]

    def _choose_candidate(self, prompt: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        scores = self.torch.tensor(self._candidate_logprobs(prompt, candidates), dtype=self.torch.float32)
        if self.do_sample:
            probabilities = self.torch.softmax(scores / 0.8, dim=0)
            index = int(self.torch.multinomial(probabilities, 1).item())
        else:
            index = int(self.torch.argmax(scores).item())
        return candidates[index]


async def run_episode(
    model_client: OpenAI | FallbackOnlyClient | LocalModelClient,
    env_client: httpx.AsyncClient,
    task_id: str,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    reset_response = await env_client.post("/reset", json={"task_id": task_id, "seed": seed}, timeout=30.0)
    reset_response.raise_for_status()
    reset_data = reset_response.json()

    observation = reset_data["observation"]
    session_id = reset_data["session_id"]
    history: list[str] = []
    action_records: list[dict[str, Any]] = []
    action_log: list[dict[str, Any]] = []
    rewards: list[float] = []
    done = False
    fallback_used = False
    last_reward = 0.0
    final_reward_payload: dict[str, Any] = {}

    for step in range(1, max_steps + 1):
        if done:
            break

        if isinstance(model_client, FallbackOnlyClient):
            action = inference.build_fallback_action(observation, task_id, action_records)
        elif isinstance(model_client, LocalModelClient):
            action = model_client.get_action(
                observation,
                last_reward,
                history,
                step,
                task_id,
                action_records,
            )
        else:
            action = inference.get_agent_action(
                model_client,
                observation,
                last_reward,
                history,
                step,
                task_id,
                action_records,
            )

        if action.get("action_type") == "INVALID_ACTION":
            last_reward = -1.0
            rewards.append(last_reward)
            action_log.append({"step": step, "action": action, "reward": last_reward, "invalid": True})
            history.append(summarize_action_history(action))
            fallback_used = True
            continue

        response = await env_client.post("/step", json={"session_id": session_id, "action": action}, timeout=30.0)
        if response.status_code >= 400:
            fallback_used = True
            if isinstance(model_client, FallbackOnlyClient):
                action = inference.build_fallback_action(observation, task_id, action_records)
                response = await env_client.post("/step", json={"session_id": session_id, "action": action}, timeout=30.0)
            else:
                break
        response.raise_for_status()
        data = response.json()
        observation = data["observation"]
        done = bool(data["done"])
        final_reward_payload = data["reward"]
        last_reward = float(data["reward"]["total_reward"])
        rewards.append(last_reward)
        action_records.append(action)
        action_log.append({"step": step, "action": action, "reward": last_reward, "invalid": False})
        history.append(summarize_action_history(action))

    return {
        "task_id": task_id,
        "seed": seed,
        "steps": len(rewards),
        "final_reward": rewards[-1] if rewards else 0.0,
        "terminal_success": bool(final_reward_payload.get("terminal_success", False)),
        "unsafe_action": bool(final_reward_payload.get("unsafe_action", False)),
        "fallback_used": fallback_used
        or (isinstance(model_client, LocalModelClient) and observation["patient_id"] in model_client.invalid_patients),
        "diagnostic_metrics": final_reward_payload.get("diagnostic_metrics", {}),
        "trajectory": action_records,
        "action_log": action_log,
    }


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {
            "success_rate": 0.0,
            "unsafe_rate": 0.0,
            "amendment_recovery_rate": 0.0,
            "mean_final_reward": 0.0,
        }
    count = len(episodes)
    metric_names = (
        "amendment_recovery_rate",
        "eligibility_component_score",
        "amendment_component_score",
        "scheduling_component_score",
        "safety_component_score",
    )
    aggregated = {
        metric_name: round(
            sum(float(item["diagnostic_metrics"].get(metric_name, 0.0)) for item in episodes) / count,
            4,
        )
        for metric_name in metric_names
    }
    return {
        "success_rate": round(sum(1.0 for item in episodes if item["terminal_success"]) / count, 4),
        "unsafe_rate": round(sum(1.0 for item in episodes if item["unsafe_action"]) / count, 4),
        "fallback_used_rate": round(sum(1.0 for item in episodes if item.get("fallback_used", False)) / count, 4),
        "mean_final_reward": round(sum(float(item["final_reward"]) for item in episodes) / count, 4),
        **aggregated,
    }


async def main_async(args: argparse.Namespace) -> None:
    if args.policy == "fallback":
        model_client: OpenAI | FallbackOnlyClient | LocalModelClient = FallbackOnlyClient()
    elif args.policy == "local_model":
        model_client = LocalModelClient(
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )
    else:
        os.environ["MODEL_NAME"] = args.model_name
        model_client = OpenAI(base_url=args.api_base_url, api_key=args.api_key)

    tasks = [
        {"task_id": "task1", "max_steps": 8},
        {"task_id": "task2", "max_steps": 14},
        {"task_id": "task3", "max_steps": 20},
    ]
    requested_task_ids = {task_id.strip() for task_id in args.task_ids.split(",") if task_id.strip()}
    tasks = [task for task in tasks if task["task_id"] in requested_task_ids]
    async with httpx.AsyncClient(base_url=args.env_url) as env_client:
        episodes: list[dict[str, Any]] = []
        for task in tasks:
            for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                episodes.append(
                    await run_episode(
                        model_client,
                        env_client,
                        task_id=task["task_id"],
                        seed=seed,
                        max_steps=task["max_steps"],
                    )
                )

    payload = {
        "policy": args.policy,
        "model": "heuristic-fallback" if args.policy == "fallback" else args.model_name,
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "task_ids": [task["task_id"] for task in tasks],
        "aggregate": aggregate(episodes),
        "aggregate_by_task": {
            task["task_id"]: aggregate([episode for episode in episodes if episode["task_id"] == task["task_id"]])
            for task in tasks
        },
        "episodes": episodes,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline or trained models on held-out seeds.")
    parser.add_argument("--policy", choices=["fallback", "model", "local_model"], default="fallback")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--api-base-url", default=inference.API_BASE_URL)
    parser.add_argument("--api-key", default=inference.API_KEY)
    parser.add_argument("--env-url", default=inference.ENV_BASE_URL)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--task-ids", default="task1,task2,task3")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--output", default="artifacts/eval/baseline_eval.json")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
