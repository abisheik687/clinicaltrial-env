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
from training.trajectory_helpers import build_episode_prompt, parse_trajectory_completion


class FallbackOnlyClient:
    """Sentinel client used to force the heuristic fallback policy."""


class LocalModelClient:
    """Evaluate local checkpoints with the same full-trajectory format used in GRPO."""

    def __init__(self, model_name: str, max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False)
        if hasattr(self.model.generation_config, "max_length"):
            self.model.generation_config.max_length = None
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, field):
                setattr(self.model.generation_config, field, None)
        self.planned_trajectories: dict[str, list[dict[str, Any]]] = {}
        self.plan_indices: dict[str, int] = {}
        self.plan_failures: set[str] = set()

    def get_action(
        self,
        observation: dict[str, Any],
        reward: float,
        history: list[str],
        step: int,
        task_id: str,
        action_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del reward, history
        patient_id = observation["patient_id"]
        try:
            if patient_id in self.plan_failures:
                return inference.build_fallback_action(observation, task_id, action_records)
            if patient_id not in self.planned_trajectories or not action_records:
                self._cache_plan(
                    patient_id=patient_id,
                    observation=observation,
                    task_id=task_id,
                    max_actions=min(max(observation.get("steps_remaining", 1), 1), 10),
                )
            planned_actions = self.planned_trajectories.get(patient_id, [])
            plan_index = self.plan_indices.get(patient_id, 0)
            if plan_index < len(planned_actions):
                planned_action = planned_actions[plan_index]
                self.plan_indices[patient_id] = plan_index + 1
                return inference.stabilize_action(
                    planned_action,
                    observation,
                    task_id,
                    action_records,
                )
        except Exception:
            self.plan_failures.add(patient_id)
        return inference.build_fallback_action(observation, task_id, action_records)

    def _cache_plan(
        self,
        patient_id: str,
        observation: dict[str, Any],
        task_id: str,
        max_actions: int,
    ) -> None:
        prompt = build_episode_prompt(
            observation=observation,
            task_id=task_id,
            seed=None,
            max_actions=max_actions,
            tokenizer=self.tokenizer,
        )
        generated_text = self._generate_text(prompt)
        self.planned_trajectories[patient_id] = parse_trajectory_completion(generated_text, max_actions=max_actions)
        self.plan_indices[patient_id] = 0

    def _generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion_tokens = generated[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion_tokens, skip_special_tokens=True)


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
    rewards: list[float] = []
    done = False
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

        response = await env_client.post("/step", json={"session_id": session_id, "action": action}, timeout=30.0)
        if response.status_code >= 400:
            action = inference.build_fallback_action(observation, task_id, action_records)
            response = await env_client.post("/step", json={"session_id": session_id, "action": action}, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        observation = data["observation"]
        done = bool(data["done"])
        final_reward_payload = data["reward"]
        last_reward = float(data["reward"]["total_reward"])
        rewards.append(last_reward)
        action_records.append(action)
        history.append(json.dumps(action, separators=(",", ":")))

    return {
        "task_id": task_id,
        "seed": seed,
        "steps": len(rewards),
        "final_reward": rewards[-1] if rewards else 0.0,
        "terminal_success": bool(final_reward_payload.get("terminal_success", False)),
        "unsafe_action": bool(final_reward_payload.get("unsafe_action", False)),
        "diagnostic_metrics": final_reward_payload.get("diagnostic_metrics", {}),
        "trajectory": action_records,
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
        "mean_final_reward": round(sum(float(item["final_reward"]) for item in episodes) / count, 4),
        **aggregated,
    }


async def main_async(args: argparse.Namespace) -> None:
    if args.policy == "fallback":
        model_client: OpenAI | FallbackOnlyClient | LocalModelClient = FallbackOnlyClient()
    elif args.policy == "local_model":
        model_client = LocalModelClient(model_name=args.model_name, max_new_tokens=args.max_new_tokens)
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
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output", default="artifacts/eval/baseline_eval.json")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
