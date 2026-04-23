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


class FallbackOnlyClient:
    """Sentinel client used to force the heuristic fallback policy."""


class LocalModelClient:
    """Small wrapper around a local Transformers checkpoint for held-out eval."""

    def __init__(self, model_name: str, max_new_tokens: int = 256) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = pipeline(
            task="text-generation",
            model=AutoModelForCausalLM.from_pretrained(model_name),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )

    def get_action(
        self,
        observation: dict[str, Any],
        reward: float,
        history: list[str],
        step: int,
        task_id: str,
        action_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(observation, reward, history, step)
        content: str | None = None
        try:
            generated = self.generator(prompt, num_return_sequences=1, do_sample=False)
            content = generated[0]["generated_text"]
            return inference.parse_action_payload(content)
        except Exception:
            if content is not None:
                try:
                    repair_prompt = self._build_repair_prompt(observation, content, step)
                    repaired = self.generator(repair_prompt, num_return_sequences=1, do_sample=False)
                    return inference.parse_action_payload(repaired[0]["generated_text"])
                except Exception:
                    pass
        return inference.build_fallback_action(observation, task_id, action_records)

    def _build_prompt(self, observation: dict[str, Any], reward: float, history: list[str], step: int) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": inference.build_system_prompt()},
                    {"role": "user", "content": inference.build_user_message(observation, reward, history, step)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"{inference.build_system_prompt()}\n\n"
            f"{inference.build_user_message(observation, reward, history, step)}\n"
        )

    def _build_repair_prompt(self, observation: dict[str, Any], invalid_content: str, step: int) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": inference.build_system_prompt()},
                    {"role": "user", "content": inference.build_repair_message(observation, invalid_content, step)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"{inference.build_system_prompt()}\n\n"
            f"{inference.build_repair_message(observation, invalid_content, step)}\n"
        )


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
        }
    count = len(episodes)
    return {
        "success_rate": round(sum(1.0 for item in episodes if item["terminal_success"]) / count, 4),
        "unsafe_rate": round(sum(1.0 for item in episodes if item["unsafe_action"]) / count, 4),
        "amendment_recovery_rate": round(
            sum(float(item["diagnostic_metrics"].get("amendment_recovery_rate", 0.0)) for item in episodes) / count,
            4,
        ),
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
        "model": args.model_name,
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "aggregate": aggregate(episodes),
        "episodes": episodes,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline or trained models on held-out seeds.")
    parser.add_argument("--policy", choices=["fallback", "model", "local_model"], default="fallback")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--api-base-url", default=inference.API_BASE_URL)
    parser.add_argument("--api-key", default=inference.API_KEY)
    parser.add_argument("--env-url", default=inference.ENV_BASE_URL)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="artifacts/eval/baseline_eval.json")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
