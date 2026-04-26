"""Prompt dataset builder for the GRPO Phase 1 training loop."""

from __future__ import annotations

from typing import Any

from datasets import Dataset

from training.env_client import EnvClient
from training.trajectory_helpers import build_episode_prompt


def build_prompt_dataset(
    env_url: str,
    task_id: str,
    seed_start: int,
    num_episodes: int,
    max_actions: int,
    tokenizer: Any | None = None,
    local_debug_mode: bool = False,
) -> Dataset:
    prompts: list[str] = []
    task_ids: list[str] = []
    seeds: list[int] = []
    client = EnvClient(base_url=env_url)
    for offset in range(num_episodes):
        seed = seed_start + offset
        reset_data = client.reset(task_id=task_id, seed=seed)
        prompts.append(
            build_episode_prompt(
                reset_data["observation"],
                task_id,
                seed,
                max_actions,
                tokenizer=tokenizer,
                local_debug_mode=local_debug_mode,
            )
        )
        task_ids.append(task_id)
        seeds.append(seed)
    return Dataset.from_dict({"prompt": prompts, "task_id": task_ids, "seed": seeds})
