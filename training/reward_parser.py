"""Reward parsing and safety fallbacks for the hackathon GRPO loop."""

from __future__ import annotations

import random
from typing import Any

from training.config import FALLBACK_ACTION
from training.trajectory_helpers import normalize_completion_text, parse_trajectory_completion


INVALID_COMPLETION_REWARD = -1.0
INVALID_TRAJECTORY_WEIGHT = 0.1


def reward_noise() -> float:
    return random.random() * 0.1 - 0.05


def safe_parse_trajectory(completion_text: Any, max_actions: int) -> list[dict[str, Any]]:
    try:
        parsed = parse_trajectory_completion(normalize_completion_text(completion_text), max_actions)
        if not parsed:
            raise ValueError("Empty parsed trajectory")
        return parsed
    except Exception:
        return [FALLBACK_ACTION]


def weighted_reward(raw_reward: float, final_payload: dict[str, Any]) -> tuple[float, float, bool]:
    reward_payload = final_payload.get("reward", {}) if isinstance(final_payload, dict) else {}
    info_payload = final_payload.get("info", {}) if isinstance(final_payload, dict) else {}
    invalid_or_unsafe = bool(reward_payload.get("unsafe_action")) or bool(info_payload.get("invalid_action"))
    weight = INVALID_TRAJECTORY_WEIGHT if invalid_or_unsafe else 1.0
    return float(raw_reward) * weight, weight, invalid_or_unsafe


def fallback_reward() -> float:
    return INVALID_COMPLETION_REWARD + reward_noise()
