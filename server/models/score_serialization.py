"""Shared score serialization helpers for validator-safe API outputs."""

from __future__ import annotations

from typing import Any


EXTERNAL_SCORE_FLOOR = 0.01
EXTERNAL_SCORE_CEILING = 0.99


def serialize_open_interval_score(value: float) -> float:
    """Clamp score-like numeric values into a strict open interval."""
    return round(min(max(float(value), EXTERNAL_SCORE_FLOOR), EXTERNAL_SCORE_CEILING), 4)


def serialize_open_interval_mapping(mapping: dict[str, float]) -> dict[str, float]:
    """Drop non-positive entries and clamp positive scores for external payloads."""
    serialized: dict[str, float] = {}
    for key, value in mapping.items():
        numeric = float(value)
        if numeric <= 0.0:
            continue
        serialized[key] = serialize_open_interval_score(numeric)
    return serialized


def serialize_nested_scores(value: Any) -> Any:
    """Recursively clamp score-like payloads used in API responses."""
    if isinstance(value, dict):
        serialized: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, (int, float)) and (
                "score" in key
                or "reward" in key
                or "accuracy" in key
                or "bonus" in key
                or "penalty" in key
                or "credit" in key
            ):
                serialized[key] = serialize_open_interval_score(float(item))
            else:
                nested = serialize_nested_scores(item)
                if nested == {}:
                    continue
                serialized[key] = nested
        return serialized
    if isinstance(value, list):
        return [serialize_nested_scores(item) for item in value]
    return value
