"""Shared numeric serialization helpers for validator-safe API outputs."""

from __future__ import annotations

from typing import Any


def serialize_numeric_score(value: float) -> float:
    """Round numeric values without changing their sign or interval."""
    return round(float(value), 4)


def serialize_numeric_mapping(mapping: dict[str, float]) -> dict[str, float]:
    """Round all numeric mapping values while preserving keys."""
    return {key: serialize_numeric_score(value) for key, value in mapping.items()}


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
                serialized[key] = serialize_numeric_score(float(item))
            else:
                nested = serialize_nested_scores(item)
                serialized[key] = nested
        return serialized
    if isinstance(value, list):
        return [serialize_nested_scores(item) for item in value]
    if isinstance(value, float):
        return serialize_numeric_score(value)
    return value
