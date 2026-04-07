"""Abstract task grader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGrader(ABC):
    """Base class for deterministic episode graders."""

    SCORE_EPSILON = 0.01

    @abstractmethod
    def grade(self, truth: dict[str, str], evaluated: dict[str, str], final_action: str, context: dict[str, Any]) -> dict[str, Any]:
        """Return normalized episode score and component metadata."""

    def clamp_open_unit_interval(self, score: float) -> float:
        """Keep task scores strictly inside (0, 1) for validator compatibility."""
        return min(max(score, self.SCORE_EPSILON), 1.0 - self.SCORE_EPSILON)
