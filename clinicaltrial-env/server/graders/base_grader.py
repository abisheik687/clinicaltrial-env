"""Abstract task grader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGrader(ABC):
    """Base class for deterministic episode graders."""

    @abstractmethod
    def grade(self, truth: dict[str, str], evaluated: dict[str, str], final_action: str, context: dict[str, Any]) -> dict[str, Any]:
        """Return normalized episode score and component metadata."""

