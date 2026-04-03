"""Task 1 definition."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task1Definition:
    task_id: str = "task1"
    name: str = "Single Criterion Hypertension Screening"
    protocol_file: str = "trial_a.yaml"
    max_steps: int = 8
    clarification_budget: int = 0
    difficulty: str = "easy"
    max_possible_reward: float = 1.40

