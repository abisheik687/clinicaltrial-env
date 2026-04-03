"""Task 2 definition."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task2Definition:
    task_id: str = "task2"
    name: str = "Multi-Criteria Oncology Screening"
    protocol_file: str = "trial_b.yaml"
    max_steps: int = 14
    clarification_budget: int = 2
    difficulty: str = "medium"
    max_possible_reward: float = 1.85

