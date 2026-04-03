"""Task 3 definition."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task3Definition:
    task_id: str = "task3"
    name: str = "Ambiguous Gene Therapy Screening with Protocol Amendment"
    protocol_file: str = "trial_c.yaml"
    max_steps: int = 20
    clarification_budget: int = 5
    difficulty: str = "hard"
    max_possible_reward: float = 2.50

