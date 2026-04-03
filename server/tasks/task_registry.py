"""Task registry and shared helpers."""

from dataclasses import asdict

from server.tasks.task1 import Task1Definition
from server.tasks.task2 import Task2Definition
from server.tasks.task3 import Task3Definition


TASKS = {
    "task1": Task1Definition(),
    "task2": Task2Definition(),
    "task3": Task3Definition(),
}


def get_task_definition(task_id: str) -> object:
    """Return the static task definition."""
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def task_info(task_id: str) -> dict[str, object]:
    """Serialize task metadata for API responses."""
    return asdict(get_task_definition(task_id))

