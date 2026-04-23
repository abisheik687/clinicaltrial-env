"""State model for ClinicalTrialEnv."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .observation import PatientObservation
from .score_serialization import serialize_numeric_score


class TrialState(BaseModel):
    """Current full episode state."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    task_id: Literal["task1", "task2", "task3"]
    episode_number: int = Field(ge=1)
    current_step: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    patient: PatientObservation
    evaluated_criteria: dict[str, str]
    clarifications_used: int = Field(ge=0)
    clarification_budget: int = Field(ge=0)
    amendment_injected: bool = False
    workflow_phase: Literal["screening", "followup_scheduling", "safety_event", "complete", "excluded"] = "screening"
    enrollment_decision: Optional[Literal["enroll", "exclude", "defer"]] = None
    scheduled_followup_day: Optional[int] = None
    safety_event_active: bool = False
    safety_response: Optional[Literal["reschedule", "escalate"]] = None
    cumulative_reward: float = 0.0
    done: bool = False
    termination_reason: Optional[str] = None

    @field_serializer("cumulative_reward")
    def serialize_cumulative_reward(self, value: float) -> float:
        return serialize_numeric_score(value)
