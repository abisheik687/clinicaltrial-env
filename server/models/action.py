"""Action models for ClinicalTrialEnv."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(str, Enum):
    """Available agent action types."""

    INSPECT_PATIENT = "inspect_patient"
    INSPECT_PROTOCOL = "inspect_protocol"
    EVALUATE_CRITERION = "evaluate_criterion"
    ASK_CLARIFICATION = "ask_clarification"
    ENROLL = "enroll"
    EXCLUDE = "exclude"
    DEFER = "defer"
    SCHEDULE_FOLLOWUP = "schedule_followup"
    HANDLE_SAFETY_EVENT = "handle_safety_event"


class CriterionEvaluation(BaseModel):
    """Evaluation payload for a protocol criterion."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    verdict: Literal["met", "not_met", "uncertain"]
    reasoning: str


class ScreeningAction(BaseModel):
    """Validated environment action."""

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    criterion_id: Optional[str] = None
    evaluation: Optional[CriterionEvaluation] = None
    clarification_target: Optional[str] = None
    final_decision_reason: Optional[str] = None
    followup_day: Optional[int] = None
    safety_response: Optional[Literal["reschedule", "escalate"]] = None
    reschedule_day: Optional[int] = None
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)

    @model_validator(mode="after")
    def validate_shape(self) -> "ScreeningAction":
        if self.action_type == ActionType.EVALUATE_CRITERION:
            if not self.criterion_id or not self.evaluation:
                raise ValueError("criterion_id and evaluation are required for evaluate_criterion")
            if self.evaluation.criterion_id != self.criterion_id:
                raise ValueError("evaluation.criterion_id must match criterion_id")
        if self.action_type == ActionType.ASK_CLARIFICATION and not self.clarification_target:
            raise ValueError("clarification_target is required for ask_clarification")
        if self.action_type in {ActionType.ENROLL, ActionType.EXCLUDE} and not self.final_decision_reason:
            raise ValueError("final_decision_reason is required for final decisions")
        if self.action_type == ActionType.SCHEDULE_FOLLOWUP and self.followup_day is None:
            raise ValueError("followup_day is required for schedule_followup")
        if self.action_type == ActionType.HANDLE_SAFETY_EVENT:
            if self.safety_response is None:
                raise ValueError("safety_response is required for handle_safety_event")
            if self.safety_response == "reschedule" and self.reschedule_day is None:
                raise ValueError("reschedule_day is required when safety_response is reschedule")
        return self
