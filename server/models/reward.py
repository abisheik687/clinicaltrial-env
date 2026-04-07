"""Reward models for ClinicalTrialEnv."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .score_serialization import serialize_open_interval_mapping, serialize_open_interval_score


class EnrollmentReward(BaseModel):
    """Composite reward returned after each environment step."""

    model_config = ConfigDict(extra="forbid")

    total_reward: float = Field(ge=0.0, le=1.0)
    eligibility_accuracy: float = Field(ge=0.0, le=1.0)
    efficiency_bonus: float = Field(ge=0.0, le=0.3)
    penalty: float = Field(ge=0.0, le=1.0)
    partial_credit: dict[str, float]
    grader_feedback: str
    is_final: bool

    @field_serializer("total_reward", "eligibility_accuracy", "efficiency_bonus", "penalty")
    def serialize_score_fields(self, value: float) -> float:
        return serialize_open_interval_score(value)

    @field_serializer("partial_credit")
    def serialize_partial_credit(self, value: dict[str, float]) -> dict[str, float]:
        return serialize_open_interval_mapping(value)
