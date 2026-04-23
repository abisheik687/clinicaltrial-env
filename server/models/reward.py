"""Reward models for ClinicalTrialEnv."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .score_serialization import serialize_numeric_mapping, serialize_numeric_score


class EnrollmentReward(BaseModel):
    """Verifier-centric reward returned after each environment step."""

    model_config = ConfigDict(extra="forbid")

    total_reward: float = Field(ge=-1.0, le=1.0)
    terminal_success: bool = False
    unsafe_action: bool = False
    invalid_action_penalty: float = Field(ge=-0.25, le=0.0, default=0.0)
    diagnostic_metrics: dict[str, float] = Field(default_factory=dict)
    verifier_feedback: str
    is_final: bool

    @field_serializer("total_reward", "invalid_action_penalty")
    def serialize_score_fields(self, value: float) -> float:
        return serialize_numeric_score(value)

    @field_serializer("diagnostic_metrics")
    def serialize_partial_credit(self, value: dict[str, float]) -> dict[str, float]:
        return serialize_numeric_mapping(value)
