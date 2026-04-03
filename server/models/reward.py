"""Reward models for ClinicalTrialEnv."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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

