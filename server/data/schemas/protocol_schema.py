"""Schema for trial protocol YAML files."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProtocolCriterion(BaseModel):
    """Single inclusion or exclusion rule."""

    model_config = ConfigDict(extra="allow")

    id: str
    description: str
    type: str
    field: str
    operator: str
    values: list[object]
    certainty_always: Optional[Literal["confirmed", "pending", "estimated"]] = None
    weight: float = Field(gt=0)
    is_critical: bool = False
    clarification_available: bool = False
    is_ambiguous: bool = False
    duration_months_min: Optional[int] = None
    unit: Optional[str] = None


class ProtocolAmendment(BaseModel):
    """Protocol amendment metadata."""

    model_config = ConfigDict(extra="forbid")

    amendment_id: str
    trigger_step: int
    description: str
    affects_criterion: str
    change_type: str
    original_values: list[object]
    updated_values: list[object]


class CriteriaGroup(BaseModel):
    """Grouped protocol criteria."""

    model_config = ConfigDict(extra="forbid")

    inclusion: list[ProtocolCriterion]
    exclusion: list[ProtocolCriterion]


class TrialProtocol(BaseModel):
    """Top-level protocol document."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    title: str
    phase: Literal["I", "II", "III", "IV"]
    sponsor: str
    version: str
    amendment_enabled: bool
    max_enrollment: int
    criteria: CriteriaGroup
    amendments: list[ProtocolAmendment] = Field(default_factory=list)

