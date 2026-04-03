"""Schema for synthetic internal patient cases."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class HiddenClarification(BaseModel):
    """Hidden value exposed after a clarification request."""

    model_config = ConfigDict(extra="forbid")

    visible_lab_key: Optional[str] = None
    actual_value: Optional[float] = None
    actual_certainty: Optional[Literal["confirmed", "pending", "estimated"]] = None
    medication_name: Optional[str] = None
    is_contraindicated: Optional[bool] = None
    info_message: str


class InternalPatientCase(BaseModel):
    """Internal environment state for a generated patient case."""

    model_config = ConfigDict(extra="forbid")

    observation: dict
    criterion_truth: dict[str, str]
    final_eligible: bool
    critical_exclusion_present: bool
    clarifications: dict[str, HiddenClarification]
    meta: dict[str, object]

