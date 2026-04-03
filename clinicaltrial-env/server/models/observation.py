"""Observation models for ClinicalTrialEnv."""

from __future__ import annotations

from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class Demographics(BaseModel):
    """Patient demographic information."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0, le=120)
    sex: Literal["M", "F", "Other"]
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)

    @computed_field(return_type=float)
    @property
    def bmi(self) -> float:
        height_m = self.height_cm / 100.0
        return round(self.weight_kg / (height_m * height_m), 2)


class Diagnosis(BaseModel):
    """Diagnosis block included in the observation."""

    model_config = ConfigDict(extra="forbid")

    primary_condition: str
    icd10_code: str
    disease_stage: Optional[str] = None
    diagnosis_date: str


class LabValue(BaseModel):
    """Single lab value with certainty metadata."""

    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str
    reference_range: tuple[float, float]
    certainty: Literal["confirmed", "pending", "estimated"]

    @computed_field(return_type=bool)
    @property
    def is_abnormal(self) -> bool:
        low, high = self.reference_range
        return not (low <= self.value <= high)


class Medication(BaseModel):
    """Current medication entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dose_mg: float = Field(ge=0)
    frequency: str
    is_contraindicated: Optional[bool] = None


class CriterionSummary(BaseModel):
    """Compact protocol criterion summary shown to the agent."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str
    is_ambiguous: bool
    clarification_available: bool


class TrialProtocolSummary(BaseModel):
    """Protocol metadata exposed in the observation."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    title: str
    phase: Literal["I", "II", "III", "IV"]
    inclusion_criteria: list[CriterionSummary]
    exclusion_criteria: list[CriterionSummary]
    amendment_active: bool
    amendment_description: Optional[str] = None


class PatientObservation(BaseModel):
    """Full environment observation."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    demographics: Demographics
    diagnosis: Diagnosis
    lab_values: dict[str, LabValue]
    current_medications: list[Medication]
    trial_protocol_summary: TrialProtocolSummary
    step_number: int = Field(ge=0)
    steps_remaining: int = Field(ge=0)
    previous_actions: list[str]
    info_message: Optional[str] = None

    @field_validator("patient_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        UUID(value)
        return value

