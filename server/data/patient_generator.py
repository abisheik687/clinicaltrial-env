"""Deterministic synthetic patient generation."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from typing import Any
from uuid import UUID, uuid5

import numpy as np
from faker import Faker

from server.data.protocol_loader import ProtocolLoader
from server.data.schemas.patient_schema import HiddenClarification, InternalPatientCase
from server.models.observation import (
    CriterionSummary,
    Demographics,
    Diagnosis,
    LabValue,
    Medication,
    PatientObservation,
    TrialProtocolSummary,
)
from server.tasks.task_registry import get_task_definition


COMMON_MEDS = ["metformin", "atorvastatin", "levothyroxine", "omeprazole", "albuterol", "vitamin d"]
ACE_INHIBITORS = ["lisinopril", "enalapril", "ramipril"]
STEROIDS = ["prednisone", "dexamethasone", "methylprednisolone"]
ONCOLOGY_MEDS = ["rituximab", "cyclophosphamide", "allopurinol"]
SEIZURE_MEDS = ["levetiracetam", "clobazam", "valproate"]
UUID_NAMESPACE = UUID("12345678-1234-5678-1234-567812345678")


class GeneratedCase:
    """Convenience container for generated episode data."""

    def __init__(
        self,
        observation_model: PatientObservation,
        criterion_truth: dict[str, str],
        final_eligible: bool,
        critical_exclusion_present: bool,
        clarifications: dict[str, HiddenClarification],
        meta: dict[str, Any],
    ) -> None:
        self.observation_model = observation_model
        self.criterion_truth = criterion_truth
        self.final_eligible = final_eligible
        self.critical_exclusion_present = critical_exclusion_present
        self.clarifications = clarifications
        self.meta = meta

    def to_state_payload(self) -> dict[str, Any]:
        """Serialize internals for hidden state storage."""
        return {
            "criterion_truth": deepcopy(self.criterion_truth),
            "final_eligible": self.final_eligible,
            "critical_exclusion_present": self.critical_exclusion_present,
            "clarifications": {k: v.model_dump() for k, v in self.clarifications.items()},
            "meta": deepcopy(self.meta),
        }


class PatientGenerator:
    """Generate deterministic, protocol-aligned patient cases."""

    def __init__(self, protocol_loader: ProtocolLoader, seed: int = 42) -> None:
        self.protocol_loader = protocol_loader
        self.seed = seed

    def generate_patient(self, task_id: str, seed: int) -> PatientObservation:
        """Generate only the public observation."""
        return self.generate_case(task_id, seed).observation_model

    def generate_case(self, task_id: str, seed: int) -> GeneratedCase:
        """Generate the public observation plus hidden evaluation truth."""
        task = get_task_definition(task_id)
        protocol = self.protocol_loader.load(task.protocol_file)
        faker = Faker()
        faker.seed_instance(seed + self.seed)
        rng = np.random.default_rng(seed + self.seed * 97)
        eligible = (seed % 10) < 6
        patient_id = str(uuid5(UUID_NAMESPACE, f"{task_id}-{seed % 50}"))

        pool_index = seed % 50
        if task_id == "task1":
            case = self._task1_case(protocol, rng, patient_id, eligible, pool_index)
        elif task_id == "task2":
            case = self._task2_case(protocol, rng, patient_id, eligible, pool_index)
        else:
            case = self._task3_case(protocol, rng, patient_id, eligible, pool_index)

        observation_model = PatientObservation.model_validate(case.observation)
        return GeneratedCase(
            observation_model=observation_model,
            criterion_truth=case.criterion_truth,
            final_eligible=case.final_eligible,
            critical_exclusion_present=case.critical_exclusion_present,
            clarifications=case.clarifications,
            meta=case.meta,
        )

    def _protocol_summary(self, protocol, ambiguous: set[str], clarifiable: set[str]) -> TrialProtocolSummary:
        return TrialProtocolSummary(
            trial_id=protocol.trial_id,
            title=protocol.title,
            phase=protocol.phase,
            inclusion_criteria=[
                CriterionSummary(
                    criterion_id=criterion.id,
                    description=criterion.description,
                    is_ambiguous=criterion.id in ambiguous,
                    clarification_available=criterion.id in clarifiable,
                )
                for criterion in protocol.criteria.inclusion
            ],
            exclusion_criteria=[
                CriterionSummary(
                    criterion_id=criterion.id,
                    description=criterion.description,
                    is_ambiguous=criterion.id in ambiguous,
                    clarification_available=criterion.id in clarifiable,
                )
                for criterion in protocol.criteria.exclusion
            ],
            amendment_active=False,
            amendment_description=None,
        )

    def _task1_case(self, protocol, rng: np.random.Generator, patient_id: str, eligible: bool, pool_index: int) -> InternalPatientCase:
        age = int(np.clip(rng.normal(52, 14), 18, 82))
        systolic = float(np.clip(rng.normal(155, 15), 120, 210))
        egfr = float(np.clip(rng.normal(75, 20), 10, 120))
        diagnosis_date = date.today() - timedelta(days=int(rng.integers(220, 1800)))
        meds = self._make_meds(rng, 2, 4, COMMON_MEDS)

        truth = {
            "INC-001": "met" if 18 <= age <= 75 else "not_met",
            "INC-002": "met" if (date.today() - diagnosis_date).days >= 180 else "not_met",
            "INC-003": "met" if 140 <= systolic <= 180 else "not_met",
            "EXC-001": "met" if egfr < 30 else "not_met",
            "EXC-002": "not_met",
        }
        if not eligible:
            violation = ["INC-001", "INC-003", "EXC-001", "EXC-002"][int(rng.integers(0, 4))]
            if violation == "INC-001":
                age = 80
                truth["INC-001"] = "not_met"
            elif violation == "INC-003":
                systolic = 132.0
                truth["INC-003"] = "not_met"
            elif violation == "EXC-001":
                egfr = 24.0
                truth["EXC-001"] = "met"
            else:
                meds.append(Medication(name="lisinopril", dose_mg=10.0, frequency="daily", is_contraindicated=True))
                truth["EXC-002"] = "met"

        observation = {
            "patient_id": patient_id,
            "demographics": self._demographics_payload(Demographics(
                age=age,
                sex=_sample_sex(rng),
                weight_kg=round(float(np.clip(rng.normal(82, 16), 45, 140)), 1),
                height_cm=round(float(np.clip(rng.normal(171, 10), 145, 200)), 1),
            )),
            "diagnosis": Diagnosis(
                primary_condition="Essential hypertension",
                icd10_code="I10",
                disease_stage=None,
                diagnosis_date=diagnosis_date.isoformat(),
            ).model_dump(),
            "lab_values": {
                "systolic_bp": self._lab_payload(LabValue(value=round(systolic, 1), unit="mmHg", reference_range=(90.0, 120.0), certainty="confirmed")),
                "egfr": self._lab_payload(LabValue(value=round(egfr, 1), unit="mL/min/1.73m2", reference_range=(60.0, 120.0), certainty="confirmed")),
            },
            "current_medications": [med.model_dump() for med in meds],
            "trial_protocol_summary": self._protocol_summary(protocol, set(), set()).model_dump(),
            "step_number": 0,
            "steps_remaining": 8,
            "previous_actions": [],
            "info_message": "Evaluate all 5 criteria before making an enrollment decision.",
        }
        final_eligible = all(truth[key] == "met" for key in ("INC-001", "INC-002", "INC-003")) and all(
            truth[key] == "not_met" for key in ("EXC-001", "EXC-002")
        )
        return InternalPatientCase(
            observation=observation,
            criterion_truth=truth,
            final_eligible=final_eligible,
            critical_exclusion_present=truth["EXC-001"] == "met",
            clarifications={},
            meta={"task_id": "task1", "pool_index": pool_index},
        )

    def _task2_case(self, protocol, rng: np.random.Generator, patient_id: str, eligible: bool, pool_index: int) -> InternalPatientCase:
        age = int(np.clip(rng.normal(54, 8), 18, 72))
        anc_actual = float(np.clip(rng.normal(2.5, 1.0), 0.1, 8.0))
        platelets_actual = float(np.clip(rng.normal(160, 55), 20, 400))
        anc_visible = anc_actual
        anc_certainty = "confirmed"
        clarifications: dict[str, HiddenClarification] = {}
        if rng.random() < 0.45:
            anc_certainty = "pending"
            anc_visible = round(max(0.2, anc_actual - float(rng.uniform(0.2, 0.7))), 2)
            clarifications["INC-004"] = HiddenClarification(
                visible_lab_key="anc",
                actual_value=round(anc_actual, 2),
                actual_certainty="confirmed",
                info_message="ANC result finalized from the reference laboratory.",
            )
        ecog = int(rng.integers(0, 3))
        active_cns = False
        prior_car_t = False
        autoimmune = False
        measurable_disease = True
        meds = self._make_meds(rng, 2, 5, COMMON_MEDS + ONCOLOGY_MEDS)
        steroid_equivalent = 0.0
        if not eligible or rng.random() < 0.35:
            steroid_name = STEROIDS[int(rng.integers(0, len(STEROIDS)))]
            steroid_dose = 4.0 if eligible else 20.0
            meds.append(
                Medication(
                    name=steroid_name,
                    dose_mg=steroid_dose,
                    frequency="daily",
                    is_contraindicated=steroid_dose > 10,
                )
            )
            steroid_equivalent = self._prednisone_equivalent(steroid_name, steroid_dose, "daily")

        truth = {
            "INC-001": "met" if 18 <= age <= 65 else "not_met",
            "INC-002": "met",
            "INC-003": "met" if ecog <= 2 else "not_met",
            "INC-004": "met" if anc_actual >= 1.0 and platelets_actual >= 75 else "not_met",
            "INC-005": "met" if measurable_disease else "not_met",
            "EXC-001": "met" if active_cns else "not_met",
            "EXC-002": "met" if prior_car_t else "not_met",
            "EXC-003": "met" if autoimmune else "not_met",
            "EXC-004": "met" if steroid_equivalent > 10 else "not_met",
        }
        if not eligible:
            violation = ["INC-001", "INC-004", "EXC-001", "EXC-002", "EXC-003", "EXC-004"][int(rng.integers(0, 6))]
            if violation == "INC-001":
                age = 69
                truth["INC-001"] = "not_met"
            elif violation == "INC-004":
                anc_actual = 0.6
                platelets_actual = 64.0
                truth["INC-004"] = "not_met"
                if "INC-004" in clarifications:
                    clarifications["INC-004"].actual_value = anc_actual
            elif violation == "EXC-001":
                active_cns = True
                truth["EXC-001"] = "met"
            elif violation == "EXC-002":
                prior_car_t = True
                truth["EXC-002"] = "met"
            elif violation == "EXC-003":
                autoimmune = True
                truth["EXC-003"] = "met"
            else:
                truth["EXC-004"] = "met"

        observation = {
            "patient_id": patient_id,
            "demographics": self._demographics_payload(Demographics(
                age=age,
                sex=_sample_sex(rng),
                weight_kg=round(float(np.clip(rng.normal(76, 14), 42, 130)), 1),
                height_cm=round(float(np.clip(rng.normal(168, 11), 145, 195)), 1),
            )),
            "diagnosis": Diagnosis(
                primary_condition="Diffuse Large B-Cell Lymphoma",
                icd10_code="C83.3",
                disease_stage="Relapsed measurable disease",
                diagnosis_date=(date.today() - timedelta(days=int(rng.integers(120, 1800)))).isoformat(),
            ).model_dump(),
            "lab_values": {
                "anc": self._lab_payload(LabValue(value=round(anc_visible, 2), unit="x10^9/L", reference_range=(1.0, 8.0), certainty=anc_certainty)),
                "platelets": self._lab_payload(LabValue(value=round(platelets_actual, 1), unit="x10^9/L", reference_range=(75.0, 400.0), certainty="confirmed")),
                "ecog_status": self._lab_payload(LabValue(value=float(ecog), unit="score", reference_range=(0.0, 2.0), certainty="confirmed")),
            },
            "current_medications": [med.model_dump() for med in meds],
            "trial_protocol_summary": self._protocol_summary(protocol, set(), {"INC-004"} if "INC-004" in clarifications else set()).model_dump(),
            "step_number": 0,
            "steps_remaining": 14,
            "previous_actions": [],
            "info_message": "Criterion INC-004 combines ANC and platelet requirements.",
        }
        final_eligible = all(truth[key] == "met" for key in ("INC-001", "INC-002", "INC-003", "INC-004", "INC-005")) and all(
            truth[key] == "not_met" for key in ("EXC-001", "EXC-002", "EXC-003", "EXC-004")
        )
        return InternalPatientCase(
            observation=observation,
            criterion_truth=truth,
            final_eligible=final_eligible,
            critical_exclusion_present=truth["EXC-001"] == "met",
            clarifications=clarifications,
            meta={
                "task_id": "task2",
                "pool_index": pool_index,
                "drug_interaction_case": steroid_equivalent > 0,
                "steroid_equivalent": round(steroid_equivalent, 2),
            },
        )

    def _task3_case(self, protocol, rng: np.random.Generator, patient_id: str, eligible: bool, pool_index: int) -> InternalPatientCase:
        age = int(np.clip(rng.normal(18, 10), 4, 48))
        weight = round(float(np.clip(rng.normal(30, 12), 10, 80)), 1)
        severity_actual = float(np.clip(rng.normal(11.5 if not eligible else 18.0, 4.0), 6, 40))
        severity_visible = round(severity_actual - float(rng.uniform(0.5, 1.5)), 1)
        alt = float(np.clip(rng.normal(45, 20), 10, 220))
        ast = float(np.clip(rng.normal(42, 18), 10, 220))
        ul_alt = 55.0
        ul_ast = 45.0
        seizure_uncontrolled = rng.random() < (0.15 if eligible else 0.45)
        hypersensitivity = rng.random() < (0.05 if eligible else 0.20)
        life_expectancy_months = 24 if eligible else (8 if rng.random() < 0.35 else 18)
        clarifications = {
            "INC-003": HiddenClarification(
                visible_lab_key="css_score",
                actual_value=round(severity_actual, 1),
                actual_certainty="confirmed",
                info_message="Rett CSS score confirmed after adjudication review.",
            ),
            "EXC-001": HiddenClarification(info_message="Neurology review clarified seizure control status."),
            "EXC-002": HiddenClarification(
                medication_name="aav_vector_screen",
                is_contraindicated=hypersensitivity,
                info_message="AAV vector hypersensitivity workup completed.",
            ),
        }
        if not eligible:
            violation = ["INC-003", "INC-005", "INC-006", "EXC-001", "EXC-002", "EXC-004"][int(rng.integers(0, 6))]
            if violation == "INC-003":
                severity_actual = 11.0
                clarifications["INC-003"].actual_value = 11.0
            elif violation == "INC-005":
                alt = 190.0
            elif violation == "INC-006":
                weight = 12.2
            elif violation == "EXC-001":
                seizure_uncontrolled = True
            elif violation == "EXC-002":
                hypersensitivity = True
                clarifications["EXC-002"].is_contraindicated = True
            else:
                life_expectancy_months = 8

        truth = {
            "INC-001": "met" if 4 <= age <= 45 else "not_met",
            "INC-002": "met",
            "INC-003": "met" if 12 <= severity_actual <= 36 else "not_met",
            "INC-004": "met",
            "INC-005": "met" if alt <= 3 * ul_alt and ast <= 3 * ul_ast else "not_met",
            "INC-006": "met" if weight >= 13 else "not_met",
            "EXC-001": "met" if seizure_uncontrolled else "not_met",
            "EXC-002": "met" if hypersensitivity else "not_met",
            "EXC-003": "not_met",
            "EXC-004": "met" if life_expectancy_months < 12 else "not_met",
        }

        observation = {
            "patient_id": patient_id,
            "demographics": self._demographics_payload(Demographics(
                age=age,
                sex=_sample_sex(rng),
                weight_kg=weight,
                height_cm=round(float(np.clip(rng.normal(135 if age < 12 else 160, 12), 90, 195)), 1),
            )),
            "diagnosis": Diagnosis(
                primary_condition="Rett syndrome",
                icd10_code="F84.2",
                disease_stage=None,
                diagnosis_date=(date.today() - timedelta(days=int(rng.integers(300, 5000)))).isoformat(),
            ).model_dump(),
            "lab_values": {
                "mecp2_mutation": self._lab_payload(LabValue(value=1.0, unit="binary", reference_range=(1.0, 1.0), certainty="estimated")),
                "css_score": self._lab_payload(LabValue(value=round(severity_visible, 1), unit="score", reference_range=(12.0, 36.0), certainty="pending")),
                "alt": self._lab_payload(LabValue(value=round(alt, 1), unit="U/L", reference_range=(0.0, ul_alt), certainty="confirmed")),
                "ast": self._lab_payload(LabValue(value=round(ast, 1), unit="U/L", reference_range=(0.0, ul_ast), certainty="confirmed")),
            },
            "current_medications": [med.model_dump() for med in self._make_meds(rng, 2, 5, COMMON_MEDS + SEIZURE_MEDS)],
            "trial_protocol_summary": self._protocol_summary(protocol, {"INC-002", "INC-003", "EXC-001"}, {"INC-003", "EXC-001", "EXC-002"}).model_dump(),
            "step_number": 0,
            "steps_remaining": 20,
            "previous_actions": [],
            "info_message": "Watch for ambiguous criteria and possible protocol changes.",
        }
        final_eligible = all(truth[key] == "met" for key in ("INC-001", "INC-002", "INC-003", "INC-004", "INC-005", "INC-006")) and all(
            truth[key] == "not_met" for key in ("EXC-001", "EXC-002", "EXC-003", "EXC-004")
        )
        return InternalPatientCase(
            observation=observation,
            criterion_truth=truth,
            final_eligible=final_eligible,
            critical_exclusion_present=truth["EXC-001"] == "met" or truth["EXC-004"] == "met",
            clarifications=clarifications,
            meta={
                "task_id": "task3",
                "pool_index": pool_index,
                "severity_actual": round(severity_actual, 1),
                "pre_amendment_truth": truth["INC-003"],
                "post_amendment_truth": "met" if 10 <= severity_actual <= 36 else "not_met",
            },
        )

    def _make_meds(self, rng: np.random.Generator, min_count: int, max_count: int, pool: list[str]) -> list[Medication]:
        count = int(rng.integers(min_count, max_count + 1))
        picks = rng.choice(pool, size=min(count, len(pool)), replace=False)
        meds: list[Medication] = []
        for med in picks:
            dose = round(float(np.clip(rng.normal(20, 10), 2, 120)), 1)
            frequency = ["daily", "bid", "tid"][int(rng.integers(0, 3))]
            meds.append(Medication(name=str(med), dose_mg=dose, frequency=frequency, is_contraindicated=None))
        return meds

    def _prednisone_equivalent(self, name: str, dose_mg: float, frequency: str) -> float:
        multiplier = {"daily": 1.0, "bid": 2.0, "tid": 3.0}.get(frequency, 1.0)
        conversion = {"prednisone": 1.0, "methylprednisolone": 1.25, "dexamethasone": 6.67}
        return dose_mg * multiplier * conversion.get(name, 1.0)

    def _demographics_payload(self, demographics: Demographics) -> dict[str, object]:
        return demographics.model_dump(exclude={"bmi"})

    def _lab_payload(self, lab: LabValue) -> dict[str, object]:
        return lab.model_dump(exclude={"is_abnormal"})


def _sample_sex(rng: np.random.Generator) -> str:
    return ["M", "F", "Other"][int(rng.integers(0, 3))]
