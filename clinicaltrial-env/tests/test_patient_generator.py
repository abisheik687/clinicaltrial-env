"""Patient generator tests."""

from pathlib import Path

from server.data.patient_generator import PatientGenerator
from server.data.protocol_loader import ProtocolLoader


def _generator() -> PatientGenerator:
    loader = ProtocolLoader(Path(__file__).resolve().parents[1] / "protocols")
    return PatientGenerator(loader, seed=42)


def test_same_seed_same_patient() -> None:
    generator = _generator()
    patient_a = generator.generate_patient("task1", 42).model_dump()
    patient_b = generator.generate_patient("task1", 42).model_dump()
    assert patient_a == patient_b


def test_different_seeds_different_patients() -> None:
    generator = _generator()
    patient_a = generator.generate_patient("task2", 43).model_dump()
    patient_b = generator.generate_patient("task2", 44).model_dump()
    assert patient_a != patient_b


def test_generated_patient_matches_protocol_schema() -> None:
    generator = _generator()
    patient = generator.generate_patient("task3", 44)
    assert patient.trial_protocol_summary.trial_id == "TRIAL-C-GENE-003"
    assert len(patient.trial_protocol_summary.inclusion_criteria) == 6
    assert len(patient.trial_protocol_summary.exclusion_criteria) == 4


def test_60_percent_eligible_distribution() -> None:
    generator = _generator()
    cases = [generator.generate_case("task1", seed) for seed in range(50)]
    ratio = sum(1 for case in cases if case.final_eligible) / len(cases)
    assert 0.5 <= ratio <= 0.7

