"""Training helper tests for trajectory planning."""

from training.trajectory_helpers import build_episode_prompt, parse_trajectory_completion


def test_build_episode_prompt_mentions_finals_workflow() -> None:
    observation = {
        "patient_id": "4f0f64fa-9c2a-4957-95b7-c08fca6a4048",
        "step_number": 0,
        "steps_remaining": 20,
        "demographics": {"age": 12, "sex": "Other", "weight_kg": 37.0, "height_cm": 161.0},
        "diagnosis": {
            "primary_condition": "Rett syndrome",
            "icd10_code": "F84.2",
            "disease_stage": None,
            "diagnosis_date": "2017-10-10",
        },
        "lab_values": {
            "css_score": {"value": 23.6, "certainty": "estimated", "unit": "score"},
        },
        "current_medications": [],
        "trial_protocol_summary": {
            "trial_id": "TRIAL-C-GENE-003",
            "title": "Phase I/II AAV Gene Therapy for Rett Syndrome",
            "phase": "I",
            "amendment_active": False,
            "amendment_description": None,
            "inclusion_criteria": [
                {"criterion_id": "INC-003", "clarification_available": True, "is_ambiguous": True},
            ],
            "exclusion_criteria": [],
        },
        "operational_state": {
            "workflow_phase": "screening",
            "followup_window_start": 7,
            "followup_window_end": 10,
            "amendment_review_required": False,
            "safety_event_active": False,
        },
        "info_message": "Protocol amendment pending review.",
    }

    prompt = build_episode_prompt(observation, task_id="task3", seed=44, max_actions=10)

    assert "schedule_followup" in prompt
    assert "handle_safety_event" in prompt
    assert "investigator escalation" in prompt


def test_parse_trajectory_completion_rejects_disallowed_actions() -> None:
    completion = '{"trajectory":[{"action_type":"inspect_protocol"}]}'

    try:
        parse_trajectory_completion(completion, max_actions=5)
    except ValueError as exc:
        assert "Unsupported trajectory action_type" in str(exc)
    else:
        raise AssertionError("Expected parse_trajectory_completion to reject inspect_protocol.")
