"""Core environment tests."""

from server.environment.env import ClinicalTrialEnv
from server.models.action import ActionType, CriterionEvaluation, ScreeningAction


def _correct_eval(env: ClinicalTrialEnv, session_id: str, criterion_id: str) -> ScreeningAction:
    truth = env.sessions[session_id].__dict__["hidden_case"]["criterion_truth"][criterion_id]
    return ScreeningAction(
        action_type=ActionType.EVALUATE_CRITERION,
        criterion_id=criterion_id,
        evaluation=CriterionEvaluation(
            criterion_id=criterion_id,
            verdict=truth,
            reasoning=f"Deterministic test evaluation for {criterion_id} based on visible state.",
        ),
        confidence_score=0.9,
    )


def test_reset_returns_valid_observation() -> None:
    env = ClinicalTrialEnv()
    observation, session_id, task_info = env.reset("task1", seed=42)
    assert observation.patient_id
    assert observation.demographics.age >= 0
    assert "systolic_bp" in observation.lab_values
    assert session_id
    assert task_info["max_steps"] == 8


def test_step_returns_valid_reward() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    observation, reward, done, _ = env.step(session_id, _correct_eval(env, session_id, "INC-001"))
    assert 0.0 <= reward.total_reward <= 1.0
    assert observation.step_number == 1
    assert done is False


def test_episode_terminates_at_max_steps() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    for _ in range(8):
        env.step(
            session_id,
            ScreeningAction(
                action_type=ActionType.EVALUATE_CRITERION,
                criterion_id="INC-001",
                evaluation=CriterionEvaluation(
                    criterion_id="INC-001",
                    verdict="met",
                    reasoning="Repeated evaluation to consume the step budget in a controlled test.",
                ),
                confidence_score=0.5,
            ),
        )
    assert env.sessions[session_id].done is True
    assert env.sessions[session_id].termination_reason == "max_steps_reached"


def test_session_isolation() -> None:
    env = ClinicalTrialEnv()
    _, session_a, _ = env.reset("task1", seed=42)
    _, session_b, _ = env.reset("task2", seed=43)
    env.step(session_a, _correct_eval(env, session_a, "INC-001"))
    assert env.sessions[session_a].current_step == 1
    assert env.sessions[session_b].current_step == 0


def test_amendment_injected_at_step_6_task3() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task3", seed=44)
    for criterion_id in ["INC-001", "INC-002", "INC-003", "INC-004", "INC-005", "INC-006"]:
        env.step(session_id, _correct_eval(env, session_id, criterion_id))
    state = env.sessions[session_id]
    assert state.amendment_injected is True
    assert state.patient.trial_protocol_summary.amendment_active is True
    assert "Amendment A1" in (state.patient.trial_protocol_summary.amendment_description or "")


def test_done_flag_set_on_final_action() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    env.step(session_id, _correct_eval(env, session_id, "INC-001"))
    _, reward, done, _ = env.step(
        session_id,
        ScreeningAction(
            action_type=ActionType.EXCLUDE,
            final_decision_reason="Test final action path.",
            confidence_score=0.8,
        ),
    )
    assert done is True
    assert reward.is_final is True

