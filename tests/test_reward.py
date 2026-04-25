"""Verifier-centric reward tests."""

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
            reasoning=f"Deterministic verifier test evaluation for {criterion_id}.",
        ),
        confidence_score=0.9,
    )


def test_intermediate_evaluation_has_zero_reward_and_updates_diagnostics() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)

    _, reward, done, _ = env.step(session_id, _correct_eval(env, session_id, "INC-001"))

    assert done is False
    assert reward.total_reward == 0.0
    assert reward.terminal_success is False
    assert reward.diagnostic_metrics["criterion_evaluation_accuracy"] == 1.0


def test_invalid_final_action_before_any_evidence_gets_small_penalty() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task3", seed=44)

    _, reward, done, info = env.step(
        session_id,
        ScreeningAction(action_type=ActionType.DEFER, confidence_score=0.1),
    )

    assert done is False
    assert reward.total_reward == -0.05
    assert reward.invalid_action_penalty == -0.05
    assert info["invalid_action"] is True


def test_terminal_success_returns_positive_reward() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    truth = env.sessions[session_id].__dict__["hidden_case"]["criterion_truth"]

    for criterion_id in truth:
        env.step(session_id, _correct_eval(env, session_id, criterion_id))

    final_action = "enroll" if env.sessions[session_id].__dict__["hidden_case"]["final_eligible"] else "exclude"
    _, reward, done, _ = env.step(
        session_id,
        ScreeningAction(
            action_type=ActionType(final_action),
            final_decision_reason="Completing deterministic terminal verifier test.",
            confidence_score=0.99,
        ),
    )

    assert done is True
    assert reward.is_final is True
    assert reward.total_reward == 1.0
    assert reward.terminal_success is True
    assert reward.unsafe_action is False


def test_unsafe_enrollment_returns_negative_reward() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    hidden = env.sessions[session_id].__dict__["hidden_case"]

    unsafe_case_seed = 45
    _, session_id, _ = env.reset("task1", seed=unsafe_case_seed)
    hidden = env.sessions[session_id].__dict__["hidden_case"]
    if hidden["final_eligible"]:
        _, session_id, _ = env.reset("task1", seed=47)
        hidden = env.sessions[session_id].__dict__["hidden_case"]
    truth = hidden["criterion_truth"]
    first_criterion = next(iter(truth))
    env.step(session_id, _correct_eval(env, session_id, first_criterion))

    _, reward, done, _ = env.step(
        session_id,
        ScreeningAction(
            action_type=ActionType.ENROLL,
            final_decision_reason="Unsafe enrollment test path.",
            confidence_score=0.8,
        ),
    )

    assert done is True
    assert reward.total_reward == -1.0
    assert reward.unsafe_action is True
    assert reward.terminal_success is False
    assert reward.diagnostic_metrics["unsafe_action_rate"] == 1.0


def test_amendment_recovery_metric_requires_rechecking_inc003() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task3", seed=44)

    for criterion_id in ["INC-001", "INC-002", "INC-003", "INC-004", "INC-005", "INC-006"]:
        env.step(session_id, _correct_eval(env, session_id, criterion_id))

    state = env.sessions[session_id]
    reward_before = env.reward_calculator.compute(
        state,
        ScreeningAction(
            action_type=ActionType.EXCLUDE,
            final_decision_reason="Diagnostic-only amendment recovery check.",
            confidence_score=0.5,
        ),
        {},
    )
    env.step(session_id, _correct_eval(env, session_id, "INC-003"))
    state = env.sessions[session_id]
    reward_after = env.reward_calculator.compute(
        state,
        ScreeningAction(
            action_type=ActionType.EXCLUDE,
            final_decision_reason="Diagnostic-only amendment recovery check.",
            confidence_score=0.5,
        ),
        {},
    )

    assert state.amendment_injected is True
    assert reward_before.diagnostic_metrics["amendment_recovery_rate"] <= reward_after.diagnostic_metrics["amendment_recovery_rate"]


def test_reset_state_starts_with_zero_cumulative_reward() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    payload = env.state(session_id).model_dump()

    assert payload["cumulative_reward"] == 0.0


def test_task3_safe_enrollment_requires_followup_and_safety_resolution() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task3", seed=44)
    truth = env.sessions[session_id].__dict__["hidden_case"]["criterion_truth"]

    for criterion_id in ["INC-001", "INC-002", "INC-003"]:
        env.step(session_id, _correct_eval(env, session_id, criterion_id))

    state = env.sessions[session_id]
    assert state.amendment_injected is True

    env.step(session_id, _correct_eval(env, session_id, "INC-003"))
    for criterion_id in ["INC-004", "INC-005", "INC-006", "EXC-001", "EXC-002", "EXC-003", "EXC-004"]:
        env.step(session_id, _correct_eval(env, session_id, criterion_id))

    _, enroll_reward, done_after_enroll, _ = env.step(
        session_id,
        ScreeningAction(
            action_type=ActionType.ENROLL,
            final_decision_reason="Safe enrollment for workflow test.",
            confidence_score=0.95,
        ),
    )

    assert done_after_enroll is False
    assert enroll_reward.total_reward == 0.0
    assert env.sessions[session_id].workflow_phase == "followup_scheduling"

    _, schedule_reward, done_after_schedule, _ = env.step(
        session_id,
        ScreeningAction(
            action_type=ActionType.SCHEDULE_FOLLOWUP,
            followup_day=8,
            confidence_score=0.8,
        ),
    )

    assert done_after_schedule is False
    # SCHEDULE_FOLLOWUP awards +0.3 intermediate shaping bonus when ENABLE_INTERMEDIATE_SHAPING=1 (default)
    assert schedule_reward.total_reward == 0.3
    assert env.sessions[session_id].workflow_phase == "safety_event"

    _, safety_reward, done_final, _ = env.step(
        session_id,
        ScreeningAction(
            action_type=ActionType.HANDLE_SAFETY_EVENT,
            safety_response="escalate",
            confidence_score=0.9,
        ),
    )

    assert done_final is True
    assert safety_reward.total_reward == 1.0
    assert safety_reward.terminal_success is True
    assert safety_reward.diagnostic_metrics["eligibility_component_score"] == 1.0
    assert safety_reward.diagnostic_metrics["scheduling_component_score"] == 1.0
    assert safety_reward.diagnostic_metrics["safety_component_score"] == 1.0
