"""Reward calculation edge cases."""

from server.environment.env import ClinicalTrialEnv
from server.models.action import ActionType, CriterionEvaluation, ScreeningAction


def test_repeat_evaluation_penalty() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    action = ScreeningAction(
        action_type=ActionType.EVALUATE_CRITERION,
        criterion_id="INC-001",
        evaluation=CriterionEvaluation(
            criterion_id="INC-001",
            verdict=env.sessions[session_id].__dict__["hidden_case"]["criterion_truth"]["INC-001"],
            reasoning="Initial evaluation for repeated criterion penalty test case.",
        ),
        confidence_score=0.9,
    )
    env.step(session_id, action)
    _, reward, _, _ = env.step(session_id, action)
    assert reward.penalty >= 0.05


def test_unnecessary_clarification_penalty() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task2", seed=43)
    _, reward, _, _ = env.step(
        session_id,
        ScreeningAction(
            action_type=ActionType.ASK_CLARIFICATION,
            clarification_target="INC-001",
            confidence_score=0.4,
        ),
    )
    assert reward.penalty >= 0.10


def test_final_defer_penalty() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task3", seed=44)
    _, reward, done, _ = env.step(
        session_id,
        ScreeningAction(action_type=ActionType.DEFER, confidence_score=0.1),
    )
    assert done is True
    assert reward.penalty >= 0.20


def test_total_reward_stays_inside_open_interval_on_bad_path() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task3", seed=44)
    _, reward, _, _ = env.step(
        session_id,
        ScreeningAction(action_type=ActionType.DEFER, confidence_score=0.1),
    )
    assert 0.0 < reward.total_reward < 1.0


def test_total_reward_stays_inside_open_interval_on_good_path() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    truth = env.sessions[session_id].__dict__["hidden_case"]["criterion_truth"]
    for criterion_id, verdict in truth.items():
        env.step(
            session_id,
            ScreeningAction(
                action_type=ActionType.EVALUATE_CRITERION,
                criterion_id=criterion_id,
                evaluation=CriterionEvaluation(
                    criterion_id=criterion_id,
                    verdict=verdict,
                    reasoning=f"Strict interval reward test for {criterion_id} with detailed reasoning.",
                ),
                confidence_score=0.95,
            ),
        )
    final_action = "enroll" if env.sessions[session_id].__dict__["hidden_case"]["final_eligible"] else "exclude"
    _, reward, _, _ = env.step(
        session_id,
        ScreeningAction(
            action_type=ActionType(final_action),
            final_decision_reason="Completing a deterministic positive-path reward test.",
            confidence_score=0.99,
        ),
    )
    assert 0.0 < reward.total_reward < 1.0
    assert 0.0 < env.sessions[session_id].cumulative_reward < 1.0


def test_rewards_survive_two_decimal_rounding_inside_open_interval() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task3", seed=44)
    _, reward, _, _ = env.step(
        session_id,
        ScreeningAction(action_type=ActionType.DEFER, confidence_score=0.1),
    )
    rounded_reward = float(f"{reward.total_reward:.2f}")
    rounded_cumulative = float(f"{env.sessions[session_id].cumulative_reward:.2f}")
    assert 0.0 < rounded_reward < 1.0
    assert 0.0 < rounded_cumulative < 1.0


def test_reward_model_dump_filters_closed_interval_secondary_fields() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task3", seed=44)
    _, reward, _, _ = env.step(
        session_id,
        ScreeningAction(action_type=ActionType.DEFER, confidence_score=0.1),
    )
    payload = reward.model_dump()
    assert 0.0 < payload["total_reward"] < 1.0
    assert 0.0 < payload["eligibility_accuracy"] < 1.0
    assert 0.0 < payload["efficiency_bonus"] < 1.0
    assert 0.0 < payload["penalty"] < 1.0
    assert all(0.0 < value < 1.0 for value in payload["partial_credit"].values())


def test_state_model_dump_serializes_reset_cumulative_reward_inside_open_interval() -> None:
    env = ClinicalTrialEnv()
    _, session_id, _ = env.reset("task1", seed=42)
    payload = env.state(session_id).model_dump()
    assert 0.0 < payload["cumulative_reward"] < 1.0
