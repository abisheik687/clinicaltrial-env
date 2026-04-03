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

