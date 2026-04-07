"""Reward shaping and episode scoring."""

from server.graders.task1_grader import Task1Grader
from server.graders.task2_grader import Task2Grader
from server.graders.task3_grader import Task3Grader
from server.models.action import ActionType, ScreeningAction
from server.models.reward import EnrollmentReward


class RewardCalculator:
    """Compute step rewards and final episode rewards."""

    SCORE_EPSILON = 1e-4

    def __init__(self) -> None:
        self.graders = {"task1": Task1Grader(), "task2": Task2Grader(), "task3": Task3Grader()}
        self.step_reward_scale = {"task1": 0.10, "task2": 0.125, "task3": 0.15}
        self.max_possible = {"task1": 1.40, "task2": 1.85, "task3": 2.50}

    def compute(self, state, action: ScreeningAction, info: dict[str, object]) -> EnrollmentReward:
        hidden = state.__dict__.get("hidden_case", {})
        truth = hidden["criterion_truth"]
        partial: dict[str, float] = {}
        raw_sum = 0.0
        eligibility_accuracy = 0.0
        efficiency_bonus = 0.0
        penalty = 0.0
        feedback_parts: list[str] = []
        is_final = False

        if self._detect_loop(state.patient.previous_actions):
            penalty += 0.30
            partial["loop_penalty"] = -0.30

        if action.action_type == ActionType.EVALUATE_CRITERION and action.evaluation:
            is_correct = truth[action.criterion_id] == action.evaluation.verdict
            base = self.step_reward_scale[state.task_id] if is_correct else 0.0
            raw_sum += base
            partial[f"criterion:{action.criterion_id}"] = round(base, 4)
            if len(action.evaluation.reasoning.strip()) > 20:
                raw_sum += 0.05
                partial["reasoning_bonus"] = 0.05
            if self._repeat_eval_without_clarification(state, action.criterion_id):
                penalty += 0.05
                partial["repeat_eval_penalty"] = -0.05
            eligibility_accuracy = 1.0 if is_correct else 0.0
            feedback_parts.append(f"Criterion {action.criterion_id} evaluated {'correctly' if is_correct else 'incorrectly'}.")

        elif action.action_type == ActionType.ASK_CLARIFICATION:
            certainty = self._certainty_for_target(state, action.clarification_target or "")
            if certainty == "confirmed":
                penalty += 0.10
                partial["clarification_penalty"] = -0.10
                feedback_parts.append("Clarification used on a confirmed value.")
            else:
                partial["clarification_neutral"] = 0.0
                feedback_parts.append("Clarification used on uncertain data.")

        if action.action_type in {ActionType.ENROLL, ActionType.EXCLUDE, ActionType.DEFER} or state.done:
            is_final = True
            if action.action_type in {ActionType.ENROLL, ActionType.EXCLUDE}:
                final_correct = ((action.action_type == ActionType.ENROLL) and hidden["final_eligible"]) or ((action.action_type == ActionType.EXCLUDE) and not hidden["final_eligible"])
                if final_correct:
                    raw_sum += 0.40
                    partial["final_decision_bonus"] = 0.40
                elif hidden["critical_exclusion_present"] and action.action_type == ActionType.ENROLL:
                    penalty += 0.30
                    partial["safety_penalty"] = -0.30
            if action.action_type == ActionType.DEFER:
                penalty += 0.20
                partial["defer_penalty"] = -0.20
            evaluated_fraction = len(state.evaluated_criteria) / max(len(truth), 1)
            if evaluated_fraction < 0.5:
                penalty += 0.15
                partial["premature_decision_penalty"] = -0.15
            if state.current_step <= int(state.max_steps * 0.6):
                efficiency_bonus += 0.10
            efficiency_bonus += min(state.patient.steps_remaining * 0.05, 0.15)
            raw_sum += efficiency_bonus
            partial["efficiency_bonus"] = round(efficiency_bonus, 4)
            grade = self.graders[state.task_id].grade(
                truth,
                state.evaluated_criteria,
                action.action_type.value,
                {
                    "final_eligible": hidden["final_eligible"],
                    "drug_interaction_miss": hidden.get("drug_interaction_miss", False),
                    "unnecessary_clarifications": hidden.get("unnecessary_clarifications", 0),
                    "amendment_detected": hidden.get("amendment_detected", False),
                    "ambiguity_handled": hidden.get("ambiguity_handled", False),
                    "ignored_amendment": hidden.get("ignored_amendment", False),
                    "steps_used": state.current_step,
                },
            )
            eligibility_accuracy = float(grade["score"])
            partial.update({f"grader:{key}": float(value) for key, value in grade["partial_credit"].items()})
            feedback_parts.append(str(grade["feedback"]))

        if state.current_step > state.max_steps:
            overflow = state.current_step - state.max_steps
            penalty += 0.05 * overflow
            partial["step_overflow_penalty"] = round(-0.05 * overflow, 4)

        normalized = self._clamp_open_unit_interval((raw_sum - penalty) / self.max_possible[state.task_id])
        eligibility_accuracy = self._clamp_open_unit_interval(eligibility_accuracy)
        return EnrollmentReward(
            total_reward=round(normalized, 4),
            eligibility_accuracy=round(eligibility_accuracy, 4),
            efficiency_bonus=round(min(max(efficiency_bonus, 0.0), 0.3), 4),
            penalty=round(min(max(penalty, 0.0), 1.0), 4),
            partial_credit={key: round(value, 4) for key, value in partial.items()},
            grader_feedback=" ".join(feedback_parts) if feedback_parts else "No-op action processed.",
            is_final=is_final,
        )

    def _detect_loop(self, history: list[str]) -> bool:
        return len(history) >= 3 and len(set(history[-3:])) == 1

    def _repeat_eval_without_clarification(self, state, criterion_id: str) -> bool:
        return bool(state.__dict__.get("hidden_case", {}).get("repeat_same_criterion", False))

    def _certainty_for_target(self, state, target: str) -> str:
        hidden = state.__dict__.get("hidden_case", {})
        if target == "INC-004" and "anc" in state.patient.lab_values:
            return state.patient.lab_values["anc"].certainty
        if target == "INC-003" and "css_score" in state.patient.lab_values:
            return state.patient.lab_values["css_score"].certainty
        if target in {"EXC-001", "EXC-002"}:
            return "estimated"
        if "clarification_certainty" in hidden:
            return hidden["clarification_certainty"]
        return "confirmed"

    def _clamp_open_unit_interval(self, value: float) -> float:
        return min(max(value, self.SCORE_EPSILON), 1.0 - self.SCORE_EPSILON)
