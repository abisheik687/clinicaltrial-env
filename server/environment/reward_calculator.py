"""Verifier-centric rewards and diagnostic metrics."""

from server.models.action import ActionType, ScreeningAction
from server.models.reward import EnrollmentReward


class RewardCalculator:
    """Compute terminal rewards plus diagnostic metrics."""

    INVALID_ACTION_PENALTY = -0.05

    def __init__(self) -> None:
        pass

    def compute(self, state, action: ScreeningAction, info: dict[str, object]) -> EnrollmentReward:
        hidden = state.__dict__.get("hidden_case", {})
        current_truth = self._current_truth(state, hidden)
        if state.task_id == "task3":
            return self._compute_task3(state, action, current_truth)
        is_final = action.action_type in {ActionType.ENROLL, ActionType.EXCLUDE, ActionType.DEFER} or state.done
        diagnostics = self._diagnostic_metrics(state, current_truth, unsafe_action=False)
        reward = 0.0
        feedback = "Action recorded."
        terminal_success = False
        unsafe_action = False

        if action.action_type == ActionType.EVALUATE_CRITERION and action.evaluation:
            is_correct = current_truth[action.criterion_id] == action.evaluation.verdict
            feedback = f"Criterion {action.criterion_id} evaluated {'correctly' if is_correct else 'incorrectly'} under the current protocol."
        elif action.action_type == ActionType.INSPECT_PATIENT:
            feedback = "Patient state inspected."
        elif action.action_type == ActionType.INSPECT_PROTOCOL:
            feedback = "Protocol state inspected."
        elif action.action_type == ActionType.ASK_CLARIFICATION:
            feedback = f"Clarification requested for {action.clarification_target or 'unknown target'}."

        if is_final:
            unsafe_action = self._is_unsafe_enrollment(action, current_truth)
            terminal_success = self._is_terminal_success(action, current_truth)
            reward = -1.0 if unsafe_action else (1.0 if terminal_success else 0.0)
            diagnostics = self._diagnostic_metrics(state, current_truth, unsafe_action=unsafe_action)
            feedback = self._terminal_feedback(action, terminal_success, unsafe_action, current_truth)

        return EnrollmentReward(
            total_reward=round(reward, 4),
            terminal_success=terminal_success,
            unsafe_action=unsafe_action,
            invalid_action_penalty=0.0,
            diagnostic_metrics=diagnostics,
            verifier_feedback=feedback,
            is_final=is_final,
        )

    def _compute_task3(self, state, action: ScreeningAction, truth: dict[str, str]) -> EnrollmentReward:
        unsafe_action = state.termination_reason == "unsafe_enrollment" or self._is_unsafe_enrollment(action, truth)
        is_final = bool(state.done)
        diagnostics = self._diagnostic_metrics(state, truth, unsafe_action=unsafe_action)
        feedback = self._task3_feedback(state, action, diagnostics, unsafe_action)
        reward = 0.0
        terminal_success = False

        if unsafe_action and is_final:
            reward = -1.0
            feedback = self._terminal_feedback(action, False, True, truth)
        elif is_final:
            component_values = [
                value
                for value in (
                    diagnostics["eligibility_component_score"],
                    diagnostics["amendment_component_score"],
                    diagnostics["scheduling_component_score"],
                    diagnostics["safety_component_score"],
                )
                if value >= 0.0
            ]
            reward = sum(component_values) / len(component_values) if component_values else 0.0
            terminal_success = bool(component_values) and all(value == 1.0 for value in component_values)
            feedback = self._task3_terminal_feedback(action, diagnostics, terminal_success)

        return EnrollmentReward(
            total_reward=round(reward, 4),
            terminal_success=terminal_success,
            unsafe_action=unsafe_action,
            invalid_action_penalty=0.0,
            diagnostic_metrics=diagnostics,
            verifier_feedback=feedback,
            is_final=is_final,
        )

    def compute_invalid(self, state, reason: str) -> EnrollmentReward:
        hidden = state.__dict__.get("hidden_case", {})
        current_truth = self._current_truth(state, hidden)
        return EnrollmentReward(
            total_reward=round(self.INVALID_ACTION_PENALTY, 4),
            terminal_success=False,
            unsafe_action=False,
            invalid_action_penalty=round(self.INVALID_ACTION_PENALTY, 4),
            diagnostic_metrics=self._diagnostic_metrics(state, current_truth, unsafe_action=False),
            verifier_feedback=reason,
            is_final=state.done,
        )

    def _current_truth(self, state, hidden: dict[str, object]) -> dict[str, str]:
        truth = dict(hidden["criterion_truth"])
        if state.task_id == "task3" and state.amendment_injected:
            truth["INC-003"] = hidden["meta"]["post_amendment_truth"]
        return truth

    def _is_unsafe_enrollment(self, action: ScreeningAction, truth: dict[str, str]) -> bool:
        if action.action_type != ActionType.ENROLL:
            return False
        hard_exclusion_present = any(
            verdict == "met" and criterion_id.startswith("EXC-")
            for criterion_id, verdict in truth.items()
        )
        inclusion_failed = any(
            verdict != "met" and criterion_id.startswith("INC-")
            for criterion_id, verdict in truth.items()
        )
        return hard_exclusion_present or inclusion_failed

    def _is_terminal_success(self, action: ScreeningAction, truth: dict[str, str]) -> bool:
        all_inclusions_met = all(
            verdict == "met"
            for criterion_id, verdict in truth.items()
            if criterion_id.startswith("INC-")
        )
        no_exclusions_triggered = all(
            verdict != "met"
            for criterion_id, verdict in truth.items()
            if criterion_id.startswith("EXC-")
        )
        final_eligible = all_inclusions_met and no_exclusions_triggered
        if action.action_type == ActionType.ENROLL:
            return final_eligible
        if action.action_type == ActionType.EXCLUDE:
            return not final_eligible
        return False

    def _diagnostic_metrics(self, state, truth: dict[str, str], unsafe_action: bool) -> dict[str, float]:
        evaluated = state.evaluated_criteria
        if evaluated:
            correct = sum(1 for criterion_id, verdict in evaluated.items() if truth.get(criterion_id) == verdict)
            criterion_accuracy = correct / len(evaluated)
        else:
            criterion_accuracy = 0.0

        hidden = state.__dict__.get("hidden_case", {})
        unnecessary_clarifications = int(hidden.get("unnecessary_clarifications", 0))
        if state.clarification_budget > 0:
            clarification_efficiency = max(
                0.0,
                1.0 - (unnecessary_clarifications / state.clarification_budget),
            )
        else:
            clarification_efficiency = 1.0

        amendment_recovery = 1.0
        if state.task_id == "task3" and state.amendment_injected:
            pre_truth = hidden["meta"]["pre_amendment_truth"]
            post_truth = hidden["meta"]["post_amendment_truth"]
            if pre_truth != post_truth:
                amendment_recovery = 1.0 if evaluated.get("INC-003") == post_truth else 0.0

        diagnostics = {
            "criterion_evaluation_accuracy": round(criterion_accuracy, 4),
            "clarification_efficiency": round(clarification_efficiency, 4),
            "unsafe_action_rate": 1.0 if unsafe_action else 0.0,
            "amendment_recovery_rate": round(amendment_recovery, 4),
        }
        if state.task_id == "task3":
            diagnostics.update(self._task3_component_metrics(state, truth, amendment_recovery, unsafe_action))
        return diagnostics

    def _task3_component_metrics(
        self,
        state,
        truth: dict[str, str],
        amendment_recovery: float,
        unsafe_action: bool,
    ) -> dict[str, float]:
        enrollment_decision = state.enrollment_decision
        all_inclusions_met = all(
            verdict == "met"
            for criterion_id, verdict in truth.items()
            if criterion_id.startswith("INC-")
        )
        no_exclusions_triggered = all(
            verdict != "met"
            for criterion_id, verdict in truth.items()
            if criterion_id.startswith("EXC-")
        )
        final_eligible = all_inclusions_met and no_exclusions_triggered

        eligibility_component = -1.0
        if enrollment_decision == "enroll":
            eligibility_component = 1.0 if final_eligible and not unsafe_action else 0.0
        elif enrollment_decision in {"exclude", "defer"}:
            eligibility_component = 1.0 if not final_eligible else 0.0

        amendment_component = -1.0
        if state.amendment_injected:
            amendment_component = amendment_recovery

        scheduling_component = -1.0
        safety_component = -1.0
        ops = state.patient.operational_state
        if enrollment_decision == "enroll":
            window_start = ops.followup_window_start if ops else None
            window_end = ops.followup_window_end if ops else None
            if window_start is not None and window_end is not None and state.scheduled_followup_day is not None:
                scheduling_component = 1.0 if window_start <= state.scheduled_followup_day <= window_end else 0.0
            else:
                scheduling_component = 0.0 if state.done else -1.0
            required_action = hidden_required_action = state.__dict__.get("hidden_case", {}).get("meta", {}).get("required_safety_action", "escalate")
            if state.safety_response is not None:
                safety_component = 1.0 if state.safety_response == hidden_required_action else 0.0
            else:
                safety_component = 0.0 if state.done else -1.0

        return {
            "eligibility_component_score": round(eligibility_component, 4),
            "amendment_component_score": round(amendment_component, 4),
            "scheduling_component_score": round(scheduling_component, 4),
            "safety_component_score": round(safety_component, 4),
        }

    def _certainty_for_target(self, state, target: str) -> str:
        hidden = state.__dict__.get("hidden_case", {})
        if target == "INC-004" and "anc" in state.patient.lab_values:
            return state.patient.lab_values["anc"].certainty
        if target == "INC-003" and "css_score" in state.patient.lab_values:
            return state.patient.lab_values["css_score"].certainty
        if target in {"EXC-001", "EXC-002"}:
            return "estimated"
        return hidden.get("clarification_certainty", "confirmed")

    def _terminal_feedback(
        self,
        action: ScreeningAction,
        terminal_success: bool,
        unsafe_action: bool,
        truth: dict[str, str],
    ) -> str:
        if unsafe_action:
            active_exclusions = sorted(
                criterion_id for criterion_id, verdict in truth.items() if criterion_id.startswith("EXC-") and verdict == "met"
            )
            failed_inclusions = sorted(
                criterion_id for criterion_id, verdict in truth.items() if criterion_id.startswith("INC-") and verdict != "met"
            )
            return (
                "Unsafe enrollment attempted. "
                f"Active exclusions: {active_exclusions or ['none']}; "
                f"Failed inclusions: {failed_inclusions or ['none']}."
            )
        if terminal_success:
            return f"Final decision `{action.action_type.value}` matches the latest protocol state."
        if action.action_type == ActionType.DEFER:
            return "Episode deferred without reaching a verifiable safe final decision."
        return f"Final decision `{action.action_type.value}` does not match the latest protocol state."

    def _task3_feedback(self, state, action: ScreeningAction, diagnostics: dict[str, float], unsafe_action: bool) -> str:
        if unsafe_action:
            return self._terminal_feedback(action, False, True, self._current_truth(state, state.__dict__.get("hidden_case", {})))
        if action.action_type == ActionType.SCHEDULE_FOLLOWUP:
            return f"Follow-up visit proposed for day {action.followup_day}."
        if action.action_type == ActionType.HANDLE_SAFETY_EVENT:
            return f"Safety response `{action.safety_response}` recorded."
        if action.action_type == ActionType.ENROLL:
            return "Eligibility decision recorded. Continue with follow-up scheduling."
        if action.action_type == ActionType.EXCLUDE:
            return "Screening concluded with exclusion."
        if action.action_type == ActionType.EVALUATE_CRITERION and action.evaluation:
            is_correct = self._current_truth(state, state.__dict__.get("hidden_case", {})).get(action.criterion_id) == action.evaluation.verdict
            return f"Criterion {action.criterion_id} evaluated {'correctly' if is_correct else 'incorrectly'} under the current protocol."
        return "Action recorded."

    def _task3_terminal_feedback(self, action: ScreeningAction, diagnostics: dict[str, float], terminal_success: bool) -> str:
        if terminal_success:
            return (
                "Workflow completed successfully: eligibility, amendment handling, follow-up scheduling, "
                "and safety response all matched the verifier."
            )
        failed_components = [
            label
            for label, key in (
                ("eligibility", "eligibility_component_score"),
                ("amendment", "amendment_component_score"),
                ("scheduling", "scheduling_component_score"),
                ("safety", "safety_component_score"),
            )
            if diagnostics.get(key, -1.0) == 0.0
        ]
        return (
            f"Workflow ended with verifier gaps in: {', '.join(failed_components) or 'none'}."
            if failed_components
            else f"Workflow ended after `{action.action_type.value}` without full verifier success."
        )
