"""State transition logic and amendment handling."""

from server.models.action import ActionType, ScreeningAction
from server.models.state import TrialState


class StateMachine:
    """Apply action side effects and amendment events."""

    def apply_action(self, state: TrialState, action: ScreeningAction) -> dict[str, object]:
        self._advance_turn(state, action.action_type.value)
        info: dict[str, object] = {}
        if action.action_type == ActionType.EVALUATE_CRITERION and action.evaluation:
            state.evaluated_criteria[action.criterion_id or ""] = action.evaluation.verdict
            if state.task_id == "task3" and state.amendment_injected and action.criterion_id == "INC-003":
                state.patient.operational_state.amendment_review_required = False
            state.patient.info_message = f"Stored evaluation for {action.criterion_id}."
        elif action.action_type == ActionType.INSPECT_PATIENT:
            state.patient.info_message = "Patient state inspected."
        elif action.action_type == ActionType.INSPECT_PROTOCOL:
            state.patient.info_message = "Protocol state inspected."
        elif action.action_type == ActionType.ASK_CLARIFICATION:
            state.clarifications_used += 1
        elif action.action_type == ActionType.ENROLL and state.task_id == "task3":
            state.enrollment_decision = "enroll"
            state.workflow_phase = "followup_scheduling"
            state.patient.operational_state.workflow_phase = "followup_scheduling"
            state.patient.info_message = "Patient provisionally enrolled. Schedule a follow-up visit on day 7 to 10."
        elif action.action_type == ActionType.SCHEDULE_FOLLOWUP:
            state.scheduled_followup_day = action.followup_day
            state.workflow_phase = "safety_event"
            state.safety_event_active = True
            state.patient.operational_state.workflow_phase = "safety_event"
            state.patient.operational_state.scheduled_followup_day = action.followup_day
            state.patient.operational_state.safety_event_active = True
            state.patient.operational_state.safety_event_description = (
                state.__dict__["hidden_case"]["meta"]["safety_event_description"]
            )
            state.patient.info_message = (
                "Safety event: patient reports new seizure symptoms before the follow-up visit. "
                "Protocol requires investigator escalation before rescheduling."
            )
        elif action.action_type == ActionType.HANDLE_SAFETY_EVENT:
            state.safety_response = action.safety_response
            state.safety_event_active = False
            state.workflow_phase = "complete"
            state.patient.operational_state.workflow_phase = "complete"
            state.patient.operational_state.safety_event_active = False
            state.patient.operational_state.safety_response = action.safety_response
            if action.reschedule_day is not None:
                state.scheduled_followup_day = action.reschedule_day
                state.patient.operational_state.scheduled_followup_day = action.reschedule_day
            state.done = True
            state.termination_reason = "workflow_complete"
            state.patient.info_message = "Safety event response recorded."
            info["termination_reason"] = state.termination_reason
        elif action.action_type in {ActionType.EXCLUDE, ActionType.DEFER}:
            state.enrollment_decision = action.action_type.value
            state.workflow_phase = "excluded" if action.action_type == ActionType.EXCLUDE else "complete"
            if state.patient.operational_state is not None:
                state.patient.operational_state.workflow_phase = state.workflow_phase
            state.done = True
            state.termination_reason = f"final_action:{action.action_type.value}"
            info["termination_reason"] = state.termination_reason
        elif action.action_type == ActionType.ENROLL:
            state.done = True
            state.termination_reason = f"final_action:{action.action_type.value}"
            info["termination_reason"] = state.termination_reason
        if state.current_step >= state.max_steps and not state.done:
            state.done = True
            state.termination_reason = "max_steps_reached"
            info["termination_reason"] = state.termination_reason
        return info

    def apply_unsafe_enrollment(self, state: TrialState, action: ScreeningAction) -> dict[str, object]:
        """Terminate the episode immediately on an unsafe enroll action."""
        self._advance_turn(state, action.action_type.value)
        state.enrollment_decision = "enroll"
        state.workflow_phase = "complete"
        if state.patient.operational_state is not None:
            state.patient.operational_state.workflow_phase = "complete"
        state.done = True
        state.termination_reason = "unsafe_enrollment"
        state.patient.info_message = "Unsafe enrollment attempted."
        return {"termination_reason": state.termination_reason}

    def apply_invalid_action(self, state: TrialState, action: ScreeningAction, reason: str) -> dict[str, object]:
        """Record an invalid but schema-valid action as a penalized turn."""
        self._advance_turn(state, action.action_type.value)
        state.patient.info_message = reason
        info: dict[str, object] = {"invalid_action": True, "invalid_action_reason": reason}
        if state.current_step >= state.max_steps and not state.done:
            state.done = True
            state.termination_reason = "max_steps_reached"
            info["termination_reason"] = state.termination_reason
        return info

    def maybe_inject_amendment(self, state: TrialState) -> dict[str, object]:
        if (
            state.task_id != "task3"
            or state.amendment_injected
            or state.current_step < 3
            or state.workflow_phase != "screening"
        ):
            return {}
        protocol = state.patient.trial_protocol_summary
        protocol.amendment_active = True
        protocol.amendment_description = "Amendment A1 active: INC-003 CSS score range updated from 12-36 to 10-36."
        for criterion in protocol.inclusion_criteria:
            if criterion.criterion_id == "INC-003":
                criterion.description = "Rett Syndrome severity score between 10 and 36 on CSS scale"
        state.amendment_injected = True
        if state.patient.operational_state is not None:
            state.patient.operational_state.amendment_review_required = True
        state.patient.info_message = "Protocol amendment detected. Re-check criterion INC-003."
        return {"amendment_notice": protocol.amendment_description}

    def apply_clarification(self, state: TrialState, target: str) -> str | None:
        hidden = state.__dict__.get("hidden_case", {})
        detail = hidden.get("clarifications", {}).get(target)
        if not detail:
            return None
        if detail.get("visible_lab_key"):
            lab = state.patient.lab_values[detail["visible_lab_key"]]
            lab.value = float(detail["actual_value"])
            lab.certainty = str(detail["actual_certainty"])
        if target == "EXC-002":
            for med in state.patient.current_medications:
                if med.name.lower() in {"prednisone", "dexamethasone", "methylprednisolone"}:
                    med.is_contraindicated = bool(detail.get("is_contraindicated"))
        state.patient.info_message = detail["info_message"]
        return detail["info_message"]

    def _advance_turn(self, state: TrialState, action_name: str) -> None:
        state.current_step += 1
        state.patient.step_number = state.current_step
        state.patient.steps_remaining = max(state.max_steps - state.current_step, 0)
        state.patient.previous_actions.append(action_name)
