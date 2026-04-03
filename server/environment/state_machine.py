"""State transition logic and amendment handling."""

from server.models.action import ActionType, ScreeningAction
from server.models.state import TrialState


class StateMachine:
    """Apply action side effects and amendment events."""

    def apply_action(self, state: TrialState, action: ScreeningAction) -> dict[str, object]:
        state.current_step += 1
        state.patient.step_number = state.current_step
        state.patient.steps_remaining = max(state.max_steps - state.current_step, 0)
        state.patient.previous_actions.append(action.action_type.value)
        info: dict[str, object] = {}
        if action.action_type == ActionType.EVALUATE_CRITERION and action.evaluation:
            state.evaluated_criteria[action.criterion_id or ""] = action.evaluation.verdict
            state.patient.info_message = f"Stored evaluation for {action.criterion_id}."
        elif action.action_type == ActionType.ASK_CLARIFICATION:
            state.clarifications_used += 1
        elif action.action_type in {ActionType.ENROLL, ActionType.EXCLUDE, ActionType.DEFER}:
            state.done = True
            state.termination_reason = f"final_action:{action.action_type.value}"
            info["termination_reason"] = state.termination_reason
        if state.current_step >= state.max_steps and not state.done:
            state.done = True
            state.termination_reason = "max_steps_reached"
            info["termination_reason"] = state.termination_reason
        return info

    def maybe_inject_amendment(self, state: TrialState) -> dict[str, object]:
        if state.task_id != "task3" or state.amendment_injected or state.current_step < 6:
            return {}
        protocol = state.patient.trial_protocol_summary
        protocol.amendment_active = True
        protocol.amendment_description = "Amendment A1 active: INC-003 CSS score range updated from 12-36 to 10-36."
        for criterion in protocol.inclusion_criteria:
            if criterion.criterion_id == "INC-003":
                criterion.description = "Rett Syndrome severity score between 10 and 36 on CSS scale"
        state.amendment_injected = True
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

