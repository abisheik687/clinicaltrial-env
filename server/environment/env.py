"""Core OpenEnv-compatible environment."""

from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from server.config import get_settings
from server.data.patient_generator import PatientGenerator
from server.data.protocol_loader import ProtocolLoader
from server.environment.episode_manager import EpisodeManager
from server.environment.reward_calculator import RewardCalculator
from server.environment.state_machine import StateMachine
from server.models.action import ActionType, ScreeningAction
from server.models.observation import PatientObservation
from server.models.reward import EnrollmentReward
from server.models.state import TrialState
from server.tasks.task_registry import get_task_definition


class ClinicalTrialEnv:
    """In-memory environment manager with OpenEnv-style methods."""

    def __init__(self) -> None:
        settings = get_settings()
        protocol_dir = Path(__file__).resolve().parents[2] / "protocols"
        self.loader = ProtocolLoader(protocol_dir)
        self.generator = PatientGenerator(self.loader, seed=settings.DEFAULT_SEED)
        self.state_machine = StateMachine()
        self.reward_calculator = RewardCalculator()
        self.episode_manager = EpisodeManager(timeout_minutes=settings.SESSION_TIMEOUT_MINUTES)
        self.sessions: dict[str, TrialState] = {}
        self.episode_counter = 0

    def reset(self, task_id: str, seed: int | None = None) -> tuple[PatientObservation, str, dict[str, object]]:
        self.cleanup_sessions()
        task = get_task_definition(task_id)
        actual_seed = seed if seed is not None else get_settings().DEFAULT_SEED
        generated = self.generator.generate_case(task_id, actual_seed)
        session_id = str(uuid4())
        self.episode_counter += 1
        state = TrialState(
            session_id=session_id,
            task_id=task_id,
            episode_number=self.episode_counter,
            current_step=0,
            max_steps=task.max_steps,
            patient=generated.observation_model.model_copy(deep=True),
            evaluated_criteria={},
            clarifications_used=0,
            clarification_budget=task.clarification_budget,
            amendment_injected=False,
            workflow_phase="screening",
            enrollment_decision=None,
            scheduled_followup_day=None,
            safety_event_active=False,
            safety_response=None,
            cumulative_reward=0.0,
            done=False,
            termination_reason=None,
        )
        state.__dict__["hidden_case"] = generated.to_state_payload()
        state.__dict__["last_seed"] = actual_seed
        self.sessions[session_id] = state
        self.episode_manager.touch(session_id)
        return state.patient.model_copy(deep=True), session_id, {
            "name": task.name,
            "max_steps": task.max_steps,
            "clarification_budget": task.clarification_budget,
        }

    def step(self, session_id: str, action: ScreeningAction) -> tuple[PatientObservation, EnrollmentReward, bool, dict[str, object]]:
        state = self._get_session(session_id)
        if state.done:
            raise HTTPException(status_code=400, detail="Episode already done")
        hidden = state.__dict__["hidden_case"]
        if state.task_id == "task3" and state.workflow_phase == "screening":
            pre_action_notice = self.state_machine.maybe_inject_amendment(state)
        else:
            pre_action_notice = {}
        invalid_reason = self._invalid_action_reason(state, action)
        if invalid_reason is not None:
            info = self.state_machine.apply_invalid_action(state, action, invalid_reason)
            amendment_notice = self.state_machine.maybe_inject_amendment(state)
            info.update(pre_action_notice)
            info.update(amendment_notice)
            reward = self.reward_calculator.compute_invalid(state, invalid_reason)
            state.cumulative_reward = round(state.cumulative_reward + reward.total_reward, 4)
            self.episode_manager.touch(session_id)
            return state.patient.model_copy(deep=True), reward, state.done, info

        self._validate_action_ids(state, action)
        current_truth = self.reward_calculator._current_truth(state, hidden)
        if state.task_id == "task3" and action.action_type == ActionType.ENROLL and self.reward_calculator._is_unsafe_enrollment(action, current_truth):
            info = self.state_machine.apply_unsafe_enrollment(state, action)
            info.update(pre_action_notice)
            reward = self.reward_calculator.compute(state, action, info)
            state.cumulative_reward = round(state.cumulative_reward + reward.total_reward, 4)
            self.episode_manager.touch(session_id)
            return state.patient.model_copy(deep=True), reward, state.done, info
        if action.action_type == ActionType.EVALUATE_CRITERION and action.criterion_id:
            counts = hidden.setdefault("evaluation_counts", {})
            hidden["repeat_same_criterion"] = counts.get(action.criterion_id, 0) > 0
            counts[action.criterion_id] = counts.get(action.criterion_id, 0) + 1
            if action.criterion_id == "EXC-004" and hidden.get("meta", {}).get("drug_interaction_case"):
                hidden["drug_interaction_miss"] = action.evaluation is not None and action.evaluation.verdict != hidden["criterion_truth"]["EXC-004"]
        if action.action_type == ActionType.ASK_CLARIFICATION:
            hidden["clarification_certainty"] = self.reward_calculator._certainty_for_target(state, action.clarification_target or "")
        info = self.state_machine.apply_action(state, action)
        if action.action_type == ActionType.ASK_CLARIFICATION:
            message = self.state_machine.apply_clarification(state, action.clarification_target or "")
            state.patient.info_message = message or "No clarification available."
            certainty = hidden.get("clarification_certainty", "confirmed")
            if certainty == "confirmed":
                hidden["unnecessary_clarifications"] = hidden.get("unnecessary_clarifications", 0) + 1
            if action.clarification_target in {"INC-003", "EXC-001", "EXC-002"}:
                hidden["ambiguity_handled"] = True
        if state.task_id == "task3" and action.action_type == ActionType.ENROLL:
            meta = hidden["meta"]
            if state.amendment_injected and not hidden.get("amendment_detected", False) and meta["pre_amendment_truth"] != meta["post_amendment_truth"]:
                hidden["ignored_amendment"] = True
        if state.amendment_injected and action.action_type == ActionType.EVALUATE_CRITERION and action.criterion_id == "INC-003":
            hidden["criterion_truth"]["INC-003"] = hidden["meta"]["post_amendment_truth"]
            hidden["amendment_detected"] = True
        amendment_notice = self.state_machine.maybe_inject_amendment(state)
        info.update(pre_action_notice)
        info.update(amendment_notice)
        reward = self.reward_calculator.compute(state, action, info)
        state.cumulative_reward = round(state.cumulative_reward + reward.total_reward, 4)
        self.episode_manager.touch(session_id)
        return state.patient.model_copy(deep=True), reward, state.done, info

    def state(self, session_id: str) -> TrialState:
        state = self._get_session(session_id)
        self.episode_manager.touch(session_id)
        return state.model_copy(deep=True)

    def validate_session(self, session_id: str) -> dict[str, object]:
        if session_id not in self.sessions or self.episode_manager.is_expired(session_id):
            return {"valid": False, "done": False, "steps_used": 0}
        state = self.sessions[session_id]
        return {"valid": True, "done": state.done, "steps_used": state.current_step}

    def cleanup_sessions(self) -> None:
        self.episode_manager.cleanup(self.sessions)

    def _get_session(self, session_id: str) -> TrialState:
        self.cleanup_sessions()
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="session_id not found")
        return self.sessions[session_id]

    def _validate_action_ids(self, state: TrialState, action: ScreeningAction) -> None:
        truth = state.__dict__["hidden_case"]["criterion_truth"]
        if action.action_type == ActionType.EVALUATE_CRITERION and action.criterion_id not in truth:
            raise HTTPException(status_code=400, detail="Unknown criterion_id")
        if action.action_type == ActionType.ASK_CLARIFICATION:
            if action.clarification_target not in truth:
                raise HTTPException(status_code=400, detail="Unknown clarification_target")

    def _invalid_action_reason(self, state: TrialState, action: ScreeningAction) -> str | None:
        if action.action_type == ActionType.ASK_CLARIFICATION and state.clarifications_used >= state.clarification_budget:
            return "Clarification budget exhausted."
        if action.action_type in {ActionType.ENROLL, ActionType.EXCLUDE, ActionType.DEFER} and not state.evaluated_criteria:
            return "At least one criterion must be evaluated before a final decision."
        if state.task_id == "task3":
            ops = state.patient.operational_state
            if action.action_type in {ActionType.ENROLL, ActionType.EXCLUDE, ActionType.DEFER} and state.workflow_phase != "screening":
                return "Screening decisions are only available during the screening phase."
            if action.action_type in {ActionType.ENROLL, ActionType.EXCLUDE, ActionType.DEFER} and not state.amendment_injected:
                return "Protocol review incomplete before final decision. Wait for the amendment notice."
            if action.action_type == ActionType.SCHEDULE_FOLLOWUP:
                if state.workflow_phase != "followup_scheduling":
                    return "Follow-up scheduling is only available after safe enrollment."
                if ops is None or action.followup_day is None:
                    return "A valid follow-up day is required."
                if not (ops.followup_window_start <= action.followup_day <= ops.followup_window_end):
                    return f"Follow-up day must stay within the allowed window of day {ops.followup_window_start} to {ops.followup_window_end}."
            if action.action_type == ActionType.HANDLE_SAFETY_EVENT:
                if state.workflow_phase != "safety_event" or not state.safety_event_active:
                    return "Safety handling is only available when a safety event is active."
                if action.safety_response == "reschedule":
                    if ops is None or action.reschedule_day is None:
                        return "A new follow-up day is required when rescheduling."
                    if not (ops.followup_window_start <= action.reschedule_day <= ops.followup_window_end):
                        return f"Rescheduled follow-up day must stay within the allowed window of day {ops.followup_window_start} to {ops.followup_window_end}."
        return None
