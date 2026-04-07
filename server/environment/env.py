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

    SCORE_EPSILON = 0.01

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
        self._validate_action(state, action)
        hidden = state.__dict__["hidden_case"]
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
        info.update(amendment_notice)
        reward = self.reward_calculator.compute(state, action, info)
        state.cumulative_reward = round(
            self._clamp_open_unit_interval(state.cumulative_reward + reward.total_reward),
            4,
        )
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

    def _validate_action(self, state: TrialState, action: ScreeningAction) -> None:
        truth = state.__dict__["hidden_case"]["criterion_truth"]
        if action.action_type == ActionType.EVALUATE_CRITERION and action.criterion_id not in truth:
            raise HTTPException(status_code=400, detail="Unknown criterion_id")
        if action.action_type == ActionType.ASK_CLARIFICATION:
            if state.clarifications_used >= state.clarification_budget:
                raise HTTPException(status_code=400, detail="Clarification budget exhausted")
            if action.clarification_target not in truth:
                raise HTTPException(status_code=400, detail="Unknown clarification_target")
        if action.action_type in {ActionType.ENROLL, ActionType.EXCLUDE} and len(state.evaluated_criteria) == 0:
            raise HTTPException(status_code=400, detail="Must evaluate criteria before final decision")

    def _clamp_open_unit_interval(self, value: float) -> float:
        return min(max(value, self.SCORE_EPSILON), 1.0 - self.SCORE_EPSILON)
