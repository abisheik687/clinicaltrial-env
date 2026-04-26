"""Environment API routes."""

import json
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from server.environment.env import ClinicalTrialEnv
from server.models.action import ScreeningAction
from server.models.score_serialization import serialize_nested_scores
from server.tasks.task_registry import TASKS


router = APIRouter()
env = ClinicalTrialEnv()
DEMO_HTML_PATH = Path(__file__).resolve().parents[2] / "demo_frontend.html"
EVAL_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "eval"


class ResetRequest(BaseModel):
    """Request body for /reset."""

    model_config = ConfigDict(extra="forbid")
    task_id: Literal["task1", "task2", "task3"] = "task1"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    """Request body for /step."""

    model_config = ConfigDict(extra="forbid")
    session_id: str
    action: ScreeningAction


class SessionRequest(BaseModel):
    """Request body for /validate_session."""

    model_config = ConfigDict(extra="forbid")
    session_id: str


@router.post("/reset")
def reset(request: ResetRequest | None = Body(default=None)) -> dict:
    if request is None:
        request = ResetRequest()
    observation, session_id, task_info = env.reset(request.task_id, request.seed)
    return {"observation": observation.model_dump(), "session_id": session_id, "task_info": task_info}


@router.post("/step")
def step(request: StepRequest) -> dict:
    observation, reward, done, info = env.step(request.session_id, request.action)
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": serialize_nested_scores(info),
    }


@router.get("/state")
def state(session_id: str) -> dict:
    return env.state(session_id).model_dump()


@router.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "version": "1.0.0", "tasks_available": list(TASKS.keys())}


@router.post("/validate_session")
def validate_session(request: SessionRequest) -> dict[str, object]:
    return env.validate_session(request.session_id)


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-file judge-facing demo frontend."""

    def load_json(filename: str, default: dict | list | None = None):
        path = EVAL_ARTIFACTS_DIR / filename
        if not path.exists():
            return {} if default is None else default
        return json.loads(path.read_text(encoding="utf-8"))

    baseline_eval = load_json("base_model_task3_eval.json")
    compact_eval = load_json("policy_gradient_task3_eval.json")
    lm_summary = load_json("lm_grpo_validation_summary_failed.json")
    behavior_diff = load_json("before_after_trajectory_diff.json")

    baseline_episodes = baseline_eval.get("episodes", [])
    compact_episodes = compact_eval.get("episodes", [])
    baseline_unsafe = next((episode for episode in baseline_episodes if episode.get("unsafe_action")), None)
    baseline_long = baseline_episodes[0] if baseline_episodes else {}
    compact_episode = compact_episodes[0] if compact_episodes else {}

    payload = {
        "banner": "LLM training attempt failed validation — compact RL policy demonstrates learnability",
        "baseline": {
            "aggregate": baseline_eval.get("aggregate", {}),
            "long_episode": baseline_long,
            "unsafe_episode": baseline_unsafe or {},
        },
        "compact_policy": {
            "aggregate": compact_eval.get("aggregate", {}),
            "episode": compact_episode,
        },
        "lm_grpo": {
            "summary": lm_summary,
            "behavior_diff": behavior_diff,
        },
    }

    html = DEMO_HTML_PATH.read_text(encoding="utf-8")
    return html.replace("__CLINICALTRIALENV_DATA__", json.dumps(payload))
