"""Environment API routes."""

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
    """Serve the single-file interactive demo frontend."""
    return DEMO_HTML_PATH.read_text(encoding="utf-8")
