"""Environment API routes."""

from typing import Literal, Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel, ConfigDict
from fastapi.responses import HTMLResponse

from server.environment.env import ClinicalTrialEnv
from server.models.action import ScreeningAction
from server.tasks.task_registry import TASKS


router = APIRouter()
env = ClinicalTrialEnv()


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
    return {"observation": observation.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}


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
    """Human-friendly landing page for Hugging Face Spaces reviewers."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ClinicalTrialEnv</title>
  <style>
    :root {
      --bg: #f4efe6;
      --paper: #fffaf2;
      --ink: #183153;
      --muted: #5d7288;
      --accent: #1d7f6f;
      --accent-2: #d96c3f;
      --line: #d9d1c5;
      --shadow: 0 18px 40px rgba(24, 49, 83, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(217,108,63,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(29,127,111,0.18), transparent 24%),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 24px;
      align-items: stretch;
    }
    .card {
      background: rgba(255,250,242,0.92);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 28px;
      backdrop-filter: blur(8px);
    }
    .eyebrow {
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent);
      margin-bottom: 12px;
      font-weight: 700;
    }
    h1 {
      font-size: clamp(2.4rem, 5vw, 4.2rem);
      line-height: 0.96;
      margin: 0 0 16px;
      max-width: 10ch;
    }
    p {
      color: var(--muted);
      font-size: 1.02rem;
      line-height: 1.7;
      margin: 0 0 16px;
    }
    .pillrow, .grid {
      display: grid;
      gap: 14px;
    }
    .pillrow {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 22px;
    }
    .pill {
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 0.95rem;
    }
    .stat {
      font-size: 2rem;
      font-weight: 700;
      color: var(--accent-2);
      margin-bottom: 4px;
    }
    .grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 24px;
    }
    .task {
      border-top: 4px solid var(--accent);
    }
    .task h3, .ops h3 {
      margin: 0 0 8px;
      font-size: 1.2rem;
    }
    .badge {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #eaf5f2;
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 700;
      margin-bottom: 14px;
    }
    code, pre {
      font-family: Consolas, "Courier New", monospace;
    }
    pre {
      margin: 0;
      padding: 16px;
      background: #162433;
      color: #eff7ff;
      border-radius: 16px;
      overflow-x: auto;
      font-size: 0.9rem;
    }
    .ops {
      margin-top: 24px;
    }
    .linklist {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 20px;
    }
    .linklist a {
      color: var(--ink);
      text-decoration: none;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 0.92rem;
    }
    .validator-note {
      margin-top: 18px;
      padding: 14px 16px;
      border-radius: 16px;
      background: #eaf5f2;
      color: var(--accent);
      border: 1px solid #c6e3dc;
      font-size: 0.95rem;
      font-weight: 700;
    }
    @media (max-width: 900px) {
      .hero, .grid, .pillrow {
        grid-template-columns: 1fr;
      }
      h1 { max-width: none; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="card">
        <div class="eyebrow">OpenEnv Hackathon Submission</div>
        <h1>Clinical Trial Screening for Real Agents</h1>
        <p>ClinicalTrialEnv simulates the work of a clinical trial coordinator: screening synthetic patients, handling uncertain evidence, reacting to protocol amendments, and making safety-sensitive enrollment decisions.</p>
        <p>The environment is deterministic by seed, exposes typed OpenEnv-compatible APIs, and includes three grader-backed tasks spanning easy, medium, and hard difficulty.</p>
        <div class="linklist">
          <a href="https://github.com/abisheik687/clinicaltrial-env" target="_blank" rel="noreferrer">GitHub Repo</a>
          <a href="https://huggingface.co/spaces/abisheiks/clinicaltrial-env" target="_blank" rel="noreferrer">HF Space</a>
        </div>
        <div class="validator-note">Validator-ready: /health and /reset are live, Docker runs on port 7860, and openenv validate passes.</div>
        <div class="pillrow">
          <div class="pill"><strong>3 Tasks</strong><br>Easy to hard progression</div>
          <div class="pill"><strong>Typed Models</strong><br>Pydantic v2 observation, action, reward</div>
          <div class="pill"><strong>HF Ready</strong><br>Docker Space on port 7860</div>
        </div>
      </div>
      <div class="card">
        <div class="eyebrow">Live API</div>
        <div class="stat">/reset</div>
        <p>Start a new episode with a default task or a task-specific seed.</p>
        <div class="stat">/step</div>
        <p>Submit structured actions and receive shaped reward, updated observation, and termination status.</p>
        <div class="stat">/state</div>
        <p>Inspect the full state for debugging and amendment verification.</p>
      </div>
    </section>
    <section class="grid">
      <article class="card task">
        <div class="badge">Task 1 · Easy</div>
        <h3>Hypertension Screening</h3>
        <p>Clear eligibility checks with no clarification budget. This task measures disciplined criterion-by-criterion review.</p>
      </article>
      <article class="card task">
        <div class="badge">Task 2 · Medium</div>
        <h3>Oncology CAR-T Review</h3>
        <p>Compound marrow criteria and corticosteroid reasoning introduce realistic interaction and dosing logic.</p>
      </article>
      <article class="card task">
        <div class="badge">Task 3 · Hard</div>
        <h3>Gene Therapy Amendment</h3>
        <p>Ambiguous signals, clarification budgeting, and a protocol amendment at step 6 reward adaptive clinical reasoning.</p>
      </article>
    </section>
    <section class="card ops">
      <div class="eyebrow">Quick Check</div>
      <h3>Validator-Compatible Reset</h3>
      <pre>curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'</pre>
    </section>
  </div>
</body>
</html>
"""
