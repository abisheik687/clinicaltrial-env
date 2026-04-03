"""Inference script smoke tests."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import inference
from server.api.routes import env
from server.main import app


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **_: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class FakeClient:
    def __init__(self, content: str = "{not-json") -> None:
        self.chat = _FakeChat(content)


def setup_function() -> None:
    env.sessions.clear()
    env.episode_manager.last_access.clear()


def test_log_format_strict_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    inference.log_start("Task", "clinicaltrial-env", "model-x")
    inference.log_step(2, '{"action_type":"evaluate_criterion"}', 0.125, False, None)
    inference.log_end(True, 3, [0.1, 0.2, 0.3])
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines[0] == "[START] task=Task env=clinicaltrial-env model=model-x"
    assert lines[1] == '[STEP]  step=2 action={"action_type":"evaluate_criterion"} reward=0.12 done=false error=null'
    assert lines[2] == "[END]   success=true steps=3 rewards=0.10,0.20,0.30"


def test_malformed_model_output_falls_back_to_safe_action() -> None:
    observation = {
        "trial_protocol_summary": {
            "trial_id": "TRIAL-A-HTN-001",
            "amendment_active": False,
            "inclusion_criteria": [
                {"criterion_id": "INC-001", "clarification_available": False},
            ],
            "exclusion_criteria": [],
        },
        "demographics": {"age": 45, "weight_kg": 80.0},
        "diagnosis": {"icd10_code": "I10", "diagnosis_date": "2025-01-01", "primary_condition": "Essential hypertension"},
        "lab_values": {"systolic_bp": {"value": 150.0}},
        "current_medications": [],
    }
    action = inference.get_agent_action(FakeClient(), observation, 0.0, [], 1, "task1", [])
    assert action["action_type"] == "evaluate_criterion"
    assert action["criterion_id"] == "INC-001"


@pytest.mark.asyncio
async def test_fallback_policy_completes_tasks_within_budget() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        fake_client = FakeClient()
        for task_config in inference.TASKS:
            result = await inference.run_task(fake_client, client, task_config)
            assert result["steps"] <= task_config["max_steps"]
            assert result["task_id"] == task_config["task_id"]
