"""FastAPI integration tests."""

from fastapi.testclient import TestClient

from server.api.routes import env
from server.main import app


client = TestClient(app)


def setup_function() -> None:
    env.sessions.clear()
    env.episode_manager.last_access.clear()


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_landing_page() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ClinicalTrialEnv" in response.text
    assert "Clinical Trial Operations Arena" in response.text


def test_reset_endpoint_task1() -> None:
    response = client.post("/reset", json={"task_id": "task1", "seed": 42})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_info"]["max_steps"] == 8
    assert "observation" in payload


def test_reset_endpoint_no_body_defaults_to_task1() -> None:
    response = client.post("/reset")
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_info"]["max_steps"] == 8


def test_reset_endpoint_empty_body_defaults_to_task1() -> None:
    response = client.post("/reset", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_info"]["max_steps"] == 8


def test_step_endpoint_valid_action() -> None:
    reset = client.post("/reset", json={"task_id": "task1", "seed": 42}).json()
    session_id = reset["session_id"]
    truth = env.sessions[session_id].__dict__["hidden_case"]["criterion_truth"]["INC-001"]
    response = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {
                "action_type": "evaluate_criterion",
                "criterion_id": "INC-001",
                "evaluation": {
                    "criterion_id": "INC-001",
                    "verdict": truth,
                    "reasoning": "API integration test evaluation with enough detail.",
                },
                "confidence_score": 0.9,
            },
        },
    )
    assert response.status_code == 200
    reward = response.json()["reward"]
    assert reward["total_reward"] == 0.0
    assert reward["terminal_success"] is False
    assert reward["unsafe_action"] is False
    assert reward["diagnostic_metrics"]["criterion_evaluation_accuracy"] == 1.0


def test_step_endpoint_invalid_session() -> None:
    response = client.post(
        "/step",
        json={
            "session_id": "missing",
            "action": {"action_type": "defer", "confidence_score": 0.1},
        },
    )
    assert response.status_code == 404


def test_state_endpoint() -> None:
    reset = client.post("/reset", json={"task_id": "task2", "seed": 43}).json()
    response = client.get("/state", params={"session_id": reset["session_id"]})
    assert response.status_code == 200
    assert response.json()["task_id"] == "task2"
    assert response.json()["cumulative_reward"] == 0.0


def test_complete_task1_episode() -> None:
    reset = client.post("/reset", json={"task_id": "task1", "seed": 42}).json()
    session_id = reset["session_id"]
    truth = env.sessions[session_id].__dict__["hidden_case"]["criterion_truth"]
    for criterion_id in truth:
        response = client.post(
            "/step",
            json={
                "session_id": session_id,
                "action": {
                    "action_type": "evaluate_criterion",
                    "criterion_id": criterion_id,
                    "evaluation": {
                        "criterion_id": criterion_id,
                        "verdict": truth[criterion_id],
                        "reasoning": f"Full episode walkthrough evaluation for {criterion_id}.",
                    },
                    "confidence_score": 0.95,
                },
            },
        )
        assert response.status_code == 200
    final_action = "enroll" if env.sessions[session_id].__dict__["hidden_case"]["final_eligible"] else "exclude"
    final = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {
                "action_type": final_action,
                "final_decision_reason": "Completed deterministic task1 walkthrough.",
                "confidence_score": 0.99,
            },
        },
    )
    assert final.status_code == 200
    assert final.json()["done"] is True
    reward = final.json()["reward"]
    assert reward["total_reward"] == 1.0
    assert reward["terminal_success"] is True
    assert reward["unsafe_action"] is False


def test_task3_workflow_api_walkthrough() -> None:
    reset = client.post("/reset", json={"task_id": "task3", "seed": 44}).json()
    session_id = reset["session_id"]
    truth = env.sessions[session_id].__dict__["hidden_case"]["criterion_truth"]

    for criterion_id in ["INC-001", "INC-002", "INC-003"]:
        response = client.post(
            "/step",
            json={
                "session_id": session_id,
                "action": {
                    "action_type": "evaluate_criterion",
                    "criterion_id": criterion_id,
                    "evaluation": {
                        "criterion_id": criterion_id,
                        "verdict": truth[criterion_id],
                        "reasoning": f"Task3 workflow evaluation for {criterion_id}.",
                    },
                    "confidence_score": 0.92,
                },
            },
        )
        assert response.status_code == 200

    assert env.sessions[session_id].amendment_injected is True

    response = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {
                "action_type": "evaluate_criterion",
                "criterion_id": "INC-003",
                "evaluation": {
                    "criterion_id": "INC-003",
                    "verdict": env.sessions[session_id].__dict__["hidden_case"]["meta"]["post_amendment_truth"],
                    "reasoning": "Re-checked after amendment.",
                },
                "confidence_score": 0.95,
            },
        },
    )
    assert response.status_code == 200

    for criterion_id in ["INC-004", "INC-005", "INC-006", "EXC-001", "EXC-002", "EXC-003", "EXC-004"]:
        response = client.post(
            "/step",
            json={
                "session_id": session_id,
                "action": {
                    "action_type": "evaluate_criterion",
                    "criterion_id": criterion_id,
                    "evaluation": {
                        "criterion_id": criterion_id,
                        "verdict": truth[criterion_id],
                        "reasoning": f"Task3 workflow evaluation for {criterion_id}.",
                    },
                    "confidence_score": 0.9,
                },
            },
        )
        assert response.status_code == 200

    enroll = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {
                "action_type": "enroll",
                "final_decision_reason": "Safe enroll before scheduling.",
                "confidence_score": 0.95,
            },
        },
    )
    assert enroll.status_code == 200
    assert enroll.json()["done"] is False

    schedule = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {
                "action_type": "schedule_followup",
                "followup_day": 8,
                "confidence_score": 0.8,
            },
        },
    )
    assert schedule.status_code == 200
    assert schedule.json()["done"] is False

    safety = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {
                "action_type": "handle_safety_event",
                "safety_response": "escalate",
                "confidence_score": 0.88,
            },
        },
    )
    assert safety.status_code == 200
    assert safety.json()["done"] is True
    reward = safety.json()["reward"]
    assert reward["terminal_success"] is True
    assert reward["diagnostic_metrics"]["safety_component_score"] == 1.0
