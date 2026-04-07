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
    assert 0.0 < reward["total_reward"] < 1.0
    assert 0.0 < reward["eligibility_accuracy"] < 1.0
    assert 0.0 < reward["efficiency_bonus"] < 1.0
    assert 0.0 < reward["penalty"] < 1.0
    assert all(0.0 < value < 1.0 for value in reward["partial_credit"].values())


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
    assert 0.0 < response.json()["cumulative_reward"] < 1.0


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
    assert 0.0 < reward["total_reward"] < 1.0
    assert 0.0 < reward["eligibility_accuracy"] < 1.0
    assert 0.0 < reward["efficiency_bonus"] < 1.0
    assert 0.0 < reward["penalty"] < 1.0
    assert all(0.0 < value < 1.0 for value in reward["partial_credit"].values())
