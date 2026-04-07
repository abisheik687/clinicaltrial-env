"""Grader unit tests."""

from server.graders.task1_grader import Task1Grader
from server.graders.task2_grader import Task2Grader
from server.graders.task3_grader import Task3Grader


def test_task1_perfect_score() -> None:
    truth = {"INC-001": "met", "INC-002": "met", "INC-003": "met", "EXC-001": "not_met", "EXC-002": "not_met"}
    result = Task1Grader().grade(truth, truth.copy(), "enroll", {"final_eligible": True})
    assert result["score"] >= 0.95


def test_task1_wrong_final_decision_caps_score() -> None:
    truth = {"INC-001": "met", "INC-002": "met", "INC-003": "met", "EXC-001": "not_met", "EXC-002": "not_met"}
    result = Task1Grader().grade(truth, truth.copy(), "exclude", {"final_eligible": True})
    assert result["score"] <= 0.5


def test_task2_drug_interaction_penalty() -> None:
    truth = {**{f"INC-00{i}": "met" for i in range(1, 6)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    clean = Task2Grader().grade(truth, truth.copy(), "enroll", {"final_eligible": True, "drug_interaction_miss": False, "unnecessary_clarifications": 0})
    penalized = Task2Grader().grade(truth, truth.copy(), "enroll", {"final_eligible": True, "drug_interaction_miss": True, "unnecessary_clarifications": 0})
    assert penalized["score"] <= clean["score"] - 0.09


def test_task3_amendment_detection_bonus() -> None:
    truth = {**{f"INC-00{i}": "met" for i in range(1, 7)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    base = Task3Grader().grade(truth, truth.copy(), "enroll", {"final_eligible": True, "amendment_detected": False, "ambiguity_handled": False, "ignored_amendment": False, "steps_used": 18})
    boosted = Task3Grader().grade(truth, truth.copy(), "enroll", {"final_eligible": True, "amendment_detected": True, "ambiguity_handled": False, "ignored_amendment": False, "steps_used": 18})
    assert boosted["score"] >= base["score"] + 0.14


def test_task3_defer_penalty() -> None:
    truth = {**{f"INC-00{i}": "met" for i in range(1, 7)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    result = Task3Grader().grade(truth, truth.copy(), "defer", {"final_eligible": True, "amendment_detected": False, "ambiguity_handled": False, "ignored_amendment": False, "steps_used": 10})
    assert result["score"] <= 0.4


def test_grader_scores_in_valid_range() -> None:
    truth1 = {"INC-001": "met", "INC-002": "met", "INC-003": "not_met", "EXC-001": "not_met", "EXC-002": "met"}
    truth2 = {**{f"INC-00{i}": "met" for i in range(1, 6)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    truth3 = {**{f"INC-00{i}": "met" for i in range(1, 7)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    cases = [
        Task1Grader().grade(truth1, {}, "exclude", {"final_eligible": False}),
        Task2Grader().grade(truth2, {}, "exclude", {"final_eligible": False, "drug_interaction_miss": True, "unnecessary_clarifications": 2}),
        Task3Grader().grade(truth3, {}, "defer", {"final_eligible": True, "amendment_detected": False, "ambiguity_handled": False, "ignored_amendment": True, "steps_used": 25}),
    ]
    assert all(0.0 < case["score"] < 1.0 for case in cases)


def test_perfect_scores_stay_strictly_below_one() -> None:
    truth1 = {"INC-001": "met", "INC-002": "met", "INC-003": "met", "EXC-001": "not_met", "EXC-002": "not_met"}
    truth2 = {**{f"INC-00{i}": "met" for i in range(1, 6)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    truth3 = {**{f"INC-00{i}": "met" for i in range(1, 7)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    perfect_cases = [
        Task1Grader().grade(truth1, truth1.copy(), "enroll", {"final_eligible": True}),
        Task2Grader().grade(truth2, truth2.copy(), "enroll", {"final_eligible": True, "drug_interaction_miss": False, "unnecessary_clarifications": 0}),
        Task3Grader().grade(truth3, truth3.copy(), "enroll", {"final_eligible": True, "amendment_detected": True, "ambiguity_handled": True, "ignored_amendment": False, "steps_used": 10}),
    ]
    assert all(case["score"] < 1.0 for case in perfect_cases)


def test_worst_scores_stay_strictly_above_zero() -> None:
    truth1 = {"INC-001": "met", "INC-002": "met", "INC-003": "not_met", "EXC-001": "not_met", "EXC-002": "met"}
    truth2 = {**{f"INC-00{i}": "met" for i in range(1, 6)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    truth3 = {**{f"INC-00{i}": "met" for i in range(1, 7)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    worst_cases = [
        Task1Grader().grade(truth1, {}, "enroll", {"final_eligible": False}),
        Task2Grader().grade(truth2, {}, "enroll", {"final_eligible": False, "drug_interaction_miss": True, "unnecessary_clarifications": 4}),
        Task3Grader().grade(truth3, {}, "defer", {"final_eligible": False, "amendment_detected": False, "ambiguity_handled": False, "ignored_amendment": True, "steps_used": 30}),
    ]
    assert all(case["score"] > 0.0 for case in worst_cases)


def test_scores_survive_two_decimal_rounding_inside_open_interval() -> None:
    truth1 = {"INC-001": "met", "INC-002": "met", "INC-003": "not_met", "EXC-001": "not_met", "EXC-002": "met"}
    truth2 = {**{f"INC-00{i}": "met" for i in range(1, 6)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    truth3 = {**{f"INC-00{i}": "met" for i in range(1, 7)}, **{f"EXC-00{i}": "not_met" for i in range(1, 5)}}
    scores = [
        Task1Grader().grade(truth1, {}, "enroll", {"final_eligible": False})["score"],
        Task1Grader().grade(truth1, truth1.copy(), "enroll", {"final_eligible": True})["score"],
        Task2Grader().grade(truth2, {}, "enroll", {"final_eligible": False, "drug_interaction_miss": True, "unnecessary_clarifications": 4})["score"],
        Task2Grader().grade(truth2, truth2.copy(), "enroll", {"final_eligible": True, "drug_interaction_miss": False, "unnecessary_clarifications": 0})["score"],
        Task3Grader().grade(truth3, {}, "defer", {"final_eligible": False, "amendment_detected": False, "ambiguity_handled": False, "ignored_amendment": True, "steps_used": 30})["score"],
        Task3Grader().grade(truth3, truth3.copy(), "enroll", {"final_eligible": True, "amendment_detected": True, "ambiguity_handled": True, "ignored_amendment": False, "steps_used": 10})["score"],
    ]
    rounded = [float(f"{score:.2f}") for score in scores]
    assert all(0.0 < value < 1.0 for value in rounded)
