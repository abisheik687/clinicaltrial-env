"""Canonical Task 3 success trajectory used by training, plots, and demos."""

from __future__ import annotations

from typing import Any

TASK3_ANCHOR_SEED = 44
TASK3_MAX_REWARD = 1.0

TASK3_ANCHOR_TRAJECTORY: list[dict[str, Any]] = [
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "INC-001",
        "evaluation": {"criterion_id": "INC-001", "verdict": "met", "reasoning": "age ok"},
        "confidence_score": 0.95,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "INC-002",
        "evaluation": {"criterion_id": "INC-002", "verdict": "met", "reasoning": "mutation ok"},
        "confidence_score": 0.95,
    },
    {
        "action_type": "ask_clarification",
        "clarification_target": "INC-003",
        "confidence_score": 0.8,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "INC-003",
        "evaluation": {"criterion_id": "INC-003", "verdict": "met", "reasoning": "amended severity ok"},
        "confidence_score": 0.95,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "INC-004",
        "evaluation": {"criterion_id": "INC-004", "verdict": "met", "reasoning": "no prior gene therapy"},
        "confidence_score": 0.9,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "INC-005",
        "evaluation": {"criterion_id": "INC-005", "verdict": "met", "reasoning": "liver labs ok"},
        "confidence_score": 0.9,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "INC-006",
        "evaluation": {"criterion_id": "INC-006", "verdict": "met", "reasoning": "weight ok"},
        "confidence_score": 0.9,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "EXC-001",
        "evaluation": {"criterion_id": "EXC-001", "verdict": "not_met", "reasoning": "screening seizure controlled"},
        "confidence_score": 0.9,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "EXC-002",
        "evaluation": {"criterion_id": "EXC-002", "verdict": "not_met", "reasoning": "no AAV hypersensitivity"},
        "confidence_score": 0.9,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "EXC-003",
        "evaluation": {"criterion_id": "EXC-003", "verdict": "not_met", "reasoning": "no competing trial"},
        "confidence_score": 0.9,
    },
    {
        "action_type": "evaluate_criterion",
        "criterion_id": "EXC-004",
        "evaluation": {"criterion_id": "EXC-004", "verdict": "not_met", "reasoning": "life expectancy ok"},
        "confidence_score": 0.9,
    },
    {
        "action_type": "enroll",
        "final_decision_reason": "All criteria pass after the amendment re-check.",
        "confidence_score": 0.96,
    },
    {
        "action_type": "schedule_followup",
        "followup_day": 8,
        "confidence_score": 0.84,
    },
    {
        "action_type": "handle_safety_event",
        "safety_response": "escalate",
        "confidence_score": 0.9,
    },
]


def task3_anchor_completion() -> str:
    """Return the compact JSON completion used as the warm-start anchor."""
    import json

    return json.dumps({"trajectory": TASK3_ANCHOR_TRAJECTORY}, separators=(",", ":"))
