#!/usr/bin/env python3
"""Shared stepwise action prompting/parsing for local RL training and evaluation."""

from __future__ import annotations

import json
from typing import Any

from clinicaltrial_env.action import ScreeningAction


INVALID_ACTION = "INVALID_ACTION"

VALID_ACTIONS = (
    "evaluate_criterion",
    "ask_clarification",
    "enroll",
    "exclude",
    "defer",
    "schedule_followup",
    "handle_safety_event",
)


def extract_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
            return text[index : index + end]
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in model output.")


def compact_action_json(action: dict[str, Any]) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=True)


def build_system_prompt() -> str:
    return """You are a clinical trial coordinator policy.

You must output ONLY one valid JSON action object. No markdown. No commentary.

Valid action_type values:
- evaluate_criterion
- ask_clarification
- enroll
- exclude
- defer
- schedule_followup
- handle_safety_event

Action schema examples:
{"action_type":"evaluate_criterion","criterion_id":"INC-001","evaluation":{"criterion_id":"INC-001","verdict":"met","reasoning":"age within range"},"confidence_score":0.82}
{"action_type":"ask_clarification","clarification_target":"INC-003","confidence_score":0.58}
{"action_type":"enroll","final_decision_reason":"all reviewed criteria are satisfied","confidence_score":0.77}
{"action_type":"exclude","final_decision_reason":"an exclusion criterion is active","confidence_score":0.74}
{"action_type":"defer","final_decision_reason":"eligibility remains unresolved","confidence_score":0.61}
{"action_type":"schedule_followup","followup_day":8,"confidence_score":0.88}
{"action_type":"handle_safety_event","safety_response":"escalate","confidence_score":0.91}

Rules:
- Return exactly one JSON object.
- Keep reasoning concise.
- Only include fields required by the chosen action_type.
- During screening, prefer evaluate_criterion and ask_clarification before final decisions.
- After a safe enroll in task3, schedule a follow-up inside the visible window.
- If a safety event is active, use handle_safety_event.
"""


def _criterion_brief(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "criterion_id": criterion["criterion_id"],
            "c": bool(criterion.get("clarification_available", False)),
            "a": bool(criterion.get("is_ambiguous", False)),
        }
        for criterion in criteria
    ]


def build_user_prompt(
    observation: dict[str, Any],
    reward: float,
    history: list[str],
    step: int,
    task_id: str,
) -> str:
    protocol = observation["trial_protocol_summary"]
    operational_state = observation.get("operational_state") or {}
    concise_labs = {
        key: [value.get("value"), value.get("certainty")]
        for key, value in observation["lab_values"].items()
    }
    concise_meds = [
        [medication.get("name"), medication.get("dose_mg"), medication.get("frequency")]
        for medication in observation["current_medications"]
    ]
    compact_obs = {
        "task_id": task_id,
        "patient_id": observation["patient_id"],
        "step": observation["step_number"],
        "remaining": observation["steps_remaining"],
        "reward": round(float(reward), 4),
        "phase": operational_state.get("workflow_phase", "screening"),
        "amendment": protocol.get("amendment_active", False),
        "review_required": operational_state.get("amendment_review_required", False),
        "followup": [operational_state.get("followup_window_start"), operational_state.get("followup_window_end")],
        "safety_event": operational_state.get("safety_event_active", False),
        "clarifications_remaining": observation.get("clarifications_remaining"),
        "demographics": {
            "age": observation["demographics"].get("age"),
            "sex": observation["demographics"].get("sex"),
            "weight_kg": observation["demographics"].get("weight_kg"),
        },
        "diagnosis": {
            "code": observation["diagnosis"].get("icd10_code"),
            "condition": observation["diagnosis"].get("primary_condition"),
            "stage": observation["diagnosis"].get("disease_stage"),
        },
        "labs": concise_labs,
        "meds": concise_meds,
        "inclusion_criteria": _criterion_brief(protocol["inclusion_criteria"]),
        "exclusion_criteria": _criterion_brief(protocol["exclusion_criteria"]),
        "info_message": observation.get("info_message"),
        "recent_history": history[-4:] if history else [],
    }
    return (
        f"Step {step}. Review the environment state and return one valid JSON action.\n"
        f"Observation={json.dumps(compact_obs, separators=(',', ':'), ensure_ascii=True, default=str)}"
    )


def build_generation_prompt(
    tokenizer: Any,
    observation: dict[str, Any],
    reward: float,
    history: list[str],
    step: int,
    task_id: str,
) -> str:
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(observation, reward, history, step, task_id)
    return f"{system_prompt}\n\n{user_prompt}"


def summarize_action_history(action: dict[str, Any] | str) -> str:
    if isinstance(action, str):
        return action[:48]
    action_type = action.get("action_type", "unknown")
    if action_type == "evaluate_criterion":
        return f"eval:{action.get('criterion_id')}:{action.get('evaluation', {}).get('verdict')}"
    if action_type == "ask_clarification":
        return f"clarify:{action.get('clarification_target')}"
    if action_type == "schedule_followup":
        return f"followup:{action.get('followup_day')}"
    if action_type == "handle_safety_event":
        return f"safety:{action.get('safety_response')}"
    return action_type


def parse_action_output(output: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(extract_json_object(output))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        validated = ScreeningAction.model_validate(payload)
    except Exception:
        return None
    return validated.model_dump(exclude_none=True)


def build_candidate_actions(
    observation: dict[str, Any],
    task_id: str,
    action_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import inference

    operational_state = observation.get("operational_state") or {}
    workflow_phase = operational_state.get("workflow_phase", "screening")
    candidates: list[dict[str, Any]] = []

    if workflow_phase == "followup_scheduling":
        start = operational_state.get("followup_window_start", 7)
        end = operational_state.get("followup_window_end", start)
        midpoint = int((start + end) / 2)
        candidates.extend(
            [
                {"action_type": "schedule_followup", "followup_day": start, "confidence_score": 0.72},
                {"action_type": "schedule_followup", "followup_day": midpoint, "confidence_score": 0.81},
                {"action_type": "schedule_followup", "followup_day": end, "confidence_score": 0.72},
            ]
        )
        return _dedupe_candidates(candidates)

    if workflow_phase == "safety_event":
        start = operational_state.get("followup_window_start", 7)
        end = operational_state.get("followup_window_end", start)
        midpoint = int((start + end) / 2)
        candidates.extend(
            [
                {"action_type": "handle_safety_event", "safety_response": "escalate", "confidence_score": 0.88},
                {
                    "action_type": "handle_safety_event",
                    "safety_response": "reschedule",
                    "reschedule_day": midpoint,
                    "confidence_score": 0.61,
                },
            ]
        )
        return _dedupe_candidates(candidates)

    criteria = observation["trial_protocol_summary"]["inclusion_criteria"] + observation["trial_protocol_summary"]["exclusion_criteria"]
    evaluated_ids = {
        record.get("criterion_id")
        for record in action_records
        if record.get("action_type") == "evaluate_criterion" and record.get("criterion_id")
    }
    clarified_ids = {
        record.get("clarification_target")
        for record in action_records
        if record.get("action_type") == "ask_clarification" and record.get("clarification_target")
    }

    candidates.append(inference.build_fallback_action(observation, task_id, action_records))

    pending = [criterion for criterion in criteria if criterion["criterion_id"] not in evaluated_ids]
    for criterion in pending[:2]:
        criterion_id = criterion["criterion_id"]
        if criterion.get("clarification_available") and criterion_id not in clarified_ids:
            candidates.append(
                {
                    "action_type": "ask_clarification",
                    "clarification_target": criterion_id,
                    "confidence_score": 0.56,
                }
            )
        verdict, reasoning = inference.evaluate_criterion_heuristically(observation, task_id, criterion_id)
        candidates.append(
            {
                "action_type": "evaluate_criterion",
                "criterion_id": criterion_id,
                "evaluation": {
                    "criterion_id": criterion_id,
                    "verdict": verdict,
                    "reasoning": reasoning,
                },
                "confidence_score": 0.73 if verdict != "uncertain" else 0.44,
            }
        )

    if action_records:
        final_action, reason = inference.choose_final_decision(observation, task_id, action_records)
        candidates.append(
            {
                "action_type": final_action,
                "final_decision_reason": reason,
                "confidence_score": 0.68,
            }
        )
        opposite = "exclude" if final_action == "enroll" else "enroll"
        candidates.append(
            {
                "action_type": opposite,
                "final_decision_reason": "Alternative final decision for policy exploration.",
                "confidence_score": 0.34,
            }
        )
        candidates.append(
            {
                "action_type": "defer",
                "final_decision_reason": "Eligibility remains unresolved pending further review.",
                "confidence_score": 0.41,
            }
        )

    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            validated = ScreeningAction.model_validate(candidate).model_dump(exclude_none=True)
        except Exception:
            continue
        key = compact_action_json(validated)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(validated)
    return deduped
