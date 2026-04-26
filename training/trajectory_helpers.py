#!/usr/bin/env python3
"""Shared helpers for compact full-trajectory planning in training and local evaluation."""

from __future__ import annotations

import json
from typing import Any

from clinicaltrial_env.action import ActionType, ScreeningAction

ALLOWED_TRAJECTORY_ACTIONS = {
    ActionType.EVALUATE_CRITERION,
    ActionType.ASK_CLARIFICATION,
    ActionType.ENROLL,
    ActionType.EXCLUDE,
    ActionType.SCHEDULE_FOLLOWUP,
    ActionType.HANDLE_SAFETY_EVENT,
}


def _compact_criteria(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": criterion["criterion_id"],
            "clarify": bool(criterion.get("clarification_available", False)),
            "ambiguous": bool(criterion.get("is_ambiguous", False)),
        }
        for criterion in criteria
    ]


def summarize_observation(observation: dict[str, Any]) -> str:
    protocol = observation["trial_protocol_summary"]
    operational_state = observation.get("operational_state") or {}
    compact_summary = {
        "patient_id": observation["patient_id"],
        "step": observation["step_number"],
        "remaining": observation["steps_remaining"],
        "phase": operational_state.get("workflow_phase", "screening"),
        "amendment": bool(protocol.get("amendment_active", False)),
        "review_required": bool(operational_state.get("amendment_review_required", False)),
        "followup_window": [
            operational_state.get("followup_window_start"),
            operational_state.get("followup_window_end"),
        ],
        "safety_event": bool(operational_state.get("safety_event_active", False)),
        "age": observation["demographics"].get("age"),
        "weight_kg": observation["demographics"].get("weight_kg"),
        "diagnosis_code": observation["diagnosis"].get("icd10_code"),
        "diagnosis": observation["diagnosis"].get("primary_condition"),
        "labs": {
            key: [value.get("value"), value.get("certainty")]
            for key, value in observation["lab_values"].items()
        },
        "meds": [
            [medication.get("name"), medication.get("dose_mg"), medication.get("frequency")]
            for medication in observation["current_medications"]
        ],
        "inclusion": _compact_criteria(protocol["inclusion_criteria"]),
        "exclusion": _compact_criteria(protocol["exclusion_criteria"]),
        "message": observation.get("info_message"),
    }
    return json.dumps(compact_summary, separators=(",", ":"), ensure_ascii=True, default=str)


def build_episode_system_prompt() -> str:
    return (
        "Return ONLY a compact JSON object with one key named `trajectory`.\n"
        "The value must be a JSON array of valid ClinicalTrialEnv actions.\n"
        "Do not include markdown, prose, or any keys outside `trajectory`."
    )


def build_episode_user_message(
    observation: dict[str, Any],
    task_id: str,
    seed: int | None,
    max_actions: int,
    local_debug_mode: bool = False,
) -> str:
    seed_value = "unknown" if seed is None else str(seed)
    debug_constraints = ""
    if local_debug_mode:
        debug_constraints = (
            "\nDebug mode: keep the plan short, prioritize evaluate_criterion and ask_clarification, "
            "and terminate with a valid decision when allowed."
        )
    return (
        f"task={task_id} seed={seed_value} max_actions={max_actions}\n"
        "Allowed action_type values only: evaluate_criterion, ask_clarification, enroll, exclude, "
        "schedule_followup, handle_safety_event.\n"
        "Hard validity rules:\n"
        "- During screening: use only evaluate_criterion or ask_clarification until a final decision becomes valid.\n"
        "- After safe enroll: use schedule_followup.\n"
        "- During safety_event: use handle_safety_event.\n"
        "- Re-check INC-003 after amendment when task3 review is required.\n"
        "Trajectory format example:\n"
        '{"trajectory":[{"action_type":"evaluate_criterion","criterion_id":"INC-001","evaluation":{"criterion_id":"INC-001","verdict":"met","reasoning":"age in range"},"confidence_score":0.8},{"action_type":"exclude","final_decision_reason":"criterion failed"}]}\n'
        f"{debug_constraints}\n"
        f"Observation={summarize_observation(observation)}"
    )


def build_episode_prompt(
    observation: dict[str, Any],
    task_id: str,
    seed: int | None,
    max_actions: int,
    tokenizer: Any | None = None,
    local_debug_mode: bool = False,
) -> str:
    system_prompt = build_episode_system_prompt()
    user_prompt = build_episode_user_message(observation, task_id, seed, max_actions, local_debug_mode=local_debug_mode)
    prompt = f"{system_prompt}\n\n{user_prompt}"
    if tokenizer is not None:
        encoded = tokenizer(prompt, truncation=True, max_length=800, return_tensors=None)
        prompt = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    return prompt


def extract_json_block(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
            return text[index : index + end]
        except json.JSONDecodeError:
            continue
    repaired = _repair_truncated_json(text)
    if repaired is not None:
        return repaired
    raise ValueError("No valid JSON object found in model completion.")


def _repair_truncated_json(text: str) -> str | None:
    start = -1
    for i, c in enumerate(text):
        if c in "{[":
            start = i
            break
    if start < 0:
        return None
    fragment = text[start:]
    for suffix in ["]}", "]", "}", "}]}", "]}"]:
        candidate = fragment + suffix
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    last_comma = fragment.rfind(",")
    if last_comma > 0:
        trimmed = fragment[:last_comma]
        for suffix in ["]}", "]", "}"]:
            candidate = trimmed + suffix
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return None


def normalize_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def parse_trajectory_completion(completion_text: str, max_actions: int) -> list[dict[str, Any]]:
    raw_json = extract_json_block(completion_text.strip())
    parsed = json.loads(raw_json)
    if isinstance(parsed, dict):
        trajectory = parsed.get("trajectory")
    elif isinstance(parsed, list):
        trajectory = parsed
    else:
        raise ValueError("Completion must decode to a JSON object or list.")

    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Trajectory must be a non-empty list.")

    validated: list[dict[str, Any]] = []
    for action in trajectory[:max_actions]:
        try:
            validated_action = ScreeningAction.model_validate(action)
            if validated_action.action_type not in ALLOWED_TRAJECTORY_ACTIONS:
                continue
            validated.append(validated_action.model_dump(exclude_none=True))
        except Exception:
            continue
    if not validated:
        raise ValueError("No valid actions found in trajectory.")
    return validated
