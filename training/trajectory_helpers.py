#!/usr/bin/env python3
"""Shared helpers for full-trajectory planning in training and local evaluation."""

from __future__ import annotations

import json
from typing import Any

from clinicaltrial_env.action import ActionType, ScreeningAction
from training.task3_anchor import TASK3_ANCHOR_TRAJECTORY

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
            "criterion_id": criterion["criterion_id"],
            "clarifiable": bool(criterion.get("clarification_available", False)),
            "ambiguous": bool(criterion.get("is_ambiguous", False)),
        }
        for criterion in criteria
    ]


def summarize_observation(observation: dict[str, Any]) -> str:
    protocol = observation["trial_protocol_summary"]
    operational_state = observation.get("operational_state") or {}
    compact_summary = {
        "patient_id": observation["patient_id"],
        "step_number": observation["step_number"],
        "steps_remaining": observation["steps_remaining"],
        "demographics": observation["demographics"],
        "diagnosis": observation["diagnosis"],
        "lab_values": {
            key: {
                "value": value["value"],
                "certainty": value["certainty"],
                "unit": value["unit"],
            }
            for key, value in observation["lab_values"].items()
        },
        "current_medications": [
            {
                "name": medication["name"],
                "dose_mg": medication["dose_mg"],
                "frequency": medication["frequency"],
                "is_contraindicated": medication.get("is_contraindicated"),
            }
            for medication in observation["current_medications"]
        ],
        "trial_protocol_summary": {
            "trial_id": protocol["trial_id"],
            "title": protocol["title"],
            "phase": protocol["phase"],
            "amendment_active": protocol["amendment_active"],
            "amendment_description": protocol.get("amendment_description"),
            "inclusion_criteria": _compact_criteria(protocol["inclusion_criteria"]),
            "exclusion_criteria": _compact_criteria(protocol["exclusion_criteria"]),
        },
        "operational_state": operational_state,
        "info_message": observation.get("info_message"),
    }
    return json.dumps(compact_summary, separators=(",", ":"), default=str)


def build_episode_system_prompt() -> str:
    return (
        "You are a clinical trial workflow planning model.\n"
        "Return exactly one compact JSON object with a `trajectory` list and no markdown or commentary.\n"
        "The first non-whitespace character of your reply must be `{`."
    )


def build_episode_user_message(
    observation: dict[str, Any],
    task_id: str,
    seed: int | None,
    max_actions: int,
) -> str:
    seed_value = "unknown" if seed is None else str(seed)
    minimal_schema = {"trajectory": TASK3_ANCHOR_TRAJECTORY}
    return (
        f"Task={task_id}; seed={seed_value}; max_actions={max_actions}.\n"
        "Allowed action_type values only: evaluate_criterion, ask_clarification, enroll, exclude, "
        "schedule_followup, handle_safety_event.\n"
        "Never use inspect_patient, inspect_protocol, or defer.\n"
        "Hard action clipping rule: during screening use only evaluate_criterion, ask_clarification, "
        "enroll, or exclude; after enrollment use only schedule_followup; during safety_event use only "
        "handle_safety_event.\n"
        "Keep each reasoning string under 6 words.\n"
        "Omit optional keys unless required by the action.\n"
        "During screening, evaluate criteria methodically and do not repeat a criterion unless the amendment "
        "requires re-checking INC-003.\n"
        "Only ask for clarification when a target is clarifiable and evidence is pending or estimated.\n"
        "After a safe enroll, schedule the follow-up inside the visible window. Prefer day 8 when valid.\n"
        "When the seizure-symptom safety event becomes active, respond with investigator escalation.\n"
        "Output schema example:\n"
        f"{json.dumps(minimal_schema, separators=(',', ':'))}\n"
        "Episode observation:\n"
        f"{summarize_observation(observation)}"
    )


def build_episode_prompt(
    observation: dict[str, Any],
    task_id: str,
    seed: int | None,
    max_actions: int,
    tokenizer: Any | None = None,
) -> str:
    system_prompt = build_episode_system_prompt()
    user_prompt = build_episode_user_message(observation, task_id, seed, max_actions)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system_prompt}\n\n{user_prompt}"


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
    raise ValueError("No valid JSON object found in model completion.")


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
        validated_action = ScreeningAction.model_validate(action)
        if validated_action.action_type not in ALLOWED_TRAJECTORY_ACTIONS:
            raise ValueError(f"Unsupported trajectory action_type: {validated_action.action_type.value}")
        validated.append(validated_action.model_dump(exclude_none=True))
    return validated
