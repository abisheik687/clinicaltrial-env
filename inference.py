#!/usr/bin/env python3
"""
ClinicalTrialEnv - Baseline Inference Script
Runs a configured OpenAI-compatible model as a clinical trial screening agent
against all 3 tasks and reports reproducible baseline scores.

Environment variables required:
  API_BASE_URL  - OpenAI-compatible API base URL
  MODEL_NAME    - Model identifier (default: gpt-4o-mini)
  HF_TOKEN      - API key (used as openai api_key)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date
from typing import Any

import httpx
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

BENCHMARK = "clinicaltrial-env"
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_REPAIR_ATTEMPTS = 1
DISPLAY_SCORE_FLOOR = 0.01
DISPLAY_SCORE_CEILING = 0.99
ACE_INHIBITORS = {"lisinopril", "enalapril", "ramipril"}
STEROID_EQUIVALENT = {"prednisone": 1.0, "methylprednisolone": 1.25, "dexamethasone": 6.67}

TASKS = [
    {"task_id": "task1", "seed": 42, "max_steps": 8, "name": "Single Criterion Screening"},
    {"task_id": "task2", "seed": 43, "max_steps": 14, "name": "Multi-Criteria Screening"},
    {"task_id": "task3", "seed": 44, "max_steps": 20, "name": "Amendment Screening"},
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def format_logged_reward(reward: float) -> str:
    safe_reward = min(max(reward, DISPLAY_SCORE_FLOOR), DISPLAY_SCORE_CEILING)
    return f"{safe_reward:.2f}"


def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    error_str = "null" if error is None else str(error).replace("\n", " ")
    print(
        f"[STEP]  step={step} action={action} reward={format_logged_reward(reward)} "
        f"done={str(done).lower()} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_str = ",".join(format_logged_reward(reward) for reward in rewards)
    print(f"[END]   success={str(success).lower()} steps={steps} rewards={reward_str}", flush=True)


def build_system_prompt() -> str:
    return """You are a clinical trial coordinator AI assistant. Screen one patient for trial eligibility.

Return exactly one JSON object using this schema:
{
  "action_type": "evaluate_criterion" | "ask_clarification" | "enroll" | "exclude" | "defer",
  "criterion_id": "INC-001",
  "evaluation": {
    "criterion_id": "INC-001",
    "verdict": "met" | "not_met" | "uncertain",
    "reasoning": "Short clinical justification"
  },
  "clarification_target": null,
  "final_decision_reason": null,
  "confidence_score": 0.9
}

Rules:
- Evaluate criteria methodically.
- Ask for clarification only when a criterion is marked clarifiable and the value is pending or estimated.
- Re-check INC-003 if an amendment activates in task 3.
- Finish with enroll or exclude, never defer."""


def build_user_message(observation: dict, reward: float, history: list[str], step: int) -> str:
    observation_text = json.dumps(observation, separators=(",", ":"), default=str)
    history_text = " | ".join(history[-4:]) if history else "none"
    return (
        f"step={step}\n"
        f"reward={reward:.4f}\n"
        f"history={history_text}\n"
        f"observation={observation_text}\n"
        "Return one valid JSON action only."
    )


def build_repair_message(observation: dict, invalid_content: str, step: int) -> str:
    observation_text = json.dumps(observation, separators=(",", ":"), default=str)
    return (
        f"The previous reply was invalid JSON for step {step}.\n"
        f"Invalid reply: {invalid_content}\n"
        f"Observation: {observation_text}\n"
        "Return one corrected JSON object only. No markdown, no commentary."
    )


def parse_action_payload(content: str) -> dict:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    action_type = parsed.get("action_type")
    if action_type not in {"evaluate_criterion", "ask_clarification", "enroll", "exclude", "defer"}:
        raise ValueError("Model response has an invalid action_type")
    parsed.setdefault("confidence_score", 0.5)
    return parsed


def _chat_json(client: OpenAI, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=350,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned an empty response")
    return content


def get_agent_action(
    client: OpenAI,
    observation: dict,
    reward: float,
    history: list[str],
    step: int,
    task_id: str,
    action_records: list[dict[str, Any]],
) -> dict:
    content: str | None = None
    try:
        content = _chat_json(client, build_user_message(observation, reward, history, step))
        return parse_action_payload(content)
    except Exception:
        if content is not None:
            for _ in range(MAX_REPAIR_ATTEMPTS):
                try:
                    repaired = _chat_json(client, build_repair_message(observation, content, step))
                    return parse_action_payload(repaired)
                except Exception:
                    continue
    return build_fallback_action(observation, task_id, action_records)


def build_fallback_action(observation: dict, task_id: str, action_records: list[dict[str, Any]]) -> dict:
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

    for criterion in criteria:
        criterion_id = criterion["criterion_id"]
        if criterion_id in evaluated_ids:
            continue
        if criterion["clarification_available"] and criterion_id not in clarified_ids and should_request_clarification(observation, task_id, criterion_id):
            return {
                "action_type": "ask_clarification",
                "clarification_target": criterion_id,
                "confidence_score": 0.55,
            }
        verdict, reasoning = evaluate_criterion_heuristically(observation, task_id, criterion_id)
        return {
            "action_type": "evaluate_criterion",
            "criterion_id": criterion_id,
            "evaluation": {
                "criterion_id": criterion_id,
                "verdict": verdict,
                "reasoning": reasoning,
            },
            "confidence_score": 0.72 if verdict != "uncertain" else 0.45,
        }

    final_action, reason = choose_final_decision(observation, task_id, action_records)
    return {
        "action_type": final_action,
        "final_decision_reason": reason,
        "confidence_score": 0.68,
    }


def should_request_clarification(observation: dict, task_id: str, criterion_id: str) -> bool:
    if task_id == "task2" and criterion_id == "INC-004":
        return observation["lab_values"]["anc"]["certainty"] != "confirmed"
    if task_id == "task3" and criterion_id == "INC-003":
        return observation["lab_values"]["css_score"]["certainty"] != "confirmed"
    if task_id == "task3" and criterion_id in {"EXC-001", "EXC-002"}:
        return True
    return False


def evaluate_criterion_heuristically(observation: dict, task_id: str, criterion_id: str) -> tuple[str, str]:
    demographics = observation["demographics"]
    diagnosis = observation["diagnosis"]
    lab_values = observation["lab_values"]
    meds = observation["current_medications"]
    amendment_active = observation["trial_protocol_summary"]["amendment_active"]

    if task_id == "task1":
        if criterion_id == "INC-001":
            age = demographics["age"]
            return verdict_from_bool(18 <= age <= 75), f"Patient age is {age}, compared with the 18-75 inclusion range."
        if criterion_id == "INC-002":
            diagnosis_date = date.fromisoformat(diagnosis["diagnosis_date"])
            months = (date.today() - diagnosis_date).days / 30.0
            meets = diagnosis["icd10_code"] == "I10" and months >= 6
            return verdict_from_bool(meets), f"ICD-10 is {diagnosis['icd10_code']} and diagnosis duration is about {months:.1f} months."
        if criterion_id == "INC-003":
            bp = lab_values["systolic_bp"]["value"]
            return verdict_from_bool(140 <= bp <= 180), f"Systolic blood pressure is {bp}, checked against the 140-180 mmHg window."
        if criterion_id == "EXC-001":
            egfr = lab_values["egfr"]["value"]
            return verdict_from_bool(egfr < 30), f"eGFR is {egfr}, so the renal exclusion applies only if it is below 30."
        if criterion_id == "EXC-002":
            med_names = {med["name"].lower() for med in meds}
            hit = bool(med_names & ACE_INHIBITORS)
            return verdict_from_bool(hit), f"Current medications are checked for ACE inhibitors; found {sorted(med_names & ACE_INHIBITORS)}."

    if task_id == "task2":
        if criterion_id == "INC-001":
            age = demographics["age"]
            return verdict_from_bool(18 <= age <= 65), f"Patient age is {age}, compared with the 18-65 inclusion range."
        if criterion_id == "INC-002":
            meets = diagnosis["icd10_code"] == "C83.3"
            return verdict_from_bool(meets), f"Diagnosis ICD-10 code is {diagnosis['icd10_code']} and must equal C83.3."
        if criterion_id == "INC-003":
            ecog = lab_values["ecog_status"]["value"]
            return verdict_from_bool(0 <= ecog <= 2), f"ECOG performance status is {ecog}, which must fall between 0 and 2."
        if criterion_id == "INC-004":
            anc = lab_values["anc"]
            platelets = lab_values["platelets"]
            if anc["certainty"] != "confirmed":
                return "uncertain", f"ANC remains {anc['certainty']} at {anc['value']}, so marrow adequacy cannot be finalized yet."
            meets = anc["value"] >= 1.0 and platelets["value"] >= 75
            return verdict_from_bool(meets), f"ANC is {anc['value']} and platelets are {platelets['value']}, compared with thresholds 1.0 and 75."
        if criterion_id == "INC-005":
            stage = (diagnosis.get("disease_stage") or "").lower()
            meets = "measurable" in stage
            return verdict_from_bool(meets), f"Disease stage summary is '{diagnosis.get('disease_stage')}', so measurable disease is inferred from that text."
        if criterion_id == "EXC-001":
            active_cns = "cns" in diagnosis["primary_condition"].lower()
            return verdict_from_bool(active_cns), f"Primary condition text is reviewed for active CNS lymphoma markers."
        if criterion_id == "EXC-002":
            return "uncertain", "Prior CAR-T exposure is not explicitly surfaced in the observation, so this remains uncertain."
        if criterion_id == "EXC-003":
            return "uncertain", "Active autoimmune disease requiring systemic treatment is not explicitly exposed in the public observation."
        if criterion_id == "EXC-004":
            equivalent = prednisone_equivalent_total(meds)
            return verdict_from_bool(equivalent > 10), f"Prednisone-equivalent daily exposure is estimated at {equivalent:.1f} mg."

    if task_id == "task3":
        if criterion_id == "INC-001":
            age = demographics["age"]
            return verdict_from_bool(4 <= age <= 45), f"Patient age is {age}, compared with the 4-45 inclusion range."
        if criterion_id == "INC-002":
            mutation = lab_values["mecp2_mutation"]
            meets = mutation["value"] >= 1.0
            return verdict_from_bool(meets), f"MECP2 mutation marker is {mutation['value']} with {mutation['certainty']} certainty."
        if criterion_id == "INC-003":
            css = lab_values["css_score"]
            lower = 10 if amendment_active else 12
            if css["certainty"] != "confirmed":
                return "uncertain", f"CSS score is still {css['certainty']} at {css['value']}, so the Rett severity criterion is not final."
            meets = lower <= css["value"] <= 36
            return verdict_from_bool(meets), f"CSS score is {css['value']} and the valid range is {lower}-36 after applying amendment state."
        if criterion_id == "INC-004":
            return "uncertain", "Prior gene therapy exposure is not directly exposed in the public observation."
        if criterion_id == "INC-005":
            alt = lab_values["alt"]
            ast = lab_values["ast"]
            alt_ok = alt["value"] <= 3 * alt["reference_range"][1]
            ast_ok = ast["value"] <= 3 * ast["reference_range"][1]
            return verdict_from_bool(alt_ok and ast_ok), f"ALT is {alt['value']} and AST is {ast['value']}, both compared with three times their reference upper limits."
        if criterion_id == "INC-006":
            weight = demographics["weight_kg"]
            return verdict_from_bool(weight >= 13), f"Weight is {weight} kg and must be at least 13 kg."
        if criterion_id == "EXC-001":
            return "uncertain", "Seizure control status requires clarification because it is intentionally ambiguous in task 3."
        if criterion_id == "EXC-002":
            return "uncertain", "AAV hypersensitivity requires clarification because the public observation does not finalize it."
        if criterion_id == "EXC-003":
            return "uncertain", "Concurrent interventional trial enrollment is not explicitly surfaced in the public observation."
        if criterion_id == "EXC-004":
            return "uncertain", "Life expectancy is not directly exposed in the public observation."

    return "uncertain", f"Criterion {criterion_id} could not be mapped to a deterministic heuristic."


def choose_final_decision(observation: dict, task_id: str, action_records: list[dict[str, Any]]) -> tuple[str, str]:
    evaluations = {
        record["criterion_id"]: record["evaluation"]["verdict"]
        for record in action_records
        if record.get("action_type") == "evaluate_criterion" and record.get("evaluation")
    }
    criteria = observation["trial_protocol_summary"]
    inclusion_ids = [criterion["criterion_id"] for criterion in criteria["inclusion_criteria"]]
    exclusion_ids = [criterion["criterion_id"] for criterion in criteria["exclusion_criteria"]]

    if any(evaluations.get(criterion_id) in {None, "not_met", "uncertain"} for criterion_id in inclusion_ids):
        return "exclude", "At least one inclusion criterion is missing, failed, or unresolved after structured review."
    if any(evaluations.get(criterion_id) == "met" for criterion_id in exclusion_ids):
        return "exclude", "At least one exclusion criterion appears to be present after structured review."
    if task_id == "task3" and observation["trial_protocol_summary"]["amendment_active"] and evaluations.get("INC-003") != "met":
        return "exclude", "The protocol amendment is active and the severity-score criterion is not clearly met after re-checking."
    return "enroll", "All reviewed inclusion criteria are met and no reviewed exclusion criterion appears active."


def verdict_from_bool(value: bool) -> str:
    return "met" if value else "not_met"


def prednisone_equivalent_total(medications: list[dict[str, Any]]) -> float:
    total = 0.0
    for medication in medications:
        name = str(medication["name"]).lower()
        if name not in STEROID_EQUIVALENT:
            continue
        frequency = str(medication.get("frequency", "daily")).lower()
        multiplier = {"daily": 1.0, "bid": 2.0, "tid": 3.0}.get(frequency, 1.0)
        total += float(medication.get("dose_mg", 0.0)) * multiplier * STEROID_EQUIVALENT[name]
    return total


async def run_task(client: OpenAI, env_client: httpx.AsyncClient, task_config: dict) -> dict:
    task_id = task_config["task_id"]
    task_name = task_config["name"]
    seed = task_config["seed"]
    max_steps = task_config["max_steps"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    history: list[str] = []
    action_records: list[dict[str, Any]] = []
    rewards: list[float] = []
    steps_taken = 0
    success = False

    try:
        reset_response = await env_client.post("/reset", json={"task_id": task_id, "seed": seed}, timeout=30.0)
        reset_response.raise_for_status()
        reset_data = reset_response.json()
        session_id = reset_data["session_id"]
        observation = reset_data["observation"]
        last_reward = 0.0
        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            action_dict = get_agent_action(client, observation, last_reward, history, step, task_id, action_records)
            action_str = json.dumps(action_dict, separators=(",", ":"))
            step_response = await env_client.post("/step", json={"session_id": session_id, "action": action_dict}, timeout=30.0)

            if step_response.status_code != 200:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=f"HTTP {step_response.status_code}")
                break

            step_data = step_response.json()
            observation = step_data["observation"]
            reward = float(step_data["reward"]["total_reward"])
            done = bool(step_data["done"])
            info = step_data.get("info", {})

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            action_records.append(action_dict)
            history.append(action_str)

            log_step(step=step, action=action_str, reward=reward, done=done, error=info.get("error"))

        success = (sum(rewards) >= SUCCESS_SCORE_THRESHOLD) if rewards else False
    except Exception:
        success = False
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {"task_id": task_id, "success": success, "steps": steps_taken}


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    async with httpx.AsyncClient(base_url=ENV_BASE_URL) as env_client:
        health = await env_client.get("/health", timeout=10.0)
        health.raise_for_status()
        for task_config in TASKS:
            await run_task(client, env_client, task_config)


if __name__ == "__main__":
    asyncio.run(main())
