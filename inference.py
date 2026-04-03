#!/usr/bin/env python3
"""
ClinicalTrialEnv - Baseline Inference Script
Runs GPT-4o-mini (or configured model) as a clinical trial screening agent
against all 3 tasks and reports reproducible baseline scores.

Environment variables required:
  API_BASE_URL  - OpenAI-compatible API base URL
  MODEL_NAME    - Model identifier (default: gpt-4o-mini)
  HF_TOKEN      - API key (used as openai api_key)
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

BENCHMARK = "clinicaltrial-env"
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {"task_id": "task1", "seed": 42, "max_steps": 8, "name": "Single Criterion Screening"},
    {"task_id": "task2", "seed": 43, "max_steps": 14, "name": "Multi-Criteria Screening"},
    {"task_id": "task3", "seed": 44, "max_steps": 20, "name": "Amendment Screening"},
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    error_str = "null" if error is None else str(error).replace("\n", " ")
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}", flush=True)


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END]   success={str(success).lower()} steps={steps} rewards={reward_str}", flush=True)


def build_system_prompt() -> str:
    return """You are a clinical trial coordinator AI assistant. Your job is to screen 
patients for clinical trial eligibility by evaluating inclusion and exclusion criteria.

For each step, you must output a JSON action in EXACTLY this format:
{
  "action_type": "evaluate_criterion" | "ask_clarification" | "enroll" | "exclude" | "defer",
  "criterion_id": "INC-001",  // Required for evaluate_criterion
  "evaluation": {
    "criterion_id": "INC-001",
    "verdict": "met" | "not_met" | "uncertain",
    "reasoning": "Patient age is 45, within 18-75 range"
  },
  "clarification_target": null,  // criterion_id for ask_clarification
  "final_decision_reason": null,  // Required for enroll/exclude
  "confidence_score": 0.9
}

Strategy:
1. Evaluate all inclusion criteria first, then exclusion criteria
2. Use ask_clarification only for "pending" or "estimated" lab values
3. After evaluating all criteria, submit enroll or exclude with reason
4. Never defer - always make a decision
5. If a protocol amendment notice appears in the observation, re-evaluate affected criterion"""


def get_agent_action(client: OpenAI, observation: dict, reward: float, history: list[str], step: int) -> dict:
    """Get next action from LLM agent."""
    obs_summary = json.dumps(observation, indent=2, default=str)
    history_text = "\n".join(history[-5:]) if history else "No previous actions"
    user_message = f"""
Step {step} - Clinical Trial Screening

Current Patient Observation:
{obs_summary}

Last Reward: {reward:+.3f}
Recent History:
{history_text}

What is your next screening action? Output ONLY valid JSON as specified."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)


async def run_task(client: OpenAI, env_client: httpx.AsyncClient, task_config: dict) -> dict:
    """Run a single benchmark task."""
    task_id = task_config["task_id"]
    task_name = task_config["name"]
    seed = task_config["seed"]
    max_steps = task_config["max_steps"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
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
            try:
                action_dict = get_agent_action(client, observation, last_reward, history, step)
            except Exception as exc:
                log_step(step=step, action="ERROR", reward=0.0, done=False, error=str(exc))
                break

            action_str = json.dumps(action_dict, separators=(",", ":"))
            step_response = await env_client.post("/step", json={"session_id": session_id, "action": action_dict}, timeout=30.0)
            if step_response.status_code != 200:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=f"HTTP {step_response.status_code}")
                break

            step_data = step_response.json()
            observation = step_data["observation"]
            reward_obj = step_data["reward"]
            reward = reward_obj["total_reward"]
            done = step_data["done"]
            info = step_data.get("info", {})
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=info.get("error", None),
            )
            history.append(
                f"Step {step}: {action_dict.get('action_type')} "
                f"(criterion: {action_dict.get('criterion_id', 'N/A')}) "
                f"-> reward {reward:+.3f}"
            )
            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_results = []
    async with httpx.AsyncClient(base_url=ENV_BASE_URL) as env_client:
        try:
            health = await env_client.get("/health", timeout=10.0)
            health.raise_for_status()
        except Exception as exc:
            print(f"[ERROR] Environment not reachable at {ENV_BASE_URL}: {exc}", flush=True)
            sys.exit(1)

        for task_config in TASKS:
            result = await run_task(client, env_client, task_config)
            all_results.append(result)
            time.sleep(1)

    print("\n" + "=" * 60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for result in all_results:
        status = "PASS" if result["success"] else "FAIL"
        print(f"{status} | {result['task_id']} | score={result['score']:.4f} | steps={result['steps']}", flush=True)
    avg_score = sum(result["score"] for result in all_results) / len(all_results)
    print(f"\nAverage Score: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
