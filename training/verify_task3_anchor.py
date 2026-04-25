#!/usr/bin/env python3
"""Hard-gate verifier for the canonical seed-44 Task 3 success path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.task3_anchor import TASK3_ANCHOR_SEED, TASK3_ANCHOR_TRAJECTORY, TASK3_MAX_REWARD


def replay_anchor(env_url: str) -> dict[str, Any]:
    with httpx.Client(base_url=env_url, timeout=60.0) as client:
        reset_response = client.post("/reset", json={"task_id": "task3", "seed": TASK3_ANCHOR_SEED})
        reset_response.raise_for_status()
        reset_payload = reset_response.json()
        session_id = reset_payload["session_id"]
        steps: list[dict[str, Any]] = []
        final_payload: dict[str, Any] | None = None

        for action in TASK3_ANCHOR_TRAJECTORY:
            step_response = client.post("/step", json={"session_id": session_id, "action": action})
            step_response.raise_for_status()
            final_payload = step_response.json()
            steps.append(
                {
                    "action_type": action["action_type"],
                    "criterion_id": action.get("criterion_id"),
                    "reward": final_payload["reward"]["total_reward"],
                    "done": final_payload["done"],
                    "terminal_success": final_payload["reward"]["terminal_success"],
                    "unsafe_action": final_payload["reward"]["unsafe_action"],
                    "feedback": final_payload["reward"]["verifier_feedback"],
                }
            )
            if final_payload["done"]:
                break

    if final_payload is None:
        raise RuntimeError("Anchor trajectory did not execute any steps.")

    reward = final_payload["reward"]
    diagnostics = reward.get("diagnostic_metrics", {})
    passed = (
        bool(final_payload["done"])
        and bool(reward["terminal_success"])
        and not bool(reward["unsafe_action"])
        and float(reward["total_reward"]) == TASK3_MAX_REWARD
        and diagnostics.get("eligibility_component_score") == 1.0
        and diagnostics.get("amendment_component_score") == 1.0
        and diagnostics.get("scheduling_component_score") == 1.0
        and diagnostics.get("safety_component_score") == 1.0
    )
    return {
        "passed": passed,
        "seed": TASK3_ANCHOR_SEED,
        "max_reward": TASK3_MAX_REWARD,
        "final_reward": reward,
        "steps": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the seed-44 Task 3 anchor trajectory before training.")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--output", default="artifacts/eval/task3_anchor_verification.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = replay_anchor(args.env_url)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["passed"]:
        raise SystemExit("Task 3 anchor trajectory failed. Stop training and fix the environment/path first.")
    print(f"Task 3 anchor verified: reward={payload['final_reward']['total_reward']} seed={payload['seed']}")


if __name__ == "__main__":
    main()
