#!/usr/bin/env python3
"""Create a deterministic before/after trajectory diff for judge review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DECISION_ACTIONS = {"enroll", "exclude", "defer", "schedule_followup", "handle_safety_event"}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def action_types(episode: dict[str, Any]) -> list[str]:
    return [str(action.get("action_type", "")) for action in episode.get("trajectory", [])]


def compact_actions(episode: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for index, action in enumerate(episode.get("trajectory", []), start=1):
        item = {
            "step": index,
            "action_type": action.get("action_type"),
        }
        for key in ("criterion_id", "clarification_target", "followup_day", "safety_response", "confidence_score"):
            if key in action:
                item[key] = action[key]
        actions.append(item)
    return actions


def has_structural_diff(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_types = action_types(before)
    after_types = action_types(after)
    before_decisions = [action for action in before_types if action in DECISION_ACTIONS]
    after_decisions = [action for action in after_types if action in DECISION_ACTIONS]
    return len(before_types) != len(after_types) or before_decisions != after_decisions


def find_matching_pair(base_eval: dict[str, Any], trained_eval: dict[str, Any], preferred_seed: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    base_by_seed = {episode.get("seed"): episode for episode in base_eval.get("episodes", [])}
    trained_by_seed = {episode.get("seed"): episode for episode in trained_eval.get("episodes", [])}
    if preferred_seed is not None and preferred_seed in base_by_seed and preferred_seed in trained_by_seed:
        return base_by_seed[preferred_seed], trained_by_seed[preferred_seed]
    common_seeds = sorted(seed for seed in base_by_seed if seed in trained_by_seed)
    if not common_seeds:
        raise SystemExit("No matching seed found between baseline and trained eval artifacts.")
    for seed in common_seeds:
        if has_structural_diff(base_by_seed[seed], trained_by_seed[seed]):
            return base_by_seed[seed], trained_by_seed[seed]
    seed = common_seeds[0]
    return base_by_seed[seed], trained_by_seed[seed]


def build_payload(base_eval: dict[str, Any], trained_eval: dict[str, Any], preferred_seed: int | None) -> dict[str, Any]:
    before, after = find_matching_pair(base_eval, trained_eval, preferred_seed)
    before_types = action_types(before)
    after_types = action_types(after)
    before_decisions = [action for action in before_types if action in DECISION_ACTIONS]
    after_decisions = [action for action in after_types if action in DECISION_ACTIONS]
    structural_diff = len(before_types) != len(after_types) or before_decisions != after_decisions
    return {
        "seed": before.get("seed"),
        "structural_behavior_diff": structural_diff,
        "length_differs": len(before_types) != len(after_types),
        "decision_points_differ": before_decisions != after_decisions,
        "untrained_model": {
            "success": before.get("terminal_success"),
            "final_reward": before.get("final_reward"),
            "action_types": before_types,
            "decision_points": before_decisions,
            "actions": compact_actions(before),
        },
        "rl_trained_model": {
            "success": after.get("terminal_success"),
            "final_reward": after.get("final_reward"),
            "action_types": after_types,
            "decision_points": after_decisions,
            "actions": compact_actions(after),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate same-seed before/after trajectory diff.")
    parser.add_argument("--baseline-eval", default="artifacts/eval/base_model_task3_eval.json")
    parser.add_argument("--trained-eval", default="artifacts/eval/trained_task3_eval.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="artifacts/eval/before_after_trajectories.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(load_json(args.baseline_eval), load_json(args.trained_eval), args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["structural_behavior_diff"]:
        raise SystemExit("No structural behavior difference found.")


if __name__ == "__main__":
    main()
