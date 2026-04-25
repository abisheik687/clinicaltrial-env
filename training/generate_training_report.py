#!/usr/bin/env python3
"""Generate the judge-facing TRAINING_REPORT.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def metric(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(payload.get("aggregate", {}).get(key, default))


def display_model_name(model: Any) -> str:
    model_text = str(model or "not available")
    normalized = model_text.replace("\\", "/")
    marker = "artifacts/phase1_grpo/model"
    if marker in normalized:
        return marker
    return model_text


def format_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "| Step | Action |\n| --- | --- |\n| - | No trajectory logged. |"
    rows = ["| Step | Action | Detail |", "| --- | --- | --- |"]
    for action in actions:
        detail_parts = []
        for key in ("criterion_id", "clarification_target", "followup_day", "safety_response"):
            if key in action and action[key] is not None:
                detail_parts.append(f"{key}={action[key]}")
        rows.append(f"| {action.get('step', '')} | `{action.get('action_type', '')}` | {', '.join(detail_parts) or '-'} |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write TRAINING_REPORT.md.")
    parser.add_argument("--baseline-eval", default="artifacts/eval/base_model_task3_eval.json")
    parser.add_argument("--trained-eval", default="artifacts/eval/trained_task3_eval.json")
    parser.add_argument("--validation", default="artifacts/eval/training_validation_summary.json")
    parser.add_argument("--before-after", default="artifacts/eval/before_after_trajectories.json")
    parser.add_argument("--output", default="TRAINING_REPORT.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline_eval, {})
    trained = load_json(args.trained_eval, {})
    validation = load_json(args.validation, {})
    before_after = load_json(args.before_after, {})

    passed = bool(validation.get("passed", False))
    reward_start = validation.get("initial_avg_reward")
    reward_end = validation.get("final_avg_reward")
    reward_caption = (
        f"Reward increases from {reward_start:.2f} -> {reward_end:.2f} over training steps"
        if isinstance(reward_start, (int, float)) and isinstance(reward_end, (int, float))
        else "Reward curve generated from available training log history"
    )
    improvement_claim = (
        "The final validation gates passed, so the report presents this as improved RL behavior."
        if passed
        else "The final validation gates did not pass, so this report is intentionally conservative and does not claim a winning RL improvement."
    )

    untrained_actions = before_after.get("untrained_model", {}).get("actions", [])
    trained_actions = before_after.get("rl_trained_model", {}).get("actions", [])
    advantage_mean = validation.get("advantage_mean")
    advantage_std = validation.get("advantage_std")
    advantage_line = (
        f"- GRPO advantage stats: `advantage_mean={advantage_mean:.4f}`, `advantage_std={advantage_std:.4f}`."
        if isinstance(advantage_mean, (int, float)) and isinstance(advantage_std, (int, float))
        else "- GRPO advantage stats were not available in the current log artifact."
    )

    markdown = f"""# ClinicalTrialEnv Training Report

## 1. Problem

Clinical trial coordination is a high-stakes professional workflow: a coordinator must combine protocol rules, patient records, lab evidence, medication conflicts, amendments, and safety events before taking action. The capability gap is that an LLM must maintain state over a changing workflow instead of answering a static eligibility question.

RL is needed because the useful behavior is sequential: inspect evidence, re-check criteria after amendment, make a safe enrollment decision, schedule the follow-up, and escalate the seizure-symptom safety event. A single prompt answer cannot prove the agent learned that workflow.

## 2. Environment

The Task 3 environment gives the agent a synthetic, seed-deterministic patient record and a protocol with inclusion/exclusion criteria. The state changes over time through a visible protocol amendment, a follow-up scheduling phase, and a deterministic safety event before the visit.

The agent can inspect patient/protocol state, evaluate criteria, request clarification, enroll/exclude/defer, schedule a follow-up day, and handle the safety event. The verifier checks whether the final workflow is safe under the latest protocol state.

## 3. Training Setup

- Target task: `task3` only.
- Model path in current eval artifact: `{display_model_name(trained.get('model'))}`.
- Training loop: GRPO through `training/grpo_phase1.py`, with the FastAPI environment served locally and queried through `/reset` and `/step`.
- Anchor gate: the known seed-44 correct trajectory must pass before training starts.
- Staged execution: short signal pass first, then longer run only if validation gates pass.
- Agent interacts with environment -> receives reward -> updates policy.
{advantage_line}

## 4. Baseline vs Trained

Validation status: **{'PASSED' if passed else 'NOT PASSED'}**. {improvement_claim}

### Untrained Model

{format_actions(untrained_actions)}

### RL-Trained Model

{format_actions(trained_actions)}

Structural behavior difference: `{before_after.get('structural_behavior_diff', validation.get('behavior_diff', {}).get('structural_behavior_diff', False))}`.

## 5. Reward Curve

![Training reward curve](artifacts/plots/training_reward_curve.png)

{reward_caption}.

![Held-out comparison](artifacts/plots/heldout_base_vs_trained.png)

## 6. Quantitative Results

| Metric | Untrained Model | RL-Trained Model |
| --- | ---: | ---: |
| Success rate | {metric(baseline, 'success_rate'):.4f} | {metric(trained, 'success_rate'):.4f} |
| Unsafe rate | {metric(baseline, 'unsafe_rate'):.4f} | {metric(trained, 'unsafe_rate'):.4f} |
| Mean reward | {metric(baseline, 'mean_final_reward'):.4f} | {metric(trained, 'mean_final_reward'):.4f} |
| Amendment recovery | {metric(baseline, 'amendment_recovery_rate'):.4f} | {metric(trained, 'amendment_recovery_rate'):.4f} |
| Fallback used rate | {metric(baseline, 'fallback_used_rate'):.4f} | {metric(trained, 'fallback_used_rate'):.4f} |

Validation gates:

| Gate | Value |
| --- | --- |
| Trajectory reward std | `{validation.get('trajectory_final_reward_std')}` |
| Episode done rate | `{validation.get('episode_done_rate')}` |
| Valid trajectory rate | `{validation.get('valid_trajectory_rate')}` |
| Max clipped ratio | `{validation.get('max_clipped_ratio')}` |
| Reward delta | `{validation.get('reward_delta')}` |
| Hard failure same success/reward/behavior | `{validation.get('hard_failure_same_success_reward_behavior')}` |

## 7. Key Learning

The desired learned behavior is not medical knowledge memorization. It is operational discipline: follow the current protocol version, avoid unsafe enrollment, make the follow-up scheduling decision, and escalate the safety event.

If the validation status above is `NOT PASSED`, the correct interpretation is that the environment and pipeline are ready, but the current run did not yet produce judge-proof learning evidence. The project should be submitted honestly or rerun on HF GPU until the gates pass.

## 8. Why This Matters

Clinical trials fail operationally when teams miss protocol changes, schedule outside allowed windows, or underreact to safety symptoms. This environment turns those real workflow risks into a bounded OpenEnv training loop for Theme 3.1 Professional Tasks, with secondary long-horizon planning pressure from the amendment -> enrollment -> scheduling -> safety sequence.
"""
    Path(args.output).write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
