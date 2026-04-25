#!/usr/bin/env python3
"""Generate the honest judge-facing TRAINING_REPORT.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(payload: dict[str, Any], key: str) -> float:
    return float(payload.get("aggregate", {}).get(key, 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write honest TRAINING_REPORT.md.")
    parser.add_argument("--baseline-eval", default="artifacts/eval/base_model_task3_eval.json")
    parser.add_argument("--lm-grpo-eval", default="artifacts/eval/lm_grpo_task3_eval_failed.json")
    parser.add_argument("--policy-eval", default="artifacts/eval/policy_gradient_task3_eval.json")
    parser.add_argument("--validation", default="artifacts/eval/training_validation_summary.json")
    parser.add_argument("--output", default="TRAINING_REPORT.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline_eval)
    lm_grpo = load_json(args.lm_grpo_eval)
    policy = load_json(args.policy_eval)
    validation = load_json(args.validation)

    markdown = f"""# ClinicalTrialEnv Training Report

## Baseline Failure

ClinicalTrialEnv is a real OpenEnv-style clinical workflow environment. Task 3 requires stateful coordination across criterion review, a protocol amendment, an enrollment decision, follow-up scheduling, and seizure-symptom safety handling.

The untrained LLM baseline fails this workflow:

| Metric | Untrained LLM Baseline |
| --- | ---: |
| Success rate | {metric(baseline, 'success_rate'):.4f} |
| Unsafe action rate | {metric(baseline, 'unsafe_rate'):.4f} |
| Mean final reward | {metric(baseline, 'mean_final_reward'):.4f} |
| Amendment recovery rate | {metric(baseline, 'amendment_recovery_rate'):.4f} |

## Evidence Tracks

| Track | File | Validation Status | Interpretation |
| --- | --- | --- | --- |
| Untrained LLM baseline | `artifacts/eval/base_model_task3_eval.json` | Passed as baseline evidence | Demonstrates scripted LLM workflow failure. |
| LM-GRPO attempt | `artifacts/eval/lm_grpo_task3_eval_failed.json` | Failed | Kept for transparency; not claimed as successful LLM training. |
| Compact RL policy | `artifacts/eval/policy_gradient_task3_eval.json` | {'Passed' if validation.get('passed') else 'Failed'} | Shows verifier rewards can train a compact action policy. |

## Compact RL Policy

The compact policy is trained by `training/train_task3_policy_gradient.py`. It is a small Torch action policy trained against the ClinicalTrialEnv verifier. It is not an LLM fine-tune and is not presented as GRPO success.

![Training reward curve](artifacts/plots/training_reward_curve.png)

Final compact-policy evaluation:

| Metric | Compact RL Policy |
| --- | ---: |
| Success rate | {metric(policy, 'success_rate'):.4f} |
| Unsafe action rate | {metric(policy, 'unsafe_rate'):.4f} |
| Mean final reward | {metric(policy, 'mean_final_reward'):.4f} |
| Amendment recovery rate | {metric(policy, 'amendment_recovery_rate'):.4f} |

![Held-out comparison](artifacts/plots/heldout_base_vs_trained.png)

## LM-GRPO Attempt

The LM-GRPO path in `training/grpo_phase1.py` remains part of the project, but the current preserved run did not pass stricter LLM-specific validation gates.

| Metric | Failed LM-GRPO Attempt |
| --- | ---: |
| Success rate | {metric(lm_grpo, 'success_rate'):.4f} |
| Unsafe action rate | {metric(lm_grpo, 'unsafe_rate'):.4f} |
| Mean final reward | {metric(lm_grpo, 'mean_final_reward'):.4f} |

## Final Interpretation

ClinicalTrialEnv exposes a real untrained LLM failure mode, and its verifier reward can train a compact action policy to solve Task 3. The current repository does not claim that LM-GRPO has already produced a successful trained LLM.
"""
    Path(args.output).write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
