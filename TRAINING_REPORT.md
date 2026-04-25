# ClinicalTrialEnv Training Report

## Baseline Failure

ClinicalTrialEnv is a real OpenEnv-style clinical workflow environment. Task 3 requires stateful coordination across criterion review, a protocol amendment, an enrollment decision, follow-up scheduling, and seizure-symptom safety handling.

The untrained LLM baseline fails this workflow:

| Metric | Untrained LLM Baseline |
| --- | ---: |
| Success rate | 0.0000 |
| Unsafe action rate | 0.3333 |
| Mean final reward | -0.3667 |
| Amendment recovery rate | 1.0000 |

## Evidence Tracks

| Track | File | Validation Status | Interpretation |
| --- | --- | --- | --- |
| Untrained LLM baseline | `artifacts/eval/base_model_task3_eval.json` | Passed as baseline evidence | Demonstrates scripted LLM workflow failure. |
| LM-GRPO attempt | `artifacts/eval/lm_grpo_task3_eval_failed.json` | Failed | Kept for transparency; not claimed as successful LLM training. |
| Compact RL policy | `artifacts/eval/policy_gradient_task3_eval.json` | Passed | Shows verifier rewards can train a compact action policy. |

## Compact RL Policy

The compact policy is trained by `training/train_task3_policy_gradient.py`. It is a small Torch action policy trained against the ClinicalTrialEnv verifier. It is not an LLM fine-tune and is not presented as GRPO success.

![Training reward curve](artifacts/plots/training_reward_curve.png)

Final compact-policy evaluation:

| Metric | Compact RL Policy |
| --- | ---: |
| Success rate | 1.0000 |
| Unsafe action rate | 0.0000 |
| Mean final reward | 1.0000 |
| Amendment recovery rate | 1.0000 |

![Held-out comparison](artifacts/plots/heldout_base_vs_trained.png)

## LM-GRPO Attempt

The LM-GRPO path in `training/grpo_phase1.py` remains part of the project, but the current preserved run did not pass stricter LLM-specific validation gates.

| Metric | Failed LM-GRPO Attempt |
| --- | ---: |
| Success rate | 0.3333 |
| Unsafe action rate | 0.0000 |
| Mean final reward | 0.6667 |

## Final Interpretation

ClinicalTrialEnv exposes a real untrained LLM failure mode, and its verifier reward can train a compact action policy to solve Task 3. The current repository does not claim that LM-GRPO has already produced a successful trained LLM.
