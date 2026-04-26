# Training Report

## 1. Objective

Train an LLM to solve **ClinicalTrialEnv Task 3** using **GRPO** against the OpenEnv HTTP environment.

Task 3 is the Rett syndrome workflow:

- screening
- amendment activation
- re-check of affected criteria
- final decision
- follow-up scheduling
- safety-event handling

The target was not text quality. The target was correct multi-step environment execution.

## 2. Setup

- **Model:** `distilgpt2` for local debugging
- **RL:** TRL `GRPOTrainer`
- **Environment:** ClinicalTrialEnv over OpenEnv HTTP APIs
- **Task:** `task3`

Core runtime path:

- model generates JSON trajectory candidates
- parser converts output into valid environment actions
- trajectory is replayed through `POST /step`
- scalar reward is returned to GRPO

## 3. Training Issues

The original LM training path was broken in multiple ways. These were addressed during debugging:

- **zero reward variance**  
  Fixed by repairing rollout parsing, ensuring valid fallback behavior, and restoring non-constant reward flow.

- **parsing failures**  
  Fixed by tightening the prompt format and routing invalid outputs through a valid fallback trajectory.

- **token overflow**  
  Fixed by truncating prompts consistently during dataset construction and rollout generation.

These fixes were enough to restore a measurable learning signal, but not enough to produce a reliable Task 3 policy.

## 4. Final Result

The repaired training run achieved:

- `reward_std > 0`
- non-zero `advantage_mean`
- non-zero loss and active gradients

However, the final outcome was still not strong enough to claim meaningful LLM improvement on Task 3.

In plain terms:

- the **training pipeline is no longer dead**
- the **environment reward signal is real**
- the **LM policy still did not become submission-grade**

## 5. Metrics Table

### Training Signal (repaired local GRPO run)

| Metric | Value |
| --- | ---: |
| reward mean | -0.0439 |
| reward std | 0.1292 |
| advantage mean | 0.1094 |
| advantage std | 0.1119 |
| loss | 0.000638 |
| grad norm | 8.1250 |
| invalid / unsafe trajectory rate | 0.0 |

Source: `artifacts/phase1_grpo_signal_fix/train_log_history.json`

### Evaluation Outcome

| Metric | Baseline | LM-GRPO | Compact Policy |
| --- | ---: | ---: | ---: |
| Success | 0.0 | 0.33 | 1.0 |
| Unsafe | 0.33 | 0.0 | 0.0 |
| Reward | -0.36 | 0.66 | 1.0 |

Environment-side reference files:

- baseline: `artifacts/eval/base_model_task3_eval.json`
- failed LM-GRPO summary: `artifacts/eval/lm_grpo_validation_summary_failed.json`
- compact RL policy: `artifacts/eval/policy_gradient_task3_eval.json`

Important note:

The LM-GRPO attempt produced partial signs of movement, but the validation summary still failed key gates:

- reward variance gate failed in the original validation attempt
- done-rate gate failed
- valid trajectory rate gate failed
- reward delta gate failed

That is why the LM result must be treated as **an unfinished attempt**, not a successful trained policy.

## 6. Key Insight

**Environment is learnable. Current LLM setup insufficient.**

The compact RL policy reaching perfect Task 3 performance shows the environment and reward design are workable. The bottleneck is the current language-model training setup, not the environment itself.

## 7. Conclusion

Pipeline works, learning signal exists, scaling required.

More specifically:

- ClinicalTrialEnv is a functioning RL environment
- the repaired GRPO training path now produces non-zero signal
- the compact policy proves the task is solvable
- the current LLM configuration still does not deliver reliable improvement on Task 3

This is progress, but not a finished LLM-RL success story yet.
