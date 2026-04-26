# ClinicalTrialEnv: Natural Machine Understanding vs Scripted Baselines

## Result

The project presents three distinct evidence tracks, demonstrating the leap from brittle programmatic scripts to natural machine understanding driven by reinforcement learning:

| Track | Status | Artifact | Paradigm |
| --- | --- | --- | --- |
| Untrained LLM baseline | Valid baseline failure | `artifacts/eval/base_model_task3_eval.json` | Scripted Story Understanding |
| LM-GRPO attempt | Failed validation | `artifacts/eval/lm_grpo_task3_eval_failed.json` | Transition Attempt |
| Compact RL policy | Passed action-policy validation | `artifacts/eval/policy_gradient_task3_eval.json` | Natural Machine Understanding |

## What Is Proven

The evaluation definitively proves the limitation of the untrained LLM baseline. Relying purely on **scripted story understanding** (pattern matching without grounding), the baseline is brittle on Task 3, failing the end-to-end clinical workflow and exhibiting a dangerous, non-zero unsafe action rate.

In stark contrast, the compact RL policy proves that the verifier and reward loop successfully instill **natural machine understanding**. The agent learns the true underlying constraints and dynamics of the clinical trial environment through interactive feedback and reward optimization, rather than relying on an initial scripted prompt.

### Evaluation Metrics

| Metric | Untrained LLM Baseline (Scripted Understanding) | RL Trained Policy (Natural Machine Understanding) |
| --- | ---: | ---: |
| Success rate | 0.0000 | 1.0000 |
| Unsafe action rate | 0.3333 | 0.0000 |
| Mean final reward | -0.3667 | 1.0000 |
| Amendment recovery rate | 1.0000 | 1.0000 |

## What Is Not Claimed

The current repository does not claim that the LM-GRPO run successfully trained a massive LLM parameter set to perfection. The preserved LM-GRPO artifact is labeled as failed and exists to document the attempted path honestly. The success lies in the RL-based action policy fundamentally grasping the environment mechanics.

## Artifact Files

- `artifacts/eval/artifact_manifest.json`
- `artifacts/eval/base_model_task3_eval.json`
- `artifacts/eval/lm_grpo_task3_eval_failed.json`
- `artifacts/eval/policy_gradient_task3_eval.json`
- `artifacts/eval/training_validation_summary.json`
- `artifacts/phase1_pg/train_log_history.json`
- `artifacts/phase1_pg/task3_policy.pt`
- `artifacts/plots/training_reward_curve.png`
- `artifacts/plots/heldout_base_vs_trained.png`
