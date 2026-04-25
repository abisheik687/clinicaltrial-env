# ClinicalTrialEnv: Honest Training Results

## Result

The project has three distinct evidence tracks:

| Track | Status | Artifact |
| --- | --- | --- |
| Untrained LLM baseline | Valid baseline failure | `artifacts/eval/base_model_task3_eval.json` |
| LM-GRPO attempt | Failed validation | `artifacts/eval/lm_grpo_task3_eval_failed.json` |
| Compact RL policy | Passed action-policy validation | `artifacts/eval/policy_gradient_task3_eval.json` |

## What Is Proven

The environment proves that the untrained LLM baseline is brittle on Task 3. It fails the end-to-end workflow and has a non-zero unsafe action rate.

The compact RL policy proves that the verifier and reward loop can train an action policy to solve the environment:

| Metric | Untrained LLM Baseline | Compact RL Policy |
| --- | ---: | ---: |
| Success rate | 0.0000 | 1.0000 |
| Unsafe action rate | 0.3333 | 0.0000 |
| Mean final reward | -0.3667 | 1.0000 |
| Amendment recovery rate | 1.0000 | 1.0000 |

## What Is Not Claimed

The current repository does not claim that the LM-GRPO run successfully trained an LLM. The preserved LM-GRPO artifact is labeled as failed and exists to document the attempted path honestly.

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
