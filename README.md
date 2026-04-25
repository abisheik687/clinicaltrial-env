---
title: ClinicalTrialEnv
emoji: "🏥"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - clinical-trials
  - reinforcement-learning
  - healthcare
  - professional-workflow
app_port: 7860
---

# ClinicalTrialEnv

ClinicalTrialEnv is an OpenEnv-compatible clinical trial operations environment. It tests whether an agent can execute a changing professional workflow: screen a synthetic patient, react to a mid-episode protocol amendment, make a safe enrollment decision, schedule follow-up when appropriate, and handle a seizure-symptom safety event.

The current evidence should be read carefully:

- The **environment and untrained LLM baseline failure are real**.
- The **LM-GRPO training attempt is preserved but has not passed validation**.
- The **compact RL policy is a separate verifier-trained action policy**, not proof that the LLM checkpoint learned the workflow.

## Environment

Task 3 is the finals showcase. One episode requires the agent to:

1. Screen a Rett-syndrome gene-therapy candidate.
2. Notice a mid-episode protocol amendment.
3. Re-check `INC-003` under the amended CSS threshold.
4. Safely enroll or exclude.
5. Schedule one follow-up visit inside day `7..10` if enrolled.
6. Escalate the deterministic seizure-symptom safety event.

The public API is unchanged:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `POST /validate_session`

## Evidence Tracks

| Track | Artifact | Status | What It Means |
| --- | --- | --- | --- |
| Untrained LLM baseline | `artifacts/eval/base_model_task3_eval.json` | Valid | Shows brittle scripted behavior and unsafe enrollment risk. |
| LM-GRPO attempt | `artifacts/eval/lm_grpo_task3_eval_failed.json` | Failed | Preserved as an honest failed LLM fine-tuning attempt. |
| Compact RL policy | `artifacts/eval/policy_gradient_task3_eval.json` | Passed | Shows the verifier/reward loop can train an action policy on Task 3. |

![Training reward curve](artifacts/plots/training_reward_curve.png)

Caption: compact action-policy reward improves against the untrained LLM baseline. This plot is not an LM-GRPO success claim.

![Held-out base vs trained comparison](artifacts/plots/heldout_base_vs_trained.png)

Caption: held-out Task 3 comparison for the **Untrained LLM Baseline** and **Compact RL Policy**.

## Current Metrics

| Metric | Untrained LLM Baseline | Failed LM-GRPO Attempt | Compact RL Policy |
| --- | ---: | ---: | ---: |
| Success rate | 0.0000 | 0.3333 | 1.0000 |
| Unsafe rate | 0.3333 | 0.0000 | 0.0000 |
| Mean final reward | -0.3667 | 0.6667 | 1.0000 |
| Amendment recovery rate | 1.0000 | 1.0000 | 1.0000 |

The compact policy is intentionally labeled separately because it is not a generated-language checkpoint. It trains a small Torch action policy directly against the same ClinicalTrialEnv verifier rewards.

## Run Locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
```

Verify the anchor path:

```bash
.venv/Scripts/python training/verify_task3_anchor.py --env-url http://localhost:7860
```

Regenerate the compact policy artifacts:

```bash
.venv/Scripts/python training/train_task3_policy_gradient.py --seed-start 200 --train-seed-count 50 --eval-seed-start 200 --eval-num-seeds 50 --train-steps 15 --batch-size 50 --learning-rate 0.3 --log-every 1
.venv/Scripts/python training/validate_training_outputs.py --mode action_policy --train-log artifacts/phase1_pg/train_log_history.json --trained-eval artifacts/eval/policy_gradient_task3_eval.json
```

Attempt LM-GRPO separately:

```bash
.venv/Scripts/python training/grpo_phase1.py --env-url http://localhost:7860 --task-id task3 --output-dir artifacts/phase1_grpo
.venv/Scripts/python training/validate_training_outputs.py --mode lm_grpo --allow-failed --train-log artifacts/phase1_grpo/train_log_history.json --trained-eval artifacts/eval/lm_grpo_task3_eval_failed.json
```

## Links

- **GitHub repo:** [github.com/abisheik687/clinicaltrial-env](https://github.com/abisheik687/clinicaltrial-env)
- **Hugging Face Space:** [huggingface.co/spaces/abisheiks/clinicaltrial-env](https://huggingface.co/spaces/abisheiks/clinicaltrial-env)
- **Artifact manifest:** [artifacts/eval/artifact_manifest.json](artifacts/eval/artifact_manifest.json)
- **Training report:** [TRAINING_REPORT.md](TRAINING_REPORT.md)

## Validation

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/openenv.exe validate
docker build -t clinicaltrial-env .
```
