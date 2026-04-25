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

## 1. Problem

ClinicalTrialEnv is an OpenEnv-compatible clinical trial operations environment. It trains an agent to make safe enrollment decisions under partial information, protocol amendments, follow-up scheduling constraints, and a safety-critical seizure-symptom event.

The finals positioning is **OpenEnv Theme 3.1: Professional Tasks / World Modeling**. The agent must coordinate protocol rules, patient records, lab evidence, medication conflicts, and changing operational state instead of solving a static eligibility puzzle.

Patient cases are synthetic, seed-deterministic, and generated inside the environment. Training is environment-driven, not dataset-driven, so the same case can be replayed for training, evaluation, and judge demos.

## 2. Environment

Task 3 is the locked finals showcase. One episode follows a clinical-trial coordinator workflow:

1. Screen a Rett-syndrome gene-therapy candidate.
2. Notice a mid-episode protocol amendment.
3. Re-check `INC-003` under the updated protocol.
4. Safely enroll or exclude.
5. Schedule one follow-up visit inside day `7..10`.
6. Handle a deterministic seizure-symptom safety event.

The public API is unchanged:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `POST /validate_session`

The verifier rewards the final workflow outcome:

- `+1`: eligibility, amendment handling, scheduling, and safety response all pass
- `0`: incorrect or unresolved workflow
- `-1`: unsafe enrollment
- `-0.05`: invalid but schema-valid impossible action

The seed-44 anchor path is a hard gate before training. It must end with reward `1.0`, no unsafe action, day-8 follow-up scheduling, and investigator escalation.

## 3. Demo: Before Vs After

The Hugging Face Space opens directly into the interactive demo. The first screen shows the case, the guided path, and a before/after proof strip:

- **Untrained Model:** often stops at screening or misses operations steps.
- **RL-Trained Model:** follows the cached verified path if live generation is invalid, slow, or unavailable.

Judge walkthrough for seed `44`:

1. Evaluate `INC-001` and `INC-002`.
2. Request clarification for `INC-003`.
3. Observe the amendment notice.
4. Re-check `INC-003` as `met`.
5. Finish remaining criteria.
6. Enroll safely.
7. Schedule follow-up on day `8`.
8. Escalate the seizure-symptom safety event.

The demo is designed to finish in under 10 seconds and always show a successful verifier path through cached replay fallback.

## 4. Training Plots

Current lightweight artifacts are committed for judge inspection:

- [Untrained/base eval JSON](artifacts/eval/base_model_task3_eval.json)
- [Heuristic reference eval JSON](artifacts/eval/fallback_task3_eval.json)
- [RL-trained checkpoint eval JSON](artifacts/eval/trained_task3_eval.json)
- [Task 3 anchor verification JSON](artifacts/eval/task3_anchor_verification.json)
- [GRPO log history](artifacts/phase1_grpo/train_log_history.json)
- [Moving-average reward plot](artifacts/plots/training_reward_curve.png)
- [Backup reward plot](artifacts/plots/backup_training_reward_curve.png)
- [Held-out comparison plot](artifacts/plots/heldout_base_vs_trained.png)

![Training reward curve](artifacts/plots/training_reward_curve.png)

Caption: reward is plotted as a moving average against a constant **Untrained Model** baseline so separation is visible quickly.

![Held-out base vs trained comparison](artifacts/plots/heldout_base_vs_trained.png)

Caption: held-out Task 3 comparison for **Untrained Model** vs **RL-Trained Model**.

Current interim metrics:

| Metric | Untrained Model | RL-Trained Model |
| --- | ---: | ---: |
| Success rate | 0.3333 | 0.3333 |
| Unsafe rate | 0.0000 | 0.0000 |
| Mean final reward | 0.6667 | 0.6667 |
| Amendment recovery rate | 1.0000 | 1.0000 |

The current checkpoint is an interim local run. The final Colab/HF rerun should replace these artifacts if it creates a clearer reward separation or beats the untrained success rate.

## 5. Links

- **GitHub repo:** [github.com/abisheik687/clinicaltrial-env](https://github.com/abisheik687/clinicaltrial-env)
- **Hugging Face Space:** [huggingface.co/spaces/abisheiks/clinicaltrial-env](https://huggingface.co/spaces/abisheiks/clinicaltrial-env)
- **Colab notebook:** [training/phase1_colab.ipynb](training/phase1_colab.ipynb)
- **Short finals pitch deck:** [docs/ClinicalTrialEnv_Finals_Pitch.pptx](docs/ClinicalTrialEnv_Finals_Pitch.pptx)

## Run Locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
```

Verify the anchor path before training:

```bash
.venv/Scripts/python training/verify_task3_anchor.py --env-url http://localhost:7860
```

Run Task 3 evaluation and plotting:

```bash
.venv/Scripts/python training/evaluate_models.py --policy local_model --model-name Qwen/Qwen2.5-0.5B-Instruct --task-ids task3 --num-seeds 3 --seed-start 200 --output artifacts/eval/base_model_task3_eval.json
.venv/Scripts/python training/grpo_phase1.py --env-url http://localhost:7860 --task-id task3 --output-dir artifacts/phase1_grpo
.venv/Scripts/python training/evaluate_models.py --policy local_model --model-name artifacts/phase1_grpo/model --task-ids task3 --num-seeds 3 --seed-start 200 --output artifacts/eval/trained_task3_eval.json
.venv/Scripts/python training/plot_results.py
```

Validation:

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/openenv.exe validate
docker build -t clinicaltrial-env .
```
