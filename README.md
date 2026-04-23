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

ClinicalTrialEnv is an OpenEnv-compatible clinical trial operations environment for training agents to make safe enrollment decisions under partial information, protocol amendments, and operational constraints.

This finalist version is framed for **OpenEnv Theme 3.1: Professional Tasks**. The core angle is a **multi-app professional workflow** where the agent coordinates protocol rules, patient records, lab evidence, medication conflicts, and changing operational logic.

Training is **environment-driven, not dataset-driven**. The current patient cases are **synthetic, seed-deterministic, and generated inside the environment** so the same episode can be replayed exactly for training, evaluation, and demos. External data grounding is intentionally deferred until after real training evidence exists.

## Finalist Upgrade

- Rewarding dense intermediate behavior has been replaced by a **single terminal verifier**.
- Episode success is now defined by one question:
  - **Did the agent end with the correct safe final decision under the latest protocol state and revealed evidence?**
- Unsafe enrollment is deterministic:
  - enrolling while any exclusion is active
  - enrolling while any required inclusion is definitively unmet
- Invalid but schema-valid actions now return a small penalty instead of killing the rollout.
- The repo now includes a training/evaluation path:
  - `training/grpo_phase1.py`
  - `training/evaluate_models.py`
  - `training/plot_results.py`
  - `training/phase1_colab.ipynb`

## Environment Overview

Each episode presents one synthetic patient case and a summarized trial protocol. The agent can:

- inspect the patient state
- inspect the protocol state
- evaluate criteria
- request clarification
- finalize `enroll`, `exclude`, or `defer`
- schedule one follow-up visit after a safe enrollment
- handle one deterministic safety event before the follow-up visit

Task 3 is the Phase 1 training target because it concentrates the finalist behaviors into one bounded episode:

- ambiguous evidence
- clarification budgeting
- a visible amendment during screening
- one follow-up scheduling decision
- one judge-visible seizure-symptom safety event

## Tasks

### Task 1: Hypertension Screening

- clear inclusion and exclusion checks
- no clarification budget
- good for validation and deterministic walkthroughs

### Task 2: Oncology Screening

- compound marrow reasoning
- medication-dose interpretation
- medium-complexity operational logic

### Task 3: Minimal Clinical Trial Operations Extension

- gene-therapy screening with ambiguous severity evidence
- protocol amendment requiring re-check of `INC-003`
- follow-up scheduling inside a day `7` to `10` window after safe enrollment
- deterministic safety event: new seizure symptoms before follow-up
- required safety response: investigator escalation
- best Phase 1 GRPO target because it adds long-horizon workflow without exploding complexity

## Reward And Verifier

The environment now uses a verifier-centric reward. For task 3, the final score is the average of the applicable workflow components:

- `eligibility`: correct safe enroll/exclude decision under the latest protocol state
- `amendment`: correct handling of the amendment when it changes the active truth
- `scheduling`: valid follow-up scheduling inside the allowed window
- `safety`: correct response to the seizure-symptom event
- `+1`: all applicable verifier components satisfied
- `0`: one or more applicable verifier components failed
- `-1`: unsafe enrollment
- `-0.05`: invalid or impossible action that still fits the schema

Diagnostic metrics are tracked separately and do **not** define success:

- criterion evaluation accuracy
- clarification efficiency
- unsafe action rate
- amendment recovery rate
- eligibility component score
- scheduling component score
- safety component score

## Training-First Workflow

### Phase 0: Runtime And Validation

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
```

### Phase 1: Minimal Training Evidence

1. Start the environment locally:

```bash
.venv/Scripts/python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
```

2. Run a held-out baseline:

```bash
.venv/Scripts/python training/evaluate_models.py --policy fallback --output artifacts/eval/baseline_eval.json
```

3. Run Phase 1 GRPO:

```bash
.venv/Scripts/python training/grpo_phase1.py --env-url http://localhost:7860 --output-dir artifacts/phase1_grpo
```

4. Evaluate the trained checkpoint:

```bash
.venv/Scripts/python training/evaluate_models.py --policy local_model --model-name artifacts/phase1_grpo/model --output artifacts/eval/trained_eval.json
```

5. Generate the judging plots:

```bash
.venv/Scripts/python training/plot_results.py
```

For a notebook workflow, use [training/phase1_colab.ipynb](training/phase1_colab.ipynb).

## Judging Artifacts

The repo is set up to produce the two required evidence visuals:

- **Plot 1:** training reward curve
- **Plot 2:** held-out base vs trained comparison on:
  - success rate
  - unsafe rate
  - amendment recovery rate

The intended demo episode is a seeded Task 3 workflow where judges can see:

- the screening decision
- the amendment notice and required re-check
- the scheduled follow-up day
- the seizure-symptom safety event and the response decision

## Results

Add these committed artifacts before finals submission:

- `artifacts/eval/baseline_eval.json`
- `artifacts/phase1_grpo/train_log_history.json`
- `artifacts/eval/trained_eval.json`
- `artifacts/plots/training_reward_curve.png`
- `artifacts/plots/heldout_base_vs_trained.png`

README pass criteria for judges:

- the Hugging Face Space root URL renders the interactive **Clinical Trial Operations Arena** demo
- the `/health` endpoint returns `{"status":"ok", ...}`

## API

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `POST /validate_session`

The service is packaged for Docker/Hugging Face Spaces and listens on port `7860`.

## Validation

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/openenv.exe validate
docker build -t clinicaltrial-env .
```

## Why This Fits The Scaler Bonus Direction

This environment models a **multi-app professional workflow** where the agent coordinates protocol rules, patient evidence, lab interpretation, medication conflicts, and changing operational logic instead of solving a static puzzle.
