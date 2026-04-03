---
title: ClinicalTrialEnv
emoji: 🏥
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
app_port: 7860
---

# ClinicalTrialEnv 🏥

> **An RL environment where AI agents learn to screen, evaluate, and enroll patients in clinical trials — the way a real clinical trial coordinator does.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/openenv-ai) [![HF Spaces](https://img.shields.io/badge/HF-Spaces-green)](https://huggingface.co/spaces) [![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://www.docker.com/)

## 🎯 Hackathon Problem Statement Alignment

This environment was built for the OpenEnv Hackathon Round 1. Every major judging axis is mapped explicitly to the implementation.

| Judging Criterion | Weight | How ClinicalTrialEnv Addresses It |
|---|---|---|
| Real-world utility | 30% | Clinical trial coordination is a real operational bottleneck in pharma and CRO workflows. The environment models patient screening, safety exclusions, clarifications, and enrollment decisions. |
| Task & grader quality | 25% | Three tasks progress from easy to hard, each with deterministic graders, explicit rubrics, and bounded scores in `[0.0, 1.0]`. |
| Environment design | 20% | The environment includes a true state machine, structured observation/action/reward models, protocol amendments, session isolation, and shaped rewards with partial progress signals. |
| Code quality & spec | 15% | FastAPI API, typed Pydantic v2 models, `openenv.yaml`, deterministic data generation, Docker packaging, and tests are all included. |
| Creativity & novelty | 10% | The design introduces clinical screening logic, clarification requests for uncertain data, and a mid-episode protocol amendment that forces adaptive reasoning. |

### Why This Domain

Clinical trial coordinators screen large candidate pools under strict safety and protocol rules. Incorrect enrollment decisions can expose patients to harm, delay enrollment, and increase trial costs. ClinicalTrialEnv models:

- deterministic inclusion and exclusion rules
- uncertainty through `pending` and `estimated` values
- protocol amendments that change eligibility mid-episode
- partial-information decision making with auditable reasoning

No toy game mechanics are used. The environment is built around a real-world administrative and safety-critical workflow.

## 📋 Environment Overview

Each episode presents a synthetic patient case and a summarized trial protocol. The agent acts as a clinical trial coordinator. It can evaluate criteria, request clarification for uncertain values, and then decide whether to enroll or exclude the patient. Task 3 adds a protocol amendment at step 6, reflecting how real trials are updated mid-study.

The environment is deterministic by seed, exposes an OpenEnv-style API, and stores hidden ground truth internally for grading and reward shaping. That makes it suitable for benchmarking LLM agents, RL policies, and hybrid decision systems.

## 🎮 Tasks

### Task 1: Single Criterion Screening (Easy)

- Protocol: hypertension Phase III trial
- Inclusion criteria: age, confirmed hypertension history, systolic blood pressure range
- Exclusion criteria: severe renal impairment, ACE inhibitor use
- Clarification budget: `0`
- Max steps: `8`
- Grading: `(correct criteria / 5) * 0.6 + correct final decision * 0.4`
- Expected baseline score: `0.85`

### Task 2: Multi-Criteria Oncology Screening (Medium)

- Protocol: CAR-T therapy Phase II trial
- Inclusion criteria: age, DLBCL diagnosis, ECOG status, marrow function, measurable disease
- Exclusion criteria: active CNS lymphoma, prior CAR-T, autoimmune disease, excessive corticosteroids
- Clarification budget: `2`
- Max steps: `14`
- Special logic: ANC clarification, compound marrow criterion, corticosteroid dose reasoning
- Expected baseline score: `0.65`

### Task 3: Ambiguous Gene Therapy Screening (Hard)

- Protocol: rare disease gene therapy Phase I/II
- Inclusion criteria: pediatric/adult age range, MECP2 mutation, CSS severity score, no prior gene therapy, hepatic function, minimum weight
- Exclusion criteria: uncontrolled seizures, AAV hypersensitivity, other interventional trial, short life expectancy
- Clarification budget: `5`
- Max steps: `20`
- Special mechanic: Amendment A1 changes the CSS threshold at step 6
- Expected baseline score: `0.40`

## 🔧 Action Space

| Action | Purpose | Notes |
|---|---|---|
| `evaluate_criterion` | Judge one protocol criterion | Returns shaped reward for partial progress |
| `ask_clarification` | Reveal pending or estimated information | Costs a step; penalized if unnecessary |
| `enroll` | Final eligible decision | Correct decision receives episode bonus |
| `exclude` | Final ineligible decision | Correct decision receives episode bonus |
| `defer` | Weak final action | Accepted but penalized |

Example:

```json
{
  "action_type": "evaluate_criterion",
  "criterion_id": "INC-001",
  "evaluation": {
    "criterion_id": "INC-001",
    "verdict": "met",
    "reasoning": "Patient age is within the protocol range."
  },
  "confidence_score": 0.91
}
```

Full reference: [docs/action_space.md](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/action_space.md)

## 👁️ Observation Space

Observations include:

- patient demographics with computed BMI
- diagnosis metadata and ICD-10 code
- structured lab values with certainty tags
- current medication list
- trial protocol summary with inclusion and exclusion criteria
- step counters, action history, and system messages

Task 3 uses `trial_protocol_summary.amendment_active` and `amendment_description` to signal that eligibility logic has changed. Full reference: [docs/observation_space.md](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/observation_space.md)

## 💰 Reward Function

ClinicalTrialEnv emits dense shaped reward during the episode and task-specific grader scores at the end.

- Correct criterion evaluation: `+0.10` to `+0.15`
- Reasoning bonus: `+0.05`
- Repeat evaluation penalty: `-0.05`
- Unnecessary clarification: `-0.10`
- Correct final decision: `+0.40`
- Final defer penalty: `-0.20`
- Efficiency bonus: up to `+0.25`

Normalization clamps each returned reward into `[0.0, 1.0]`. Full design rationale: [docs/reward_design.md](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/reward_design.md)

## 🚀 Quick Start

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/clinicaltrial-env
cd clinicaltrial-env
cp .env.example .env
docker-compose up
```

### Running the Baseline

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"
python inference.py
```

### Expected Output

```text
[START] task=Single Criterion Screening env=clinicaltrial-env model=gpt-4o-mini
[STEP]  step=1 action={"action_type":"evaluate_criterion","criterion_id":"INC-001"} reward=0.12 done=false error=null
...
[END]   success=true steps=6 rewards=0.12,0.12,0.10,0.08,0.25,0.40
```

## 🐳 Docker

```bash
docker build -t clinicaltrial-env .
docker run -p 7860:7860 clinicaltrial-env
```

The Docker image is HF Spaces compatible and listens on port `7860`.

### Docker Verification Checklist

- Docker Desktop is running
- `docker info` succeeds
- `docker build -t clinicaltrial-env .` succeeds
- `docker run -p 7860:7860 clinicaltrial-env` starts without crashing
- `curl http://localhost:7860/health` returns HTTP `200`
- `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'` returns HTTP `200`

Tip: the repository now includes [.dockerignore](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/.dockerignore) so your local `.venv`, test cache, and docs do not bloat the Docker context.

## ✅ Pre-Submission Validation

```bash
openenv validate
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task1"}'
docker build -t clinicaltrial-env .
```

The target checks are:

- HF Space responds to `POST /reset` with HTTP `200`
- `docker build` succeeds
- `openenv validate` succeeds
- `POST /reset` supports an empty JSON body and defaults to `task1`, which matches the sample pre-validator behavior

### Final Pre-Submission Command List

Run these in order from the project root:

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
uv run --python .venv/Scripts/python.exe pytest tests -q
uv run --python .venv/Scripts/python.exe openenv validate
uv run --python .venv/Scripts/python.exe uvicorn server.main:app --host 0.0.0.0 --port 7860
```

In another terminal:

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
docker info
docker build -t clinicaltrial-env .
docker run -p 7860:7860 clinicaltrial-env
```

For Windows PowerShell, you can also use [validate.ps1](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/validate.ps1) once your local API is running.

## 📊 Baseline Scores

| Task | Difficulty | Model | Score | Steps Used |
|------|-----------|-------|-------|------------|
| task1 | Easy | gpt-4o-mini | 0.85 | 6/8 |
| task2 | Medium | gpt-4o-mini | 0.65 | 12/14 |
| task3 | Hard | gpt-4o-mini | 0.40 | 18/20 |

## 🏗️ Architecture

```text
FastAPI Routes
  -> ClinicalTrialEnv
     -> PatientGenerator + ProtocolLoader
     -> StateMachine + EpisodeManager
     -> RewardCalculator + Task Graders
```

Deep dive: [docs/architecture.md](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/architecture.md)

## 📁 Project Structure

```text
clinicaltrial-env/
├── inference.py
├── openenv.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── server/
├── protocols/
├── tests/
└── docs/
```

## 🔬 Technical Deep Dive

- [Architecture](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/architecture.md)
- [Action Space](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/action_space.md)
- [Observation Space](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/observation_space.md)
- [Reward Design](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/reward_design.md)
- [Synthetic Cohort Analysis](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/data_analysis.md)
- [HF Spaces Deployment Checklist](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/hf_spaces_deployment.md)

## 📈 Synthetic Data Quality

The environment does not rely on a fixed CSV dump. Instead it generates deterministic protocol-aware patient cohorts. A seeded analysis across the 50-case pool for each task shows:

- task1 eligible ratio: `0.58`
- task2 eligible ratio: `0.52`
- task3 eligible ratio: `0.48`
- task2 ANC pending rate: `0.36`
- task3 critical exclusion rate: `0.42`

This keeps the benchmark reproducible while still creating non-trivial decision pressure. Full details: [docs/data_analysis.md](/E:/Users/Abisheik/downloads/meta%20Hackathon/clinicaltrial-env/docs/data_analysis.md)

## 📄 License

MIT

## 🧑‍⚕️ Real-World Impact

Clinical trial recruitment remains a major bottleneck in drug development, with protocol screening consuming coordinator time and introducing variability in enrollment decisions. A benchmark like ClinicalTrialEnv can help train and evaluate AI agents that:

- standardize eligibility review
- surface safety-critical exclusions consistently
- reduce unnecessary clarification steps
- adapt to live protocol amendments

That makes the environment directly relevant to production-grade clinical operations tooling, not just benchmark experimentation.
