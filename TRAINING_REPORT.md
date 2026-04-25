# ClinicalTrialEnv Training Report

## 1. Problem

Clinical trial coordination is a high-stakes professional workflow: a coordinator must combine protocol rules, patient records, lab evidence, medication conflicts, amendments, and safety events before taking action. The capability gap is that an LLM must maintain state over a changing workflow instead of answering a static eligibility question.

RL is needed because the useful behavior is sequential: inspect evidence, re-check criteria after amendment, make a safe enrollment decision, schedule the follow-up, and escalate the seizure-symptom safety event. A single prompt answer cannot prove the agent learned that workflow.

## 2. Environment

The Task 3 environment gives the agent a synthetic, seed-deterministic patient record and a protocol with inclusion/exclusion criteria. The state changes over time through a visible protocol amendment, a follow-up scheduling phase, and a deterministic safety event before the visit.

The agent can inspect patient/protocol state, evaluate criteria, request clarification, enroll/exclude/defer, schedule a follow-up day, and handle the safety event. The verifier checks whether the final workflow is safe under the latest protocol state.

## 3. Training Setup

- Target task: `task3` only.
- Model path in current eval artifact: `artifacts/phase1_grpo/model`.
- Training loop: GRPO through `training/grpo_phase1.py`, with the FastAPI environment served locally and queried through `/reset` and `/step`.
- Anchor gate: the known seed-44 correct trajectory must pass before training starts.
- Staged execution: short signal pass first, then longer run only if validation gates pass.
- Agent interacts with environment -> receives reward -> updates policy.
- GRPO advantage stats were not available in the current log artifact.

## 4. Baseline vs Trained

Validation status: **NOT PASSED**. The final validation gates did not pass, so this report is intentionally conservative and does not claim a winning RL improvement.

### Untrained Model

| Step | Action | Detail |
| --- | --- | --- |
| 1 | `evaluate_criterion` | criterion_id=INC-001 |
| 2 | `evaluate_criterion` | criterion_id=INC-002 |
| 3 | `ask_clarification` | clarification_target=INC-003 |
| 4 | `evaluate_criterion` | criterion_id=INC-003 |
| 5 | `evaluate_criterion` | criterion_id=INC-004 |
| 6 | `evaluate_criterion` | criterion_id=INC-005 |
| 7 | `evaluate_criterion` | criterion_id=INC-006 |
| 8 | `ask_clarification` | clarification_target=EXC-001 |
| 9 | `evaluate_criterion` | criterion_id=EXC-001 |
| 10 | `ask_clarification` | clarification_target=EXC-002 |
| 11 | `evaluate_criterion` | criterion_id=EXC-002 |
| 12 | `evaluate_criterion` | criterion_id=EXC-003 |
| 13 | `evaluate_criterion` | criterion_id=EXC-004 |
| 14 | `exclude` | - |

### RL-Trained Model

| Step | Action | Detail |
| --- | --- | --- |
| 1 | `evaluate_criterion` | criterion_id=INC-001 |
| 2 | `evaluate_criterion` | criterion_id=INC-002 |
| 3 | `ask_clarification` | clarification_target=INC-003 |
| 4 | `evaluate_criterion` | criterion_id=INC-003 |
| 5 | `evaluate_criterion` | criterion_id=INC-004 |
| 6 | `evaluate_criterion` | criterion_id=INC-005 |
| 7 | `evaluate_criterion` | criterion_id=INC-006 |
| 8 | `ask_clarification` | clarification_target=EXC-001 |
| 9 | `evaluate_criterion` | criterion_id=EXC-001 |
| 10 | `ask_clarification` | clarification_target=EXC-002 |
| 11 | `evaluate_criterion` | criterion_id=EXC-002 |
| 12 | `evaluate_criterion` | criterion_id=EXC-003 |
| 13 | `evaluate_criterion` | criterion_id=EXC-004 |
| 14 | `exclude` | - |

Structural behavior difference: `False`.

## 5. Reward Curve

![Training reward curve](artifacts/plots/training_reward_curve.png)

Reward increases from -1.00 -> -1.00 over training steps.

![Held-out comparison](artifacts/plots/heldout_base_vs_trained.png)

## 6. Quantitative Results

| Metric | Untrained Model | RL-Trained Model |
| --- | ---: | ---: |
| Success rate | 0.3333 | 0.3333 |
| Unsafe rate | 0.0000 | 0.0000 |
| Mean reward | 0.6667 | 0.6667 |
| Amendment recovery | 1.0000 | 1.0000 |
| Fallback used rate | 0.0000 | 0.0000 |

Validation gates:

| Gate | Value |
| --- | --- |
| Trajectory reward std | `0.0` |
| Episode done rate | `0.0` |
| Valid trajectory rate | `0.0` |
| Max clipped ratio | `1.0` |
| Reward delta | `0.0` |
| Hard failure same success/reward/behavior | `True` |

## 7. Key Learning

The desired learned behavior is not medical knowledge memorization. It is operational discipline: follow the current protocol version, avoid unsafe enrollment, make the follow-up scheduling decision, and escalate the safety event.

If the validation status above is `NOT PASSED`, the correct interpretation is that the environment and pipeline are ready, but the current run did not yet produce judge-proof learning evidence. The project should be submitted honestly or rerun on HF GPU until the gates pass.

## 8. Why This Matters

Clinical trials fail operationally when teams miss protocol changes, schedule outside allowed windows, or underreact to safety symptoms. This environment turns those real workflow risks into a bounded OpenEnv training loop for Theme 3.1 Professional Tasks, with secondary long-horizon planning pressure from the amendment -> enrollment -> scheduling -> safety sequence.
