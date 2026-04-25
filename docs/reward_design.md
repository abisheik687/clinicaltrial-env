# Reward Design

ClinicalTrialEnv now uses a **verifier-centric reward** designed for Phase 1 training evidence.

## Core Rule

The reward is driven by one terminal question:

**Did the agent end with the correct safe final decision under the latest protocol state and revealed evidence?**

That means the reward is no longer a blend of correctness, verbosity, efficiency, and shaping bonuses. The environment now treats the final decision as the objective and diagnostic metrics as diagnostics only.

## Terminal Reward

- `+1.0`: correct safe final decision
- `0.0`: incorrect final decision or unresolved/defer outcome
- `-1.0`: unsafe enrollment

Unsafe enrollment is fully deterministic:

- `enroll` when any exclusion criterion is active
- `enroll` when any required inclusion criterion is definitively unmet

No model-judged safety logic is used.

## Intermediate Shaping Bonuses

The following shaping terms are active when `ENABLE_INTERMEDIATE_SHAPING=1` (the default set by `grpo_phase1.py`):

- `+0.3`: awarded once per episode when `EVALUATE_CRITERION` for `INC-003` is called after `amendment_detected=True` in hidden state (guarded by `shaping_bonus_amendment`)
- `+0.3`: awarded once per episode when `SCHEDULE_FOLLOWUP` is called (guarded by `shaping_bonus_followup`)
- `-0.05`: invalid or impossible action that still fits the schema

The two `+0.3` bonuses each fire at most once per episode. They reward the agent for recognising the amendment and for scheduling appropriate follow-up — two behaviours that are necessary for a correct safe final decision but are otherwise invisible to the terminal reward signal.

Examples of the `-0.05` penalty:

- trying to finalize before evaluating any criterion
- requesting clarification after the budget is exhausted

This keeps the rollout trainable without turning the reward into a proxy soup.

## Diagnostic Metrics

These metrics are logged in the reward payload but do **not** define success:

- `criterion_evaluation_accuracy`
- `clarification_efficiency`
- `unsafe_action_rate`
- `amendment_recovery_rate`

They are intended for evaluation plots and judge-facing analysis, not for changing the optimization target.

## Why This Design

This version follows the hackathon guidance more closely:

- start simple
- use objective checks
- make reward hacking harder
- show clear improvement in reward and behavior

The previous dense scheme made it too easy to earn reward without solving the real task. The current design makes the environment a cleaner fit for GRPO and for judging.
