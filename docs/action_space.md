# Action Space

ClinicalTrialEnv uses a discrete action family with structured payloads.

## Action Types

### `evaluate_criterion`

Use this to judge whether one inclusion or exclusion criterion is met.

```json
{
  "action_type": "evaluate_criterion",
  "criterion_id": "INC-001",
  "evaluation": {
    "criterion_id": "INC-001",
    "verdict": "met",
    "reasoning": "Patient age is 54, which satisfies the protocol range."
  },
  "confidence_score": 0.93
}
```

### `ask_clarification`

Use this when a criterion is backed by a `pending` or `estimated` value and clarification is available.

```json
{
  "action_type": "ask_clarification",
  "clarification_target": "INC-003",
  "confidence_score": 0.52
}
```

### `enroll`

Submit a final eligibility decision when you believe the patient is eligible.

```json
{
  "action_type": "enroll",
  "final_decision_reason": "All inclusion criteria are satisfied and no exclusion criteria are triggered.",
  "confidence_score": 0.97
}
```

### `exclude`

Submit a final decision when any required inclusion criterion fails or an exclusion criterion is triggered.

```json
{
  "action_type": "exclude",
  "final_decision_reason": "Patient has severe renal impairment and is not safe for enrollment.",
  "confidence_score": 0.95
}
```

### `defer`

Weak final action that ends the episode and is penalized.

```json
{
  "action_type": "defer",
  "confidence_score": 0.10
}
```

## Valid States

- `evaluate_criterion` is valid whenever the referenced criterion exists.
- `ask_clarification` is valid only if clarification budget remains.
- `enroll` and `exclude` become meaningful once at least one criterion has been evaluated; early submission is allowed but penalized if fewer than 50% of criteria were reviewed.
- `defer` is always accepted but intentionally discouraged by the reward and Task 3 grader.

## Invalid Action Handling

- Unknown `criterion_id` or `clarification_target` returns HTTP `400`.
- Clarification after the budget is exhausted returns HTTP `400`.
- Any action after episode termination returns HTTP `400`.
- Payload shape errors return HTTP `422`.

## Strategy Guide

- Task 1: evaluate all five criteria in order, then finalize.
- Task 2: resolve `INC-004` only when ANC is uncertain; inspect corticosteroids carefully for `EXC-004`.
- Task 3: save clarification budget for `INC-003`, `EXC-001`, and `EXC-002`, then re-check `INC-003` after the amendment appears at step 6.

