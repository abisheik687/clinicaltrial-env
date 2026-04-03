# Observation Space

`PatientObservation` is a structured dictionary represented by typed Pydantic models.

## Fields

| Field | Type | Description |
|---|---|---|
| `patient_id` | `str` | UUID identifier for the synthetic patient. |
| `demographics.age` | `int` | Age in years. |
| `demographics.sex` | `Literal["M","F","Other"]` | Administrative sex label. |
| `demographics.weight_kg` | `float` | Weight in kilograms. |
| `demographics.height_cm` | `float` | Height in centimeters. |
| `demographics.bmi` | `float` | Computed body mass index. |
| `diagnosis.primary_condition` | `str` | Primary diagnosis summary. |
| `diagnosis.icd10_code` | `str` | ICD-10 code aligned to the task protocol. |
| `diagnosis.disease_stage` | `Optional[str]` | Stage or disease descriptor when relevant. |
| `diagnosis.diagnosis_date` | `str` | ISO date string. |
| `lab_values` | `dict[str, LabValue]` | Named lab and assessment values. |
| `current_medications` | `list[Medication]` | Medication list used for interaction checks. |
| `trial_protocol_summary` | `TrialProtocolSummary` | Visible protocol summary for the current task. |
| `step_number` | `int` | Current episode step. |
| `steps_remaining` | `int` | Remaining step budget. |
| `previous_actions` | `list[str]` | Action history for context. |
| `info_message` | `Optional[str]` | System message, clarification result, or amendment notice. |

## Certainty Levels

- `confirmed`: authoritative visible value. Asking for clarification is penalized.
- `pending`: the actual value exists internally but is withheld until clarification.
- `estimated`: a noisy or provisional signal. Clarification may improve confidence or reveal risk.

## How Observation Changes Across Steps

- `step_number` increments on every action.
- `steps_remaining` decreases on every action.
- `previous_actions` appends the latest action type.
- `info_message` changes when the system stores an evaluation, reveals clarification data, or injects the Task 3 amendment.
- Clarification can update a lab value from `pending` to `confirmed`.

## Amendment Detection

Task 3 injects Amendment A1 at step 6.

Agents can detect it by comparing the observation between steps 5 and 6:

- `trial_protocol_summary.amendment_active` flips from `false` to `true`
- `trial_protocol_summary.amendment_description` becomes non-null
- the visible description for `INC-003` changes from `12-36` to `10-36`
- `info_message` explicitly instructs the agent to re-check `INC-003`

## Example Interpretation

- A `css_score` of `10.8` with certainty `pending` is not enough to finalize Task 3 before clarification or amendment review.
- A corticosteroid medication with `is_contraindicated=null` means the agent must reason from name and dose, not from an exposed binary label.

