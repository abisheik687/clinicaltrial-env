# Synthetic Cohort Analysis

ClinicalTrialEnv uses deterministic synthetic patients rather than a static CSV dataset. That means the quality bar is not just “does it run,” but whether the generated cohorts are balanced, clinically plausible, and difficult in the right ways.

## Sampling Method

- Each task uses a deterministic pool of 50 seed-addressable cases.
- Cases are generated with seeded NumPy distributions and protocol-aware overrides.
- Ground-truth eligibility is hidden from the public observation but stored internally for graders and reward shaping.

## Cohort Summary

The following measurements were generated from seeds `0..49` using the current generator implementation.

| Task | Eligible Ratio | Mean Age | Mean Weight |
|---|---:|---:|---:|
| task1 | 0.58 | 54.84 | 79.96 kg |
| task2 | 0.52 | 55.60 | 81.49 kg |
| task3 | 0.48 | 18.72 | 28.51 kg |

## Task-Specific Signals

### Task 1

- Mean systolic blood pressure: `152.49 mmHg`
- Mean eGFR: `67.34 mL/min/1.73m2`
- Interpretation: the cohort centers around realistic hypertensive screening values while still generating renal edge cases and medication exclusions.

### Task 2

- ANC pending rate: `0.36`
- Drug interaction case rate: `0.66`
- Interpretation: a substantial fraction of oncology episodes require either marrow clarification or corticosteroid reasoning, which prevents the task from collapsing into simple rule matching.

### Task 3

- CSS pending rate: `1.00`
- Critical exclusion rate: `0.42`
- Interpretation: the hard task reliably forces ambiguity handling and makes unsafe enrollment decisions meaningfully costly.

## Why This Matters for Evaluation

- The eligible ratio stays near the intended “roughly 60% but not trivial” target.
- Task difficulty increases by information uncertainty, protocol complexity, and amendment sensitivity rather than by arbitrary noise.
- Seeds are stable, so baseline runs and grader checks remain reproducible across machines.

