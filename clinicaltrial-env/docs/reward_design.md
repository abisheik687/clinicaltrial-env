# Reward Design

ClinicalTrialEnv separates shaped RL reward from deterministic task grading.

## Philosophy

Binary rewards are weak for long, clinical workflows. Coordinators do useful work before the final enroll or exclude decision: reviewing criteria, requesting the right missing information, and avoiding unsafe shortcuts. Partial rewards let an RL agent learn those intermediate behaviors directly.

## Reward Components

### Step Rewards

- Correct criterion evaluation: `+0.10` to `+0.15` depending on task difficulty
- Coherent reasoning string longer than 20 characters: `+0.05`
- Re-evaluating the same criterion without clarification: `-0.05`
- Asking for clarification on a confirmed value: `-0.10`
- Asking for clarification on pending or estimated data: `0.00`

### Episode Rewards

- Correct final decision: `+0.40`
- Wrong enrollment with a critical exclusion: `-0.30`
- Final `defer`: `-0.20`
- Finish within 60% of the step budget: `+0.10`
- Unused steps: `+0.05` each, capped at `+0.15`

### Behavioral Penalties

- Same action repeated 3 or more times: `-0.30`
- Final decision before evaluating 50% of criteria: `-0.15`
- Step overflow guard: `-0.05` per extra step

## Mathematical Form

```text
raw_sum = step_reward + final_bonus + efficiency_bonus - penalties
normalized_score = clamp(raw_sum / max_possible_reward, 0.0, 1.0)
```

Task-specific normalizers:

- Task 1: `1.40`
- Task 2: `1.85`
- Task 3: `2.50`

## Good vs Bad Reward Traces

Good agent:

```text
step1 evaluate correctly      +0.10
step2 evaluate correctly      +0.10
step3 clarification neutral   +0.00
step4 evaluate correctly      +0.15
final correct decision        +0.40
efficiency bonus              +0.15
```

Bad agent:

```text
step1 wrong evaluation        +0.00
step2 unnecessary clarify     -0.10
step3 repeat same criterion   -0.05
step4 loop detection          -0.30
final defer                   -0.20
```

## Dual Signal: Accuracy Plus Efficiency

This reward landscape creates real tradeoffs:

- rushing to a decision can earn efficiency points but risks a premature-decision penalty
- asking for clarification is neutral when justified but costly in step budget
- repeated evaluation is discouraged, but amendment-driven re-evaluation remains strategically important in Task 3

That makes the environment suitable for RL, imitation learning, and agentic policy optimization.

## Grader vs Reward

- Reward guides learning on each environment step.
- Graders score the finished episode according to the task rubric.
- Reward transparency is preserved through `partial_credit`, while graders keep benchmark reporting deterministic and comparable across runs.

