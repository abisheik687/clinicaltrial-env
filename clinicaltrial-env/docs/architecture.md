# Architecture

ClinicalTrialEnv is built as a deterministic FastAPI service around an in-memory RL environment. The design goal is hackathon reliability first: clear typed boundaries, reproducible data generation, auditable graders, and no external infrastructure.

## System Design Rationale

- The environment exposes an HTTP API because OpenEnv validators and Hugging Face Spaces expect a simple deployable service.
- Pydantic v2 models define the observation, action, reward, and state contracts so both the API and internal environment logic stay aligned.
- Deterministic synthetic data makes baseline benchmarking reproducible and avoids leaking hidden labels into observations.
- Reward shaping is separated from episode grading so RL training signals remain dense while final task scoring remains transparent and task-specific.

## Component Interaction

```text
+-------------------+       +---------------------+
| Client / Agent    | ----> | FastAPI Routes      |
+-------------------+       +----------+----------+
                                      |
                                      v
                           +----------+----------+
                           | ClinicalTrialEnv    |
                           | reset / step / state|
                           +----+------+---------+
                                |      |
                +---------------+      +----------------+
                v                                       v
      +---------+---------+                    +--------+---------+
      | PatientGenerator  |                    | StateMachine     |
      | ProtocolLoader    |                    | EpisodeManager   |
      +---------+---------+                    +--------+---------+
                |                                       |
                v                                       v
      +---------+---------+                    +--------+---------+
      | Hidden Ground     |                    | RewardCalculator |
      | Truth + Metadata  |                    | Task Graders     |
      +-------------------+                    +------------------+
```

## Episode Lifecycle

```text
RESET
  |
  v
Session created -> Initial observation returned
  |
  v
STEP(action)
  |
  +--> Validate action
  +--> Apply state transition
  +--> Apply clarification or amendment side effects
  +--> Compute shaped reward
  +--> Check termination
  |
  v
Return observation, reward, done, info
  |
  +--> if done: episode closed
  +--> else: continue
```

## State Machine

```text
START
  |
  v
Active screening
  |
  +--> evaluate_criterion -> store verdict
  +--> ask_clarification -> reveal hidden value, consume budget
  +--> task3 step 6 -> inject protocol amendment
  +--> enroll/exclude/defer -> terminate
  +--> max_steps reached -> terminate
```

## Session Management Design

- Sessions are stored in an in-memory `dict[str, TrialState]`.
- Access timestamps are tracked separately by `EpisodeManager`.
- Expired sessions are cleaned opportunistically on `reset`, `step`, and `state`.
- This keeps the Docker Space lightweight while still preventing unbounded memory growth in long-running containers.

## Why FastAPI

- Native Pydantic integration reduces schema drift.
- Good performance on CPU-only containers.
- Minimal code for validation and OpenAPI-compatible routes.
- TestClient support makes integration testing straightforward.

## Scalability Considerations

- Single-process deployment is enough for the hackathon target, but the environment class is stateless outside the in-memory session map and can be swapped for Redis-backed state if horizontal scaling is needed later.
- Protocol YAML files are cached after first load.
- Synthetic patient generation is cheap and CPU-friendly, so the service stays within the `vcpu=2`, `8GB RAM` constraint.

