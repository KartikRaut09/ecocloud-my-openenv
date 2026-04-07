---
title: EcoCloud OpenEnv
emoji: ":seedling:"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - carbon-aware-scheduling
  - infrastructure
  - gpu-operations
  - reinforcement-learning
license: mit
---

# EcoCloud OpenEnv

EcoCloud is a real-world OpenEnv environment for carbon-aware GPU infrastructure operations. The agent acts like a cloud platform operator managing distributed GPU fleets under conflicting objectives:

- reduce carbon emissions
- reduce energy cost
- avoid thermal incidents
- preserve workload reliability

This environment is built for the Meta x Scalar OpenEnv hackathon and is designed to evaluate agent behavior on a task that human operators actually perform: workload placement and infrastructure optimization across regions with different grid carbon intensity, energy prices, and cooling conditions.

## Why This Environment Matters

Modern AI systems consume large amounts of GPU capacity. In practice, operations teams constantly make decisions about:

- where to run flexible workloads
- when to power down idle servers
- how to shift jobs away from expensive or dirty regions
- how to protect production workloads during thermal or cost stress

EcoCloud turns that real operational workflow into a deterministic, reproducible OpenEnv benchmark.

## Hackathon Compliance Summary

This repository includes the required components for the hackathon:

- real-world task simulation
- typed Pydantic `Observation`, `Action`, and `Reward` models
- full `reset()`, `step()`, and `state()` environment flow
- `openenv.yaml` in the repository root
- 3 deterministic tasks with grader scores in `[0.0, 1.0]`
- shaped reward with partial progress and penalties
- root-level `inference.py`
- Hugging Face Space-compatible Docker deployment
- README with setup, task definitions, spaces, and baseline scores

## Environment Overview

EcoCloud simulates a GPU fleet spread across multiple regions. Each region has its own:

- carbon intensity
- electricity price
- ambient temperature

Each server has:

- GPU capacity
- power characteristics
- thermal behavior
- on/off state
- assigned workloads

Each workload has:

- GPU demand
- migration eligibility
- allowed regions
- priority
- placement target class

The environment is deterministic. There is no randomness in task generation, transitions, or grading.

## OpenEnv API

The service exposes the standard environment flow through HTTP:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /validate`
- `GET /tasks`

### `reset(task_id)`

Initializes a clean episode for the selected task and returns the first observation.

### `step(action)`

Applies a typed action and returns:

- next observation
- reward
- done flag
- grader and execution info

### `state()`

Returns the current internal environment state including current metrics and recent actions.

## Action Space

The agent can take one action per step.

### Supported actions

- `migrate_workload`
- `set_power_cap`
- `shutdown_server`
- `activate_server`
- `noop`

### Action schema

```json
{
  "action_type": "migrate_workload | set_power_cap | shutdown_server | activate_server | noop",
  "task_id": "string",
  "server_id": "string | null",
  "workload_id": "string | null",
  "target_server_id": "string | null",
  "power_cap_kw": "number | null",
  "rationale": "string"
}
```

### Action semantics

- `migrate_workload`: move a migratable workload to another active server in an allowed region
- `set_power_cap`: lower or raise the effective power cap of an active server
- `shutdown_server`: turn off an active server that has no assigned workloads
- `activate_server`: bring an inactive server online
- `noop`: do nothing for the current step

## Observation Space

Each observation contains task context plus a structured view of regions, servers, workloads, and aggregate metrics.

### Observation fields

- `task_id`
- `task_name`
- `difficulty`
- `objective`
- `step_number`
- `max_steps`
- `recent_event`
- `allowed_action_types`
- `regions`
- `servers`
- `workloads`
- `metrics`

### Metrics exposed to the agent

- `total_power_kw`
- `total_carbon_kgco2e`
- `total_cost_usd`
- `average_carbon_intensity`
- `max_temperature_c`
- `overloaded_gpu`
- `thermal_risk_index`
- `thermal_warning_servers`
- `hard_thermal_violations`
- `active_idle_servers`
- `migration_count`
- `reliability_score`
- `critical_violations`

## Reward Function

Rewards are dense and normalized to `[0.0, 1.0]`.

### Reward components

- `progress`: current task grader score
- `efficiency`: task-specific operational improvement
- `reliability`: service safety and thermal stability
- `penalty`: invalid actions, wasted steps, and hard failures
- `grader_score`: current deterministic task score

### Reward design goals

- provide useful signal before episode completion
- reward partial progress toward task success
- penalize invalid or obviously bad operational actions
- prevent exploit patterns such as repeated `noop` or destructive actions

## Tasks

The environment includes three graded tasks with increasing difficulty.

### Task 1: Idle Capacity Cleanup

- `task_id`: `task_1_idle_capacity_cleanup`
- difficulty: `easy`
- max steps: `5`
- objective: shut down idle GPU servers without disrupting any live workloads

This task measures whether the agent can identify safe shutdown opportunities and remove waste without touching protected workloads.

### Task 2: Carbon-Aware Rebalancing

- `task_id`: `task_2_carbon_aware_rebalancing`
- difficulty: `medium`
- max steps: `8`
- objective: move flexible workloads from high-carbon regions into lower-carbon capacity without overloading servers or moving pinned services

This task measures workload shifting under realistic placement constraints.

### Task 3: Resilient Multi-Objective Dispatch

- `task_id`: `task_3_resilient_multi_objective_dispatch`
- difficulty: `hard`
- max steps: `10`
- objective: reduce carbon, cost, and thermal risk during a heat wave while maintaining reliability for real-time workloads

This task measures true multi-objective operations under operational stress.

## Graders

Each task has a deterministic grader that returns a score in `[0.0, 1.0]`.

### Grader properties

- deterministic
- reproducible
- bounded score range
- partial credit
- hard-failure handling

### Task 1 grading

Scores:

- idle server cleanup
- protected workload safety
- carbon improvement

### Task 2 grading

Scores:

- fraction of flexible workload shifted to low-carbon regions
- capacity safety
- protected workload preservation
- carbon improvement

### Task 3 grading

Scores:

- carbon improvement
- cost improvement
- thermal score
- reliability score
- hotspot relief

Critical-service violations and hard thermal failures cap the final score.

## Project Structure

```text
app.py
env/
  __init__.py
  baseline_policy.py
  environment.py
  models.py
graders/
  __init__.py
  graders.py
tasks/
  __init__.py
  task_data.py
tests/
  __init__.py
  test_env.py
inference.py
openenv.yaml
preflight_check.py
Dockerfile
requirements.txt
README.md
```

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the environment server

```powershell
uvicorn app:app --host 0.0.0.0 --port 7860
```

### 4. Run tests

```powershell
python -m unittest
```

### 5. Run the local preflight

```powershell
python preflight_check.py
```

## Using The Environment

### List tasks

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:7860/tasks"
```

### Reset a task

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:7860/reset" -ContentType "application/json" -Body '{"task_id":"task_1_idle_capacity_cleanup"}'
```

### Take a step

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:7860/step" -ContentType "application/json" -Body '{"action_type":"noop","task_id":"task_1_idle_capacity_cleanup","rationale":"example"}'
```

### Inspect state

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:7860/state"
```

## Baseline Inference

The required baseline script is [inference.py](./inference.py). It runs all tasks and emits structured logs using the required `[START]`, `[STEP]`, and `[END]` prefixes.

### Required environment variables

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `ENV_BASE_URL` defaults to `http://localhost:7860`
- `OPENAI_API_KEY` may also be used for OpenAI-compatible routing

### Example

```powershell
$env:ENV_BASE_URL="http://localhost:7860"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
$env:HF_TOKEN="your_token_here"
python inference.py
```

### Baseline behavior

The baseline uses a deterministic heuristic policy and optionally routes action verification through the OpenAI client when credentials are configured. This keeps the run reproducible while still satisfying the hackathon requirement that LLM calls use the OpenAI client.

## Reproducible Baseline Scores

Measured locally with the included baseline:

| Task | Difficulty | Score | Passed |
|---|---|---:|---:|
| `task_1_idle_capacity_cleanup` | easy | `0.9191` | `true` |
| `task_2_carbon_aware_rebalancing` | medium | `0.8760` | `true` |
| `task_3_resilient_multi_objective_dispatch` | hard | `0.7721` | `true` |

## Docker

### Build

```powershell
docker build -t ecocloud-openenv .
```

### Run

```powershell
docker run -p 7860:7860 ecocloud-openenv
```

## Hugging Face Spaces Deployment

This repository is prepared for a Docker-based Hugging Face Space.

### Deployment steps

1. Create a new Space with `Docker` as the SDK.
2. Push this repository to the Space.
3. Add the following Space variables and secrets:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. Wait for the build to finish.
5. Verify:
   - `GET /health`
   - `GET /tasks`
   - `POST /reset`
   - `GET /state`

## Validation Checklist

Before submission, verify all of the following:

- `python -m unittest` passes
- `python preflight_check.py` passes
- `python inference.py` completes successfully
- scores stay within `[0.0, 1.0]`
- HF Space builds without errors
- deployed Space returns `200`
- live `/reset` works
- `openenv.yaml` exists at repo root
- `inference.py` exists at repo root
- required env vars are configured in the deployment environment

## Known Remaining Risk

The environment is locally complete and validated. The remaining submission risk is external:

- final Hugging Face Space build success
- exact compliance with the official hackathon sample log formatting
- official validator compatibility if the organizers use a stricter OpenEnv validator than the local `/validate` endpoint

## License

MIT
