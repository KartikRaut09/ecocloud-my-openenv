<p align="center">
  <h1 align="center">EcoCloud OpenEnv</h1>
  <p align="center">
    A real-world OpenEnv benchmark for carbon-aware GPU infrastructure operations.
  </p>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white">
  <img alt="OpenEnv" src="https://img.shields.io/badge/OpenEnv-Compatible-111111">
  <img alt="HF Spaces" src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E">
</p>

<p align="center">
  EcoCloud simulates the real operational decisions made by infra and platform teams:
  workload placement, carbon-aware scheduling, power-cap management, idle capacity cleanup,
  and reliability-preserving dispatch under thermal and cost pressure.
</p>

---

## What Is EcoCloud?

EcoCloud is a deterministic OpenEnv environment for evaluating AI agents on **real-world GPU fleet operations**.

The environment models a distributed GPU infrastructure across multiple regions with different:

- carbon intensity
- energy prices
- ambient temperature
- thermal stress conditions

An agent interacts through the standard environment loop:

- `reset()`
- `step(action)`
- `state()`

The objective is to optimize infrastructure behavior while balancing:

- carbon
- cost
- thermal safety
- workload reliability

---

## Why This Benchmark Matters

Real AI infrastructure is expensive, power-hungry, and operationally constrained.

In practice, platform and SRE teams must continuously decide:

- which workloads can be moved to cleaner regions
- when idle servers should be powered down
- how to reduce energy spend safely
- how to avoid overheating or reliability regressions

EcoCloud turns that real workflow into a reproducible benchmark for agent evaluation.

---

## Hackathon Compliance

This repository is structured to satisfy the Meta x Scalar OpenEnv hackathon requirements:

- real-world task simulation
- typed Pydantic models for observations, actions, and rewards
- full environment flow through `reset`, `step`, and `state`
- root-level `openenv.yaml`
- 3 deterministic tasks with graders
- bounded reward and score outputs in `[0.0, 1.0]`
- root-level `inference.py`
- Docker-ready deployment for Hugging Face Spaces
- documented setup, scoring, and deployment flow

---

## Environment Design

### Core Entities

**Regions**
- carbon intensity
- electricity price
- ambient temperature

**Servers**
- GPU capacity
- active/off state
- power caps
- power draw
- thermal behavior
- workload assignments

**Workloads**
- GPU demand
- priority
- migration policy
- allowed regions
- target behavior class

### Operational Objective

The agent must find strong operational tradeoffs across:

- lower emissions
- lower cost
- lower thermal risk
- higher reliability

This makes the benchmark realistic and multi-objective rather than simplistic.

---

## API Surface

The environment is exposed as a FastAPI service.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | root metadata |
| `/health` | `GET` | health check |
| `/tasks` | `GET` | list all tasks |
| `/reset` | `POST` | start a task episode |
| `/step` | `POST` | apply one action |
| `/state` | `GET` | inspect current environment state |
| `/validate` | `GET` | validation metadata |
| `/grade/{task_id}` | `GET` | final grader output |

---

## Action Space

The agent can take one of the following actions:

| Action | Description |
|---|---|
| `migrate_workload` | move a workload to another active server |
| `set_power_cap` | adjust server power cap |
| `shutdown_server` | turn off an idle server |
| `activate_server` | bring an inactive server online |
| `noop` | take no action |

### Action Schema

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

---

## Observation Space

Each observation returned by the environment contains structured state for the current task and infrastructure snapshot.

### Observation Fields

| Field | Type | Description |
|---|---|---|
| `task_id` | string | active task identifier |
| `task_name` | string | human-readable task name |
| `difficulty` | string | task difficulty level |
| `objective` | string | task objective |
| `step_number` | integer | current step in the episode |
| `max_steps` | integer | maximum allowed steps |
| `recent_event` | string | latest environment event |
| `allowed_action_types` | list | actions available to the agent |
| `regions` | list | state of each region |
| `servers` | list | state of each server |
| `workloads` | list | state of each workload |
| `metrics` | object | aggregate operational metrics |

### Metrics Included

| Metric | Description |
|---|---|
| `total_power_kw` | total power draw across the fleet |
| `total_carbon_kgco2e` | total carbon emissions |
| `total_cost_usd` | energy cost |
| `average_carbon_intensity` | weighted average carbon intensity |
| `max_temperature_c` | highest server temperature |
| `overloaded_gpu` | overloaded GPU amount |
| `thermal_risk_index` | normalized thermal risk |
| `thermal_warning_servers` | servers near unsafe temperature |
| `hard_thermal_violations` | servers above hard thermal limits |
| `active_idle_servers` | active servers with no useful load |
| `migration_count` | number of workload migrations |
| `reliability_score` | workload safety and stability score |
| `critical_violations` | number of critical-service violations |

---

## Reward Function

Rewards are dense and normalized to `[0.0, 1.0]`.

### Reward Components

- `progress`
- `efficiency`
- `reliability`
- `penalty`
- `grader_score`

### Reward Intent

The reward system is designed to:

- provide useful signal throughout the episode
- reward partial operational progress
- penalize invalid or destructive actions
- discourage wasted steps and bad control loops

---

## Tasks

EcoCloud includes three deterministic tasks with increasing difficulty.

### 1. Idle Capacity Cleanup

- **Task ID:** `task_1_idle_capacity_cleanup`
- **Difficulty:** Easy
- **Max Steps:** 5

Objective:
Safely shut down idle GPU servers without disrupting live workloads.

This task tests:
- safe server shutdown logic
- elimination of wasted capacity
- service-preserving decision making

---

### 2. Carbon-Aware Rebalancing

- **Task ID:** `task_2_carbon_aware_rebalancing`
- **Difficulty:** Medium
- **Max Steps:** 8

Objective:
Move flexible workloads away from high-carbon regions into cleaner capacity while respecting placement constraints.

This task tests:
- constrained workload migration
- carbon-aware dispatch
- workload protection rules

---

### 3. Resilient Multi-Objective Dispatch

- **Task ID:** `task_3_resilient_multi_objective_dispatch`
- **Difficulty:** Hard
- **Max Steps:** 10

Objective:
Reduce carbon, cost, and thermal risk during infrastructure stress while preserving reliability for real-time services.

This task tests:
- multi-objective optimization
- hotspot mitigation
- thermal-aware infrastructure control
- reliability-preserving workload placement

---

## Graders

Each task has a deterministic grader with score output in `[0.0, 1.0]`.

### Grading Principles

- deterministic
- reproducible
- bounded
- partial-credit based
- hard-failure aware

### Grader Focus By Task

| Task | What Gets Measured |
|---|---|
| Task 1 | idle cleanup, safety, carbon improvement |
| Task 2 | low-carbon shift ratio, capacity safety, protected workloads |
| Task 3 | carbon gain, cost gain, thermal stability, hotspot relief, reliability |

Hard thermal failures and critical workload violations cap the final score.

---

## Baseline Inference

The repository includes a root-level `inference.py` that:

- runs all tasks
- uses the OpenAI client for LLM calls
- reads required environment variables
- emits structured `[START]`, `[STEP]`, and `[END]` logs
- produces reproducible scores

### Required Environment Variables

| Variable | Purpose |
|---|---|
| `API_BASE_URL` | OpenAI-compatible endpoint |
| `MODEL_NAME` | model identifier |
| `HF_TOKEN` | Hugging Face / API token |
| `LOCAL_IMAGE_NAME` | optional for docker-image-based flows |

---

## Baseline Scores

Measured with the included deterministic baseline:

| Task | Difficulty | Score | Passed |
|---|---|---:|---:|
| `task_1_idle_capacity_cleanup` | Easy | `0.9191` | `true` |
| `task_2_carbon_aware_rebalancing` | Medium | `0.8760` | `true` |
| `task_3_resilient_multi_objective_dispatch` | Hard | `0.7721` | `true` |

---

## Project Structure

```text
.
├── app.py
├── inference.py
├── openenv.yaml
├── preflight_check.py
├── Dockerfile
├── requirements.txt
├── env/
│   ├── __init__.py
│   ├── baseline_policy.py
│   ├── environment.py
│   └── models.py
├── graders/
│   ├── __init__.py
│   └── graders.py
├── tasks/
│   ├── __init__.py
│   └── task_data.py
└── tests/
    ├── __init__.py
    └── test_env.py
```

---

## Local Setup

### Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Start the server

```powershell
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run tests

```powershell
python -m unittest
```

### Run local preflight

```powershell
python preflight_check.py
```

---

## API Usage

### List tasks

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:7860/tasks"
```

### Reset a task

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:7860/reset" -ContentType "application/json" -Body '{"task_id":"task_1_idle_capacity_cleanup"}'
```

### Apply an action

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:7860/step" -ContentType "application/json" -Body '{"action_type":"noop","task_id":"task_1_idle_capacity_cleanup","rationale":"example"}'
```

### Inspect state

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:7860/state"
```

---

## Run The Baseline

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
$env:HF_TOKEN="your_token_here"
python inference.py
```

---

## Docker

### Build

```powershell
docker build -t ecocloud-openenv .
```

### Run

```powershell
docker run -p 7860:7860 ecocloud-openenv
```

---

## Hugging Face Spaces

This repository is prepared for Docker-based Hugging Face Spaces deployment.

### Deployment Checklist

- Space created with **Docker** SDK
- root endpoint returns `200`
- `/reset` works
- `/step` works
- `API_BASE_URL` is configured
- `MODEL_NAME` is configured
- `HF_TOKEN` is configured
- live baseline run completes successfully

---

## Validation Checklist

Before submission, verify:

- `python -m unittest` passes
- `python preflight_check.py` passes
- `python inference.py` completes without error
- all scores remain in `[0.0, 1.0]`
- HF Space is live
- live `/reset` works
- Docker build succeeds
- `openenv validate` passes if required in your environment

---

## Final Notes

EcoCloud is intentionally deterministic to support:

- fair benchmarking
- reproducible grading
- stable comparison across agent runs

It is designed not just as a hackathon deliverable, but as a useful benchmark for infrastructure-oriented agent evaluation.

---

## License

MIT
