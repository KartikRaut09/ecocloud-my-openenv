# EcoCloud OpenEnv

<p align="center">
  <strong>A real-world OpenEnv benchmark for carbon-aware GPU infrastructure operations.</strong>
</p>

<p align="center">
  EcoCloud simulates the kind of decisions real platform, infra, and SRE teams make every day:
  workload placement, server activation and shutdown, power-cap tuning, and multi-region optimization
  under carbon, cost, thermal, and reliability constraints.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white">
  <img alt="OpenEnv" src="https://img.shields.io/badge/OpenEnv-compatible-black">
  <img alt="HF Spaces" src="https://img.shields.io/badge/HuggingFace-Spaces-yellow">
</p>

---

## Overview

EcoCloud is a deterministic OpenEnv environment built for evaluating AI agents on **real-world GPU fleet operations**.

The environment models a distributed GPU infrastructure across multiple regions with different:

- carbon intensity
- electricity prices
- ambient temperature
- thermal risk profiles

An agent interacts through the standard environment loop:

- `reset()`
- `step(action)`
- `state()`

The goal is to optimize infrastructure behavior without breaking reliability.

---

## Why This Matters

Modern AI systems are expensive to run and environmentally costly. Infra teams routinely need to decide:

- which workloads can be shifted to cleaner regions
- when idle servers should be shut down
- how to avoid overheating under regional heat stress
- how to reduce energy spend without hurting production services

EcoCloud turns that workflow into a reproducible benchmark for agent evaluation.

---

## Hackathon Fit

This project is designed to match the Meta x Scalar OpenEnv hackathon requirements:

- real-world task simulation
- typed Pydantic models for observations, actions, and rewards
- full environment API with `reset`, `step`, and `state`
- root-level `openenv.yaml`
- 3 deterministic tasks with graders
- bounded reward and task scores in `[0.0, 1.0]`
- baseline `inference.py`
- Docker-based Hugging Face Spaces deployment
- documented setup, usage, and baseline results

---

## Environment Design

### Core Entities

**Regions**
- carbon intensity
- energy price
- ambient temperature

**Servers**
- GPU capacity
- current status
- power cap
- thermal profile
- assigned workloads

**Workloads**
- GPU demand
- priority
- migration permission
- allowed regions
- target class

### Agent Responsibilities

The agent must balance:

- carbon reduction
- cost reduction
- thermal stability
- reliability preservation

This creates realistic tradeoffs rather than single-metric optimization.

---

## API Surface

The environment is exposed via FastAPI and follows the OpenEnv-style interaction loop.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | service metadata |
| `/health` | `GET` | health check |
| `/tasks` | `GET` | list task summaries |
| `/reset` | `POST` | start a task episode |
| `/step` | `POST` | apply one action |
| `/state` | `GET` | inspect current state |
| `/validate` | `GET` | environment validation metadata |
| `/grade/{task_id}` | `GET` | final grader output for active task |

---

## Action Space

The agent can take one of the following actions per step:

| Action | Description |
|---|---|
| `migrate_workload` | move a workload to another active server |
| `set_power_cap` | tune server power usage |
| `shutdown_server` | turn off an idle server |
| `activate_server` | bring an inactive server online |
| `noop` | intentionally do nothing |

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


Observation Space
Each observation includes:

task metadata
step metadata
region states
server states
workload states
aggregate operational metrics
Key Metrics
Metric	Meaning
total_power_kw	total energy draw
total_carbon_kgco2e	carbon footprint
total_cost_usd	energy cost
average_carbon_intensity	weighted carbon intensity
max_temperature_c	hottest server temperature
overloaded_gpu	GPU overload amount
thermal_risk_index	normalized thermal risk
active_idle_servers	wasted active capacity
migration_count	workload movement count
reliability_score	workload safety score
critical_violations	critical failure count
Reward Function
Rewards are dense and normalized to [0.0, 1.0].

Reward Components
progress
efficiency
reliability
penalty
grader_score
Reward Behavior
The reward is designed to:

provide useful feedback before episode end
give partial credit for operational progress
penalize bad actions and wasted steps
discourage destructive or invalid behavior
This makes the benchmark useful for both evaluation and learning-based agents.

Tasks
EcoCloud includes three deterministic tasks with increasing difficulty.

Task 1: Idle Capacity Cleanup
Task ID: task_1_idle_capacity_cleanup
Difficulty: Easy
Max Steps: 5
Objective:
Safely shut down idle GPU servers without impacting active workloads.

What it tests:

identifying safe shutdown opportunities
eliminating wasted capacity
avoiding service disruption
Task 2: Carbon-Aware Rebalancing
Task ID: task_2_carbon_aware_rebalancing
Difficulty: Medium
Max Steps: 8
Objective:
Move flexible workloads from high-carbon regions into lower-carbon capacity while respecting placement constraints.

What it tests:

constrained workload migration
protected workload handling
carbon-aware infrastructure control
Task 3: Resilient Multi-Objective Dispatch
Task ID: task_3_resilient_multi_objective_dispatch
Difficulty: Hard
Max Steps: 10
Objective:
Reduce carbon, cost, and thermal risk during regional stress while preserving reliability for real-time services.

What it tests:

multi-objective reasoning
hotspot mitigation
reliability-aware optimization under pressure
Graders
Each task has a deterministic programmatic grader with a score in [0.0, 1.0].

Grading Properties
deterministic
reproducible
bounded
partial-credit based
hard-failure aware
Task-Specific Grading
Task	Grader Focus
Task 1	idle cleanup, workload safety, carbon improvement
Task 2	low-carbon shift ratio, capacity safety, pinned workload protection
Task 3	carbon gain, cost gain, thermal stability, hotspot relief, reliability
Hard thermal failures and critical workload violations cap the final score.

Baseline Inference
The repository includes a root-level inference.py baseline script.

It:

runs all tasks
uses the OpenAI client for LLM calls
reads required environment variables
emits structured stdout logs in [START], [STEP], and [END] format
produces reproducible scores
Required Environment Variables
Variable	Purpose
API_BASE_URL	LLM endpoint
MODEL_NAME	model identifier
HF_TOKEN	Hugging Face / API token
LOCAL_IMAGE_NAME	optional, for docker-image based flows
Baseline Scores
Measured with the included deterministic baseline:

Task	Difficulty	Score	Passed
task_1_idle_capacity_cleanup	Easy	0.9191	true
task_2_carbon_aware_rebalancing	Medium	0.8760	true
task_3_resilient_multi_objective_dispatch	Hard	0.7721	true
Project Structure
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
Local Setup
1. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
2. Install dependencies
pip install -r requirements.txt
3. Start the server
uvicorn app:app --host 0.0.0.0 --port 7860
4. Run tests
python -m unittest
5. Run the local preflight
python preflight_check.py
Usage Examples
List tasks
Invoke-RestMethod -Method Get -Uri "http://localhost:7860/tasks"
Reset a task
Invoke-RestMethod -Method Post -Uri "http://localhost:7860/reset" -ContentType "application/json" -Body '{"task_id":"task_1_idle_capacity_cleanup"}'
Step the environment
Invoke-RestMethod -Method Post -Uri "http://localhost:7860/step" -ContentType "application/json" -Body '{"action_type":"noop","task_id":"task_1_idle_capacity_cleanup","rationale":"example"}'
Inspect state
Invoke-RestMethod -Method Get -Uri "http://localhost:7860/state"
Run The Baseline
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
$env:HF_TOKEN="your_token_here"
python inference.py
Docker
Build
docker build -t ecocloud-openenv .
Run
docker run -p 7860:7860 ecocloud-openenv
Hugging Face Spaces
This repository is prepared for Docker-based Hugging Face Spaces deployment.

Deployment Checklist
Space is created with Docker SDK
root endpoint returns 200
/reset responds successfully
API_BASE_URL is configured
MODEL_NAME is configured
HF_TOKEN is configured
baseline script runs successfully against the live Space
Validation Checklist
Before submission, verify:

python -m unittest passes
python preflight_check.py passes
python inference.py completes without error
scores stay in [0.0, 1.0]
HF Space is live
live /reset works
Docker build succeeds
openenv validate passes if required in your environment
Notes
EcoCloud is intentionally deterministic to support fair benchmarking, reproducible grading, and stable evaluation across repeated agent runs.

It is designed to be useful not only as a hackathon project, but as a benchmark for infrastructure-oriented agent evaluation.

License
MIT
