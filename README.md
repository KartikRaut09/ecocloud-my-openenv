---
title: EcoCloud OpenEnv
emoji: "🌿"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - carbon-aware-scheduling
  - infrastructure
  - reinforcement-learning
license: mit
---

# EcoCloud OpenEnv

EcoCloud is a real-world OpenEnv environment for carbon-aware GPU infrastructure operations. An agent acts like a cloud capacity operator: it migrates workloads, activates or shuts down servers, and adjusts power caps to reduce emissions, cost, and thermal risk without breaking reliability.

This is not a toy simulator. It models a workflow humans already do in platform, capacity-planning, and SRE teams: deciding where expensive GPU jobs should run under operational constraints.

## Why This Fits The Hackathon

- Real-world task: carbon-aware infrastructure and workload dispatch
- Full API surface: `/reset`, `/step`, `/state`, `/validate`
- Typed Pydantic models for observations, actions, and rewards
- Three deterministic tasks with programmatic graders
- Reward shaping with partial progress and penalties
- `inference.py` at repo root
- Dockerfile + HF Space-ready README front matter

## Tasks

### 1. Idle Capacity Cleanup (`easy`)

Shut down idle GPU servers without disturbing live production workloads.

### 2. Carbon-Aware Rebalancing (`medium`)

Move flexible workloads away from a high-carbon region into cleaner capacity while keeping the pinned search workload in place.

### 3. Resilient Multi-Objective Dispatch (`hard`)

Handle a heat wave and price spike by balancing carbon, cost, thermal headroom, and reliability for a mixed real-time plus batch fleet.

## Action Space

The agent can emit one typed action per step:

- `migrate_workload`
- `set_power_cap`
- `shutdown_server`
- `activate_server`
- `noop`

Each action includes `task_id` and optional fields like `server_id`, `workload_id`, `target_server_id`, and `power_cap_kw`.

## Observation Space

Each observation includes:

- task metadata: `task_id`, `task_name`, `difficulty`, `objective`
- step metadata: `step_number`, `max_steps`, `recent_event`
- structured region state
- structured server state
- structured workload state
- aggregated metrics:
  - total power
  - total carbon
  - total cost
  - max temperature
  - overloaded GPU amount
  - thermal risk index
  - migration count
  - reliability score

## Reward Design

The reward is dense and bounded in `[0.0, 1.0]`.

- `progress`: current grader score for the task
- `efficiency`: task-specific operational improvement, such as carbon gain or hotspot relief
- `reliability`: service protection, capacity safety, and thermal stability
- `penalty`: invalid actions, useless `noop`s, and hard failures

This gives meaningful partial feedback over the full trajectory instead of only terminal success.

## Graders

All three tasks use deterministic graders with scores in `[0.0, 1.0]`.

- Easy: idle cleanup, service protection, carbon gain
- Medium: low-carbon workload shift, capacity safety, pinned-workload protection, carbon gain
- Hard: carbon gain, cost gain, thermal score, reliability score, hotspot relief

Hard thermal failures and critical-service violations cap the final score.

## Local Run

### Install

```bash
pip install -r requirements.txt
```

### Start the environment

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run tests

```bash
python -m unittest
```

### Run the baseline

Set the hackathon variables first:

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
set HF_TOKEN=your_token_here
```

Then run:

```bash
python inference.py
```

`inference.py` uses the OpenAI client when credentials are present and keeps action selection deterministic through the built-in baseline policy.

## Baseline Scores

Measured locally with the deterministic baseline policy:

- `task_1_idle_capacity_cleanup`: `0.9191`
- `task_2_carbon_aware_rebalancing`: `0.8760`
- `task_3_resilient_multi_objective_dispatch`: `0.7721`

## Project Layout

```text
app.py
env/
  baseline_policy.py
  environment.py
  models.py
graders/
  graders.py
tasks/
  task_data.py
tests/
  test_env.py
inference.py
openenv.yaml
Dockerfile
README.md
```

## HF Space Deployment

This repo is already structured for a Docker-based Hugging Face Space:

- README front matter sets `sdk: docker`
- `Dockerfile` launches `uvicorn app:app`
- app listens on port `7860`
- tags include `openenv`

### Deploy Steps

1. Create a new Hugging Face Space.
2. Choose `Docker` as the SDK.
3. Push this repository to the Space.
4. In Space settings, add:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
5. Wait for the container build to finish.
6. Verify:
   - `GET /health`
   - `GET /tasks`
   - `POST /reset`
   - `GET /state`

## Pre-Submission Check

Run this before submitting:

```bash
python preflight_check.py
```

This checks the local API surface and verifies that the deterministic baseline completes all tasks successfully.

## Remaining Submission Risk

The code is locally complete, but submission readiness still depends on:

- successful Hugging Face Space deployment
- successful Docker build on the target platform
- exact compliance of `inference.py` log formatting with the official sample
- passing the official hackathon validator if one is provided
=======
title: Ecocloud Openenv
emoji: ⚡
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 7559b09478fcd05fe48a2c974d7be1fb316751a6
