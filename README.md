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
