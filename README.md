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
