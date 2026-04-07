from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ServerStatus(str, Enum):
    ACTIVE = "active"
    OFF = "off"


class ActionType(str, Enum):
    MIGRATE_WORKLOAD = "migrate_workload"
    SET_POWER_CAP = "set_power_cap"
    SHUTDOWN_SERVER = "shutdown_server"
    ACTIVATE_SERVER = "activate_server"
    NOOP = "noop"


class RegionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str
    label: str
    carbon_intensity: float = Field(..., ge=0.0)
    energy_price_usd_per_kwh: float = Field(..., ge=0.0)
    ambient_temp_c: float


class ServerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str
    label: str
    region_id: str
    status: ServerStatus
    gpu_capacity: float = Field(..., gt=0.0)
    effective_capacity: float = Field(..., ge=0.0)
    base_power_kw: float = Field(..., ge=0.0)
    max_power_kw: float = Field(..., gt=0.0)
    power_cap_kw: float = Field(..., ge=0.0)
    current_power_kw: float = Field(..., ge=0.0)
    current_temp_c: float
    utilization: float = Field(..., ge=0.0)
    assigned_workloads: list[str] = Field(default_factory=list)


class WorkloadState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload_id: str
    label: str
    gpu_demand: float = Field(..., gt=0.0)
    priority: str
    can_migrate: bool
    allowed_regions: list[str]
    assigned_server_id: str
    current_region_id: str
    target_class: str


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_power_kw: float = Field(..., ge=0.0)
    total_carbon_kgco2e: float = Field(..., ge=0.0)
    total_cost_usd: float = Field(..., ge=0.0)
    average_carbon_intensity: float = Field(..., ge=0.0)
    max_temperature_c: float
    overloaded_gpu: float = Field(..., ge=0.0)
    thermal_risk_index: float = Field(..., ge=0.0, le=1.0)
    thermal_warning_servers: int = Field(..., ge=0)
    hard_thermal_violations: int = Field(..., ge=0)
    active_idle_servers: int = Field(..., ge=0)
    migration_count: int = Field(..., ge=0)
    reliability_score: float = Field(..., ge=0.0, le=1.0)
    critical_violations: int = Field(..., ge=0)


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float = Field(..., ge=0.0, le=1.0)
    progress: float = Field(..., ge=0.0, le=1.0)
    efficiency: float = Field(..., ge=0.0, le=1.0)
    reliability: float = Field(..., ge=0.0, le=1.0)
    penalty: float = Field(..., ge=0.0, le=1.0)
    grader_score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_name: str
    difficulty: str
    objective: str
    step_number: int = Field(..., ge=0)
    max_steps: int = Field(..., gt=0)
    recent_event: str
    allowed_action_types: list[ActionType]
    regions: list[RegionState]
    servers: list[ServerState]
    workloads: list[WorkloadState]
    metrics: Metrics


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    task_id: str
    server_id: str | None = None
    workload_id: str | None = None
    target_server_id: str | None = None
    power_cap_kw: float | None = None
    rationale: str = ""


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_name: str
    version: str
    episode_id: str
    task_id: str
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., gt=0)
    done: bool
    last_event: str
    last_reward: Reward | None = None
    metrics: Metrics
    recent_actions: list[Action] = Field(default_factory=list)


class TaskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    name: str
    difficulty: str
    objective: str
    max_steps: int
    success_threshold: float
