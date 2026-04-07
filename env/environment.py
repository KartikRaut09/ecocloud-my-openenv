from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

from env.models import Action, ActionType, EnvState, Metrics, Observation, RegionState, Reward, ServerState, StepResult, WorkloadState
from tasks.task_data import TASK_ORDER, TASKS, list_task_summaries

ENV_NAME = "ecocloud-openenv"
ENV_VERSION = "1.0.0"
THERMAL_WARNING_C = 72.0
THERMAL_HARD_LIMIT_C = 80.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round4(value: float) -> float:
    return round(float(value), 4)


def _priority_weight(priority: str) -> float:
    return {"low": 0.5, "medium": 0.8, "high": 1.0, "critical": 1.25}.get(priority, 0.8)


def _protected_fraction(workload_views: dict[str, dict[str, Any]], workload_ids: list[str] | None = None, require_initial_server: bool = False) -> float:
    selected = []
    for workload in workload_views.values():
        if workload_ids is not None:
            if workload["workload_id"] in workload_ids:
                selected.append(workload)
            continue
        if workload["target_class"] == "protect_service":
            selected.append(workload)
    if not selected:
        return 1.0

    total_weight = 0.0
    satisfied_weight = 0.0
    for workload in selected:
        weight = _priority_weight(workload["priority"])
        total_weight += weight
        is_valid = workload["violation_severity"] <= 0.0
        if require_initial_server:
            is_valid = is_valid and workload["assigned_server_id"] == workload["initial_server_id"]
        if is_valid:
            satisfied_weight += weight
    if total_weight <= 0.0:
        return 1.0
    return _clip01(satisfied_weight / total_weight)


def _shift_fraction(task: dict[str, Any], workload_views: dict[str, dict[str, Any]]) -> float:
    low_carbon_regions = set(task.get("low_carbon_regions", []))
    target_workloads = [workload for workload in workload_views.values() if workload["target_class"] == "shift_low_carbon"]
    if not target_workloads:
        return 1.0
    total_gpu = sum(workload["gpu_demand"] for workload in target_workloads)
    shifted_gpu = sum(workload["gpu_demand"] for workload in target_workloads if workload["current_region_id"] in low_carbon_regions)
    if total_gpu <= 0.0:
        return 1.0
    return _clip01(shifted_gpu / total_gpu)


def compute_metrics_snapshot(task: dict[str, Any], regions: dict[str, dict[str, Any]], servers: dict[str, dict[str, Any]], workloads: dict[str, dict[str, Any]], migration_count: int) -> dict[str, Any]:
    assignments: dict[str, list[dict[str, Any]]] = {server_id: [] for server_id in servers}
    for workload in workloads.values():
        assignments.setdefault(workload["assigned_server_id"], []).append(workload)

    total_power = 0.0
    total_carbon = 0.0
    total_cost = 0.0
    total_demand = sum(workload["gpu_demand"] for workload in workloads.values())
    total_overloaded_gpu = 0.0
    thermal_risk_sum = 0.0
    thermal_warning_servers = 0
    hard_thermal_violations = 0
    active_idle_servers = 0
    active_server_count = 0
    total_priority_weight = 0.0
    weighted_violations = 0.0
    critical_violations = 0
    server_views: dict[str, dict[str, Any]] = {}

    for server_id, server in servers.items():
        region = regions[server["region_id"]]
        assigned = sorted(assignments.get(server_id, []), key=lambda item: item["workload_id"])
        assigned_gpu_demand = sum(item["gpu_demand"] for item in assigned)
        is_active = server["status"] == "active"

        if is_active:
            active_server_count += 1
            effective_capacity = server["gpu_capacity"] * (server["power_cap_kw"] / server["max_power_kw"])
            utilization = assigned_gpu_demand / server["gpu_capacity"] if server["gpu_capacity"] else 0.0
            load_ratio = min(1.0, utilization)
            current_power_kw = server["base_power_kw"] if assigned_gpu_demand <= 0.0 else server["base_power_kw"] + load_ratio * server["power_cap_kw"]
            current_temp_c = region["ambient_temp_c"] + current_power_kw * server["thermal_coeff"] - server["cooling_offset"]
            overloaded_gpu = max(0.0, assigned_gpu_demand - effective_capacity)
            if assigned_gpu_demand <= 0.0:
                active_idle_servers += 1
        else:
            effective_capacity = 0.0
            utilization = 0.0
            current_power_kw = 0.0
            current_temp_c = region["ambient_temp_c"] - 1.0
            overloaded_gpu = assigned_gpu_demand

        if is_active and current_temp_c >= THERMAL_WARNING_C:
            thermal_warning_servers += 1
        if is_active and current_temp_c >= THERMAL_HARD_LIMIT_C:
            hard_thermal_violations += 1

        thermal_risk_sum += _clip01((current_temp_c - 68.0) / 15.0) if is_active else 0.0
        total_overloaded_gpu += overloaded_gpu
        total_power += current_power_kw
        total_carbon += current_power_kw * region["carbon_intensity"]
        total_cost += current_power_kw * region["energy_price_usd_per_kwh"]

        server_views[server_id] = {
            **copy.deepcopy(server),
            "assigned_workloads": [item["workload_id"] for item in assigned],
            "assigned_gpu_demand": assigned_gpu_demand,
            "effective_capacity": effective_capacity,
            "current_power_kw": current_power_kw,
            "current_temp_c": current_temp_c,
            "utilization": utilization,
            "overloaded_gpu": overloaded_gpu,
        }

    workload_views: dict[str, dict[str, Any]] = {}
    for workload_id, workload in workloads.items():
        assigned_server = servers.get(workload["assigned_server_id"])
        server_view = server_views.get(workload["assigned_server_id"])
        current_region_id = assigned_server["region_id"] if assigned_server else "unassigned"
        violation_severity = 0.0

        if assigned_server is None:
            violation_severity = 1.0
        elif assigned_server["status"] != "active":
            violation_severity = 1.0
        elif current_region_id not in workload["allowed_regions"]:
            violation_severity = 1.0
        elif server_view and server_view["assigned_gpu_demand"] > 0.0:
            overload_share = server_view["overloaded_gpu"] * (workload["gpu_demand"] / server_view["assigned_gpu_demand"])
            if workload["gpu_demand"] > 0.0:
                violation_severity = max(violation_severity, _clip01(overload_share / workload["gpu_demand"]))

        weight = _priority_weight(workload["priority"])
        total_priority_weight += weight
        weighted_violations += weight * violation_severity
        if workload["priority"] == "critical" and violation_severity > 0.0:
            critical_violations += 1

        workload_views[workload_id] = {**copy.deepcopy(workload), "current_region_id": current_region_id, "violation_severity": violation_severity}

    reliability_score = 1.0 - min(1.0, weighted_violations / total_priority_weight) if total_priority_weight > 0.0 else 1.0
    thermal_risk_index = _clip01(thermal_risk_sum / active_server_count) if active_server_count else 0.0
    average_carbon_intensity = total_carbon / total_power if total_power > 0.0 else 0.0

    return {
        "server_views": server_views,
        "workload_views": workload_views,
        "metrics": {
            "total_power_kw": _round4(total_power),
            "total_carbon_kgco2e": _round4(total_carbon),
            "total_cost_usd": _round4(total_cost),
            "average_carbon_intensity": _round4(average_carbon_intensity),
            "max_temperature_c": _round4(max((server["current_temp_c"] for server in server_views.values()), default=0.0)),
            "overloaded_gpu": _round4(total_overloaded_gpu),
            "thermal_risk_index": _round4(thermal_risk_index),
            "thermal_warning_servers": thermal_warning_servers,
            "hard_thermal_violations": hard_thermal_violations,
            "active_idle_servers": active_idle_servers,
            "migration_count": migration_count,
            "reliability_score": _round4(reliability_score),
            "critical_violations": critical_violations,
            "total_gpu_demand": _round4(total_demand),
        },
    }


def compute_task_grade(task: dict[str, Any], snapshot: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = snapshot["metrics"]
    workload_views = snapshot["workload_views"]
    server_views = snapshot["server_views"]
    baseline_carbon = max(baseline_metrics["total_carbon_kgco2e"], 1e-9)
    baseline_cost = max(baseline_metrics["total_cost_usd"], 1e-9)
    total_gpu_demand = max(metrics["total_gpu_demand"], 1e-9)

    carbon_gain = _clip01((baseline_metrics["total_carbon_kgco2e"] - metrics["total_carbon_kgco2e"]) / baseline_carbon)
    cost_gain = _clip01((baseline_metrics["total_cost_usd"] - metrics["total_cost_usd"]) / baseline_cost)
    reliability = _clip01(metrics["reliability_score"])
    thermal = _clip01(1.0 - metrics["thermal_risk_index"])
    capacity = _clip01(1.0 - (metrics["overloaded_gpu"] / total_gpu_demand))
    task_id = task["task_id"]

    if task_id == "task_1_idle_capacity_cleanup":
        idle_servers = task["metadata"]["idle_servers"]
        idle_cleanup = _clip01(sum(1 for server_id in idle_servers if server_views[server_id]["status"] == "off") / len(idle_servers))
        protected = _protected_fraction(workload_views, require_initial_server=True)
        score = 0.7 * idle_cleanup + 0.2 * protected + 0.1 * carbon_gain
        efficiency = (idle_cleanup + carbon_gain) / 2.0
        reliability_component = protected
        components = {"idle_cleanup": _round4(idle_cleanup), "protected_services": _round4(protected), "carbon_gain": _round4(carbon_gain)}
    elif task_id == "task_2_carbon_aware_rebalancing":
        shift = _shift_fraction(task, workload_views)
        protected = _protected_fraction(workload_views, workload_ids=task["metadata"].get("protected_workloads"))
        score = 0.45 * shift + 0.2 * capacity + 0.15 * protected + 0.2 * carbon_gain
        efficiency = (shift + carbon_gain) / 2.0
        reliability_component = (capacity + protected) / 2.0
        components = {"shift_fraction": _round4(shift), "capacity_score": _round4(capacity), "protected_services": _round4(protected), "carbon_gain": _round4(carbon_gain)}
    else:
        hot_regions = set(task["metadata"].get("hot_regions", []))
        movable_hot_workloads = [workload for workload in workload_views.values() if workload.get("initial_region_id") in hot_regions and workload["can_migrate"]]
        total_hot_gpu = sum(workload["gpu_demand"] for workload in movable_hot_workloads)
        relieved_hot_gpu = sum(workload["gpu_demand"] for workload in movable_hot_workloads if workload["current_region_id"] not in hot_regions)
        hotspot_relief = _clip01(relieved_hot_gpu / total_hot_gpu) if total_hot_gpu > 0.0 else 1.0
        protected = _protected_fraction(workload_views, workload_ids=task["metadata"].get("protected_workloads"))
        score = 0.2 * carbon_gain + 0.15 * cost_gain + 0.25 * thermal + 0.25 * reliability + 0.15 * hotspot_relief
        efficiency = (carbon_gain + cost_gain + hotspot_relief) / 3.0
        reliability_component = (reliability + thermal + protected) / 3.0
        components = {"carbon_gain": _round4(carbon_gain), "cost_gain": _round4(cost_gain), "thermal_score": _round4(thermal), "reliability_score": _round4(reliability), "hotspot_relief": _round4(hotspot_relief), "protected_services": _round4(protected)}

    hard_failures = metrics["hard_thermal_violations"] + metrics["critical_violations"]
    if hard_failures > 0:
        score = min(score, 0.45)

    return {
        "task_id": task_id,
        "score": _round4(_clip01(score)),
        "passed": bool(score >= task["success_threshold"] and hard_failures == 0 and metrics["critical_violations"] == 0),
        "components": components,
        "efficiency": _round4(_clip01(efficiency)),
        "reliability": _round4(_clip01(reliability_component)),
        "hard_failures": hard_failures,
    }


class EcoCloudEnv:
    def __init__(self) -> None:
        self._task: dict[str, Any] | None = None
        self._regions: dict[str, dict[str, Any]] = {}
        self._servers: dict[str, dict[str, Any]] = {}
        self._workloads: dict[str, dict[str, Any]] = {}
        self._baseline_metrics: dict[str, Any] = {}
        self._episode_id = ""
        self._step_count = 0
        self._done = False
        self._migration_count = 0
        self._last_event = "Environment initialized."
        self._last_action_error: str | None = None
        self._last_reward: Reward | None = None
        self._recent_actions: list[Action] = []
        self.reset(TASK_ORDER[0])

    @property
    def task_id(self) -> str:
        if not self._task:
            raise RuntimeError("Environment not initialized.")
        return self._task["task_id"]

    def list_tasks(self) -> list[dict[str, Any]]:
        return [summary.model_dump() for summary in list_task_summaries()]

    def validate_spec(self) -> dict[str, Any]:
        return {"valid": True, "env_name": ENV_NAME, "version": ENV_VERSION, "typed_models": True, "endpoints": {"reset": "/reset", "step": "/step", "state": "/state"}, "tasks": self.list_tasks()}

    def reset(self, task_id: str) -> Observation:
        if task_id not in TASKS:
            raise KeyError(f"Unknown task_id: {task_id}")
        self._task = copy.deepcopy(TASKS[task_id])
        self._regions = {region["region_id"]: copy.deepcopy(region) for region in self._task["regions"]}
        self._servers = {server["server_id"]: copy.deepcopy(server) for server in self._task["servers"]}
        self._workloads = {workload["workload_id"]: copy.deepcopy(workload) for workload in self._task["workloads"]}
        for workload in self._workloads.values():
            initial_server = self._servers[workload["initial_server_id"]]
            workload["initial_region_id"] = initial_server["region_id"]
        self._episode_id = uuid4().hex[:10]
        self._step_count = 0
        self._done = False
        self._migration_count = 0
        self._last_event = f"Reset task {task_id}."
        self._last_action_error = None
        self._last_reward = None
        self._recent_actions = []
        snapshot = self._snapshot()
        self._baseline_metrics = snapshot["metrics"]
        return self._build_observation(snapshot)

    def state(self) -> EnvState:
        return self._build_state(self._snapshot())

    def export_payload(self) -> dict[str, Any]:
        return {"regions": copy.deepcopy(list(self._regions.values())), "servers": copy.deepcopy(list(self._servers.values())), "workloads": copy.deepcopy(list(self._workloads.values())), "migration_count": self._migration_count, "baseline_metrics": copy.deepcopy(self._baseline_metrics)}

    def grade_current_task(self) -> dict[str, Any]:
        return compute_task_grade(self._task, self._snapshot(), self._baseline_metrics)

    def step(self, action: Action) -> StepResult:
        invalid_action = False
        if self._done:
            invalid_action = True
            self._last_event = "Episode already complete; action ignored."
            self._last_action_error = self._last_event
        else:
            self._step_count += 1
            self._last_event, invalid_action = self._apply_action(action)
            self._last_action_error = self._last_event if invalid_action else None
            self._recent_actions.append(action)
            self._recent_actions = self._recent_actions[-10:]

        snapshot = self._snapshot()
        grade = compute_task_grade(self._task, snapshot, self._baseline_metrics)
        penalty = 0.02
        if invalid_action:
            penalty += 0.22
        if action.action_type == ActionType.NOOP and grade["score"] < self._task["success_threshold"]:
            penalty += 0.05
        if grade["hard_failures"] > 0:
            penalty += min(0.15, 0.07 * grade["hard_failures"])

        reward = Reward(
            total=_round4(_clip01(0.65 * grade["score"] + 0.2 * grade["efficiency"] + 0.15 * grade["reliability"] - penalty)),
            progress=_round4(grade["score"]),
            efficiency=_round4(grade["efficiency"]),
            reliability=_round4(grade["reliability"]),
            penalty=_round4(_clip01(penalty)),
            grader_score=_round4(grade["score"]),
            reason=self._last_event,
        )
        self._last_reward = reward
        self._done = bool(self._step_count >= self._task["max_steps"] or (grade["score"] >= self._task["success_threshold"] and grade["hard_failures"] == 0))

        return StepResult(
            observation=self._build_observation(snapshot),
            reward=reward,
            done=self._done,
            info={"grader": grade, "invalid_action": invalid_action, "last_action_error": self._last_action_error, "episode_id": self._episode_id, "success_threshold": self._task["success_threshold"]},
        )

    def _snapshot(self) -> dict[str, Any]:
        return compute_metrics_snapshot(self._task, self._regions, self._servers, self._workloads, self._migration_count)

    def _build_observation(self, snapshot: dict[str, Any]) -> Observation:
        regions = [
            RegionState(**{key: value for key, value in copy.deepcopy(region).items() if key in RegionState.model_fields})
            for region in sorted(self._regions.values(), key=lambda item: item["region_id"])
        ]
        servers = [
            ServerState(**{key: value for key, value in copy.deepcopy(snapshot["server_views"][server_id]).items() if key in ServerState.model_fields})
            for server_id in sorted(snapshot["server_views"])
        ]
        workloads = [
            WorkloadState(**{key: value for key, value in copy.deepcopy(snapshot["workload_views"][workload_id]).items() if key in WorkloadState.model_fields})
            for workload_id in sorted(snapshot["workload_views"])
        ]
        metrics = Metrics(**{key: value for key, value in snapshot["metrics"].items() if key in Metrics.model_fields})
        return Observation(task_id=self._task["task_id"], task_name=self._task["name"], difficulty=self._task["difficulty"], objective=self._task["objective"], step_number=self._step_count, max_steps=self._task["max_steps"], recent_event=self._last_event, allowed_action_types=[ActionType.MIGRATE_WORKLOAD, ActionType.SET_POWER_CAP, ActionType.SHUTDOWN_SERVER, ActionType.ACTIVATE_SERVER, ActionType.NOOP], regions=regions, servers=servers, workloads=workloads, metrics=metrics)

    def _build_state(self, snapshot: dict[str, Any]) -> EnvState:
        return EnvState(env_name=ENV_NAME, version=ENV_VERSION, episode_id=self._episode_id, task_id=self._task["task_id"], step_count=self._step_count, max_steps=self._task["max_steps"], done=self._done, last_event=self._last_event, last_action_error=self._last_action_error, last_reward=self._last_reward, metrics=Metrics(**{key: value for key, value in snapshot["metrics"].items() if key in Metrics.model_fields}), recent_actions=self._recent_actions)

    def _server_gpu_demand(self, server_id: str) -> float:
        return sum(workload["gpu_demand"] for workload in self._workloads.values() if workload["assigned_server_id"] == server_id)

    def _apply_action(self, action: Action) -> tuple[str, bool]:
        if action.task_id != self.task_id:
            return "Action task_id does not match the active task.", True
        if action.action_type == ActionType.NOOP:
            return "No-op action applied.", False
        if action.action_type == ActionType.ACTIVATE_SERVER:
            if not action.server_id or action.server_id not in self._servers:
                return "Activate action missing a valid server_id.", True
            server = self._servers[action.server_id]
            if server["status"] == "active":
                return f"{action.server_id} is already active.", True
            server["status"] = "active"
            return f"Activated server {action.server_id}.", False
        if action.action_type == ActionType.SHUTDOWN_SERVER:
            if not action.server_id or action.server_id not in self._servers:
                return "Shutdown action missing a valid server_id.", True
            server = self._servers[action.server_id]
            if server["status"] != "active":
                return f"{action.server_id} is already off.", True
            if self._server_gpu_demand(action.server_id) > 0.0:
                return f"Cannot shut down {action.server_id} while workloads are assigned.", True
            server["status"] = "off"
            return f"Shut down server {action.server_id}.", False
        if action.action_type == ActionType.SET_POWER_CAP:
            if not action.server_id or action.server_id not in self._servers:
                return "Power-cap action missing a valid server_id.", True
            if action.power_cap_kw is None:
                return "Power-cap action missing power_cap_kw.", True
            server = self._servers[action.server_id]
            if server["status"] != "active":
                return f"Cannot change power cap on inactive server {action.server_id}.", True
            min_cap = 0.35 * server["max_power_kw"]
            if not (min_cap <= action.power_cap_kw <= server["max_power_kw"]):
                return f"Power cap must be between {min_cap:.2f} and {server['max_power_kw']:.2f} kW.", True
            server["power_cap_kw"] = round(action.power_cap_kw, 3)
            return f"Set power cap on {action.server_id} to {server['power_cap_kw']:.2f} kW.", False
        if action.action_type == ActionType.MIGRATE_WORKLOAD:
            if not action.workload_id or action.workload_id not in self._workloads or not action.target_server_id or action.target_server_id not in self._servers:
                return "Migration action requires valid workload_id and target_server_id.", True
            workload = self._workloads[action.workload_id]
            target_server = self._servers[action.target_server_id]
            if not workload["can_migrate"]:
                return f"Workload {action.workload_id} is not allowed to migrate.", True
            if target_server["status"] != "active":
                return f"Target server {action.target_server_id} is not active.", True
            if target_server["region_id"] not in workload["allowed_regions"]:
                return f"Target region {target_server['region_id']} is not allowed for {action.workload_id}.", True
            if workload["assigned_server_id"] == action.target_server_id:
                return f"{action.workload_id} is already on {action.target_server_id}.", True
            if self._server_gpu_demand(action.target_server_id) + workload["gpu_demand"] > target_server["gpu_capacity"]:
                return f"Target server {action.target_server_id} lacks raw GPU capacity.", True
            source_server_id = workload["assigned_server_id"]
            workload["assigned_server_id"] = action.target_server_id
            self._migration_count += 1
            return f"Migrated {action.workload_id} from {source_server_id} to {action.target_server_id}.", False
        return "Unsupported action.", True
