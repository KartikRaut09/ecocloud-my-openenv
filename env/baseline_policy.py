from __future__ import annotations

from typing import Any


def _obs_dict(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    return observation


def _server_loads(observation: dict[str, Any]) -> dict[str, float]:
    loads = {server["server_id"]: 0.0 for server in observation["servers"]}
    for workload in observation["workloads"]:
        loads[workload["assigned_server_id"]] = loads.get(workload["assigned_server_id"], 0.0) + workload["gpu_demand"]
    return loads


def _region_lookup(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {region["region_id"]: region for region in observation["regions"]}


def _best_target_server(observation: dict[str, Any], workload: dict[str, Any], active_only: bool = True) -> dict[str, Any] | None:
    loads = _server_loads(observation)
    regions = _region_lookup(observation)
    allowed = set(workload["allowed_regions"])
    candidates = []
    for server in observation["servers"]:
        if active_only and server["status"] != "active":
            continue
        if server["region_id"] not in allowed:
            continue
        free_capacity = server["gpu_capacity"] - loads.get(server["server_id"], 0.0)
        if free_capacity + 1e-9 < workload["gpu_demand"]:
            continue
        region = regions[server["region_id"]]
        candidates.append((region["carbon_intensity"], region["energy_price_usd_per_kwh"], server["current_temp_c"], server["server_id"], server))
    if not candidates:
        return None
    return sorted(candidates)[0][-1]


def _inactive_server_to_activate(observation: dict[str, Any], allowed_regions: set[str] | None = None) -> dict[str, Any] | None:
    regions = _region_lookup(observation)
    candidates = []
    for server in observation["servers"]:
        if server["status"] != "off":
            continue
        if allowed_regions and server["region_id"] not in allowed_regions:
            continue
        region = regions[server["region_id"]]
        candidates.append((region["carbon_intensity"], region["energy_price_usd_per_kwh"], server["server_id"], server))
    if not candidates:
        return None
    return sorted(candidates)[0][-1]


def _idle_shutdown_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    loads = _server_loads(observation)
    idle_servers = [server for server in observation["servers"] if server["status"] == "active" and loads.get(server["server_id"], 0.0) <= 0.0]
    if not idle_servers:
        return None
    target = sorted(idle_servers, key=lambda item: item["server_id"])[0]
    return {"action_type": "shutdown_server", "task_id": observation["task_id"], "server_id": target["server_id"], "rationale": "Idle server can be powered down safely."}


def _power_cap_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    loads = _server_loads(observation)
    candidates = []
    for server in observation["servers"]:
        if server["status"] != "active":
            continue
        demand = loads.get(server["server_id"], 0.0)
        if demand <= 0.0:
            continue
        required_ratio = min(1.0, max(0.45, (demand + 0.5) / server["gpu_capacity"]))
        desired_cap = round(server["max_power_kw"] * required_ratio, 2)
        if desired_cap + 0.15 >= server["power_cap_kw"]:
            continue
        if server["current_temp_c"] < 64.0 and demand / server["gpu_capacity"] > 0.7:
            continue
        reduction = server["power_cap_kw"] - desired_cap
        candidates.append((-reduction, server["server_id"], desired_cap, server))
    if not candidates:
        return None
    _, _, desired_cap, server = sorted(candidates)[0]
    return {"action_type": "set_power_cap", "task_id": observation["task_id"], "server_id": server["server_id"], "power_cap_kw": desired_cap, "rationale": "Load fits with a lower cap, reducing power and thermal stress."}


def choose_heuristic_action(observation: Any) -> dict[str, Any]:
    obs = _obs_dict(observation)
    task_id = obs["task_id"]
    regions = _region_lookup(obs)

    if task_id == "task_1_idle_capacity_cleanup":
        shutdown_action = _idle_shutdown_action(obs)
        if shutdown_action:
            return shutdown_action
        return {"action_type": "noop", "task_id": task_id, "rationale": "No further safe shutdowns remain."}

    if task_id == "task_2_carbon_aware_rebalancing":
        low_carbon_regions = {region["region_id"] for region in obs["regions"] if region["carbon_intensity"] <= 0.2}
        flexible = sorted((workload for workload in obs["workloads"] if workload["can_migrate"] and workload["target_class"] == "shift_low_carbon"), key=lambda item: (-item["gpu_demand"], item["workload_id"]))
        for workload in flexible:
            if workload["current_region_id"] in low_carbon_regions:
                continue
            target = _best_target_server(obs, workload, active_only=True)
            if target and target["region_id"] in low_carbon_regions:
                return {"action_type": "migrate_workload", "task_id": task_id, "workload_id": workload["workload_id"], "target_server_id": target["server_id"], "rationale": "Move flexible workload to lower-carbon active capacity."}
            inactive_target = _inactive_server_to_activate(obs, allowed_regions=low_carbon_regions)
            if inactive_target:
                return {"action_type": "activate_server", "task_id": task_id, "server_id": inactive_target["server_id"], "rationale": "Bring additional low-carbon capacity online for migration."}
        shutdown_action = _idle_shutdown_action(obs)
        if shutdown_action:
            return shutdown_action
        return {"action_type": "noop", "task_id": task_id, "rationale": "Flexible workloads already sit on low-carbon capacity."}

    hot_servers = [server for server in obs["servers"] if server["status"] == "active" and (server["current_temp_c"] >= 71.0 or regions[server["region_id"]]["carbon_intensity"] >= 0.6 or regions[server["region_id"]]["energy_price_usd_per_kwh"] >= 0.15)]
    hot_servers = sorted(hot_servers, key=lambda item: (-item["current_temp_c"], item["server_id"]))
    for server in hot_servers:
        server_workloads = [workload for workload in obs["workloads"] if workload["assigned_server_id"] == server["server_id"] and workload["can_migrate"]]
        server_workloads = sorted(server_workloads, key=lambda item: ({"critical": 3, "high": 2, "medium": 1, "low": 0}[item["priority"]], item["gpu_demand"]), reverse=True)
        for workload in server_workloads:
            target = _best_target_server(obs, workload, active_only=True)
            if target and target["server_id"] != server["server_id"]:
                return {"action_type": "migrate_workload", "task_id": task_id, "workload_id": workload["workload_id"], "target_server_id": target["server_id"], "rationale": "Relieve the hottest or most expensive server first."}
            inactive_target = _inactive_server_to_activate(obs, allowed_regions=set(workload["allowed_regions"]))
            if inactive_target:
                return {"action_type": "activate_server", "task_id": task_id, "server_id": inactive_target["server_id"], "rationale": "Activate cooler capacity before migrating the hot-region workload."}

    flexible = sorted((workload for workload in obs["workloads"] if workload["can_migrate"] and workload["target_class"] == "shift_low_carbon"), key=lambda item: (-item["gpu_demand"], item["workload_id"]))
    for workload in flexible:
        if regions[workload["current_region_id"]]["carbon_intensity"] <= 0.3:
            continue
        target = _best_target_server(obs, workload, active_only=True)
        if target and target["server_id"] != workload["assigned_server_id"]:
            return {"action_type": "migrate_workload", "task_id": task_id, "workload_id": workload["workload_id"], "target_server_id": target["server_id"], "rationale": "Shift flexible load off the highest-carbon region."}

    shutdown_action = _idle_shutdown_action(obs)
    if shutdown_action:
        return shutdown_action
    cap_action = _power_cap_action(obs)
    if cap_action:
        return cap_action
    return {"action_type": "noop", "task_id": task_id, "rationale": "No improving heuristic move remains."}
