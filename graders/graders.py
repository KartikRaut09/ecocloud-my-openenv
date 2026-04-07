from __future__ import annotations

from typing import Any

from env.environment import compute_metrics_snapshot, compute_task_grade
from tasks.task_data import TASKS


def grade_payload(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    task = TASKS[task_id]
    regions = {region["region_id"]: region for region in payload["regions"]}
    servers = {server["server_id"]: server for server in payload["servers"]}
    workloads = {workload["workload_id"]: workload for workload in payload["workloads"]}
    snapshot = compute_metrics_snapshot(task=task, regions=regions, servers=servers, workloads=workloads, migration_count=payload.get("migration_count", 0))
    return compute_task_grade(task=task, snapshot=snapshot, baseline_metrics=payload["baseline_metrics"])


def grade_task_1(payload: dict[str, Any]) -> dict[str, Any]:
    return grade_payload("task_1_idle_capacity_cleanup", payload)


def grade_task_2(payload: dict[str, Any]) -> dict[str, Any]:
    return grade_payload("task_2_carbon_aware_rebalancing", payload)


def grade_task_3(payload: dict[str, Any]) -> dict[str, Any]:
    return grade_payload("task_3_resilient_multi_objective_dispatch", payload)


GRADERS = {
    "task_1_idle_capacity_cleanup": grade_task_1,
    "task_2_carbon_aware_rebalancing": grade_task_2,
    "task_3_resilient_multi_objective_dispatch": grade_task_3,
}
