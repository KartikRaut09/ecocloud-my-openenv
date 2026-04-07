from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from app import app
from env.baseline_policy import choose_heuristic_action
from env.environment import EcoCloudEnv
from env.models import Action
from tasks.task_data import TASK_ORDER


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_api() -> dict[str, Any]:
    client = TestClient(app)
    health = client.get("/health")
    assert_true(health.status_code == 200, "GET /health failed")

    tasks = client.get("/tasks")
    assert_true(tasks.status_code == 200, "GET /tasks failed")
    task_payload = tasks.json()
    assert_true(len(task_payload) >= 3, "Expected at least 3 tasks")

    reset = client.post("/reset", json={"task_id": TASK_ORDER[0]})
    assert_true(reset.status_code == 200, "POST /reset failed")

    state = client.get("/state")
    assert_true(state.status_code == 200, "GET /state failed")

    observation = reset.json()["observation"]
    noop_step = client.post(
        "/step",
        json={
            "action_type": "noop",
            "task_id": observation["task_id"],
            "rationale": "preflight",
        },
    )
    assert_true(noop_step.status_code == 200, "POST /step failed")

    validate = client.get("/validate")
    assert_true(validate.status_code == 200, "GET /validate failed")
    assert_true(validate.json().get("valid") is True, "/validate did not report valid")

    return {
        "health": health.json(),
        "tasks_count": len(task_payload),
        "validate": validate.json(),
    }


def check_baseline() -> list[dict[str, Any]]:
    env = EcoCloudEnv()
    results: list[dict[str, Any]] = []
    for task_id in TASK_ORDER:
        observation = env.reset(task_id)
        done = False
        step_count = 0
        while not done:
            action = Action(**choose_heuristic_action(observation))
            result = env.step(action)
            observation = result.observation
            done = result.done
            step_count += 1
        grade = env.grade_current_task()
        assert_true(0.0 <= grade["score"] <= 1.0, f"Score out of range for {task_id}")
        results.append(
            {
                "task_id": task_id,
                "score": grade["score"],
                "passed": grade["passed"],
                "steps": step_count,
            }
        )
    return results


def main() -> None:
    api_summary = check_api()
    baseline_summary = check_baseline()
    report = {
        "status": "ok",
        "api": api_summary,
        "baseline": baseline_summary,
        "notes": [
            "Local API checks passed.",
            "Baseline policy completed all tasks.",
            "Docker build and live Hugging Face Space checks still need to be run externally.",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
