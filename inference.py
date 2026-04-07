from __future__ import annotations

import json
import os
from typing import Any

import requests

from env.baseline_policy import choose_heuristic_action

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""


def emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}", flush=True)


def compact_action(action: dict[str, Any]) -> dict[str, Any]:
    ordered = {
        "action_type": action.get("action_type"),
        "task_id": action.get("task_id"),
        "server_id": action.get("server_id"),
        "workload_id": action.get("workload_id"),
        "target_server_id": action.get("target_server_id"),
        "power_cap_kw": action.get("power_cap_kw"),
    }
    return {key: value for key, value in ordered.items() if value is not None}


def parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def model_review(client: Any, observation: dict[str, Any], heuristic_action: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if client is None:
        return heuristic_action, "heuristic"

    payload = {
        "task_id": observation["task_id"],
        "objective": observation["objective"],
        "metrics": observation["metrics"],
        "recent_event": observation["recent_event"],
        "proposed_action": compact_action(heuristic_action),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are validating a deterministic baseline action for an OpenEnv task. "
                "Respond with compact JSON only: "
                "{\"action\":{...exactly as proposed or a stricter subset...},\"rationale\":\"one short sentence\"}"
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=180,
        )
        text = response.choices[0].message.content or ""
        parsed = parse_json_object(text)
        if not parsed:
            return heuristic_action, "heuristic"
        returned_action = parsed.get("action") or {}
        rationale = parsed.get("rationale") or "model-reviewed"
        if compact_action(returned_action) == compact_action(heuristic_action):
            return heuristic_action, str(rationale)
        return heuristic_action, "heuristic"
    except Exception:
        return heuristic_action, "heuristic"


def request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{ENV_BASE_URL.rstrip('/')}{path}"
    response = requests.request(method=method, url=url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def run() -> None:
    client = None
    if OpenAI is not None and API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_summaries = request_json("GET", "/tasks")
    for task in task_summaries:
        task_id = task["task_id"]
        reset_payload = request_json("POST", "/reset", {"task_id": task_id})
        observation = reset_payload["observation"]
        emit("[START]", {"task_id": task_id, "model": MODEL_NAME, "max_steps": task["max_steps"]})

        step_counter = 0
        done = False
        while not done and step_counter < task["max_steps"]:
            heuristic_action = choose_heuristic_action(observation)
            chosen_action, rationale = model_review(client, observation, heuristic_action)
            step_payload = request_json("POST", "/step", chosen_action)
            observation = step_payload["observation"]
            reward = step_payload["reward"]
            grade = step_payload["info"]["grader"]
            done = bool(step_payload["done"])
            step_counter += 1
            emit(
                "[STEP]",
                {
                    "task_id": task_id,
                    "step": step_counter,
                    "action_type": chosen_action["action_type"],
                    "server_id": chosen_action.get("server_id"),
                    "workload_id": chosen_action.get("workload_id"),
                    "target_server_id": chosen_action.get("target_server_id"),
                    "reward": reward["total"],
                    "score": grade["score"],
                    "done": done,
                    "source": rationale,
                },
            )

        final_grade = request_json("GET", f"/grade/{task_id}")
        emit("[END]", {"task_id": task_id, "score": final_grade["score"], "passed": final_grade["passed"], "steps": step_counter})


if __name__ == "__main__":
    run()
