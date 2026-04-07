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


ENV_BASE_URL = "http://localhost:7860"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "ecocloud-openenv"


def request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{ENV_BASE_URL.rstrip('/')}{path}"
    response = requests.request(method=method, url=url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def compact_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": action.get("action_type"),
        "task_id": action.get("task_id"),
        "server_id": action.get("server_id"),
        "workload_id": action.get("workload_id"),
        "target_server_id": action.get("target_server_id"),
        "power_cap_kw": action.get("power_cap_kw"),
    }


def action_to_str(action: dict[str, Any]) -> str:
    action_type = action.get("action_type", "unknown")
    parts: list[str] = []
    for key in ["server_id", "workload_id", "target_server_id", "power_cap_kw"]:
        value = action.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return f"{action_type}({','.join(parts)})" if parts else f"{action_type}()"


def format_reward(value: float) -> str:
    return f"{float(value):.2f}"


def model_review(client: Any, observation: dict[str, Any], heuristic_action: dict[str, Any]) -> dict[str, Any]:
    if client is None:
        return heuristic_action

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
                "Validate the proposed OpenEnv action. "
                "Return compact JSON only: {\"action\":{...same action fields...}}. "
                "If the action is reasonable, return it unchanged."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=120,
        )
        text = response.choices[0].message.content or ""
        parsed = parse_json_object(text)
        candidate = (parsed or {}).get("action") if parsed else None
        if isinstance(candidate, dict) and compact_action(candidate) == compact_action(heuristic_action):
            return heuristic_action
    except Exception:
        pass
    return heuristic_action


def emit_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def emit_step(step: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={format_reward(reward)} "
        f"done={'true' if done else 'false'} error={error_value}",
        flush=True,
    )


def emit_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(format_reward(value) for value in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={format_reward(score)} rewards={rewards_str}",
        flush=True,
    )


def run() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if OpenAI is not None and HF_TOKEN else None
    task_summaries = request_json("GET", "/tasks")

    for task in task_summaries:
        task_id = task["task_id"]
        step_counter = 0
        final_score = 0.0
        success = False
        rewards: list[float] = []
        emit_start(task_id)
        try:
            reset_payload = request_json("POST", "/reset", {"task_id": task_id})
            observation = reset_payload["observation"]

            done = False
            while not done and step_counter < task["max_steps"]:
                heuristic_action = choose_heuristic_action(observation)
                chosen_action = model_review(client, observation, heuristic_action)
                step_payload = request_json("POST", "/step", chosen_action)
                observation = step_payload["observation"]
                reward = float(step_payload["reward"]["total"])
                done = bool(step_payload["done"])
                error = step_payload["info"].get("last_action_error")
                step_counter += 1
                rewards.append(reward)
                emit_step(step_counter, action_to_str(chosen_action), reward, done, error)

            final_grade = request_json("GET", f"/grade/{task_id}")
            final_score = float(final_grade["score"])
            success = bool(final_grade["passed"])
        except Exception as exc:
            final_score = 0.0
            success = False
        finally:
            emit_end(success, step_counter, final_score, rewards)


if __name__ == "__main__":
    run()
