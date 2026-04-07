from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO

from fastapi.testclient import TestClient

from app import app
from env.baseline_policy import choose_heuristic_action
from env.environment import EcoCloudEnv
from env.models import Action
from graders.graders import grade_payload
from inference import emit_end, emit_start, emit_step
from tasks.task_data import TASK_ORDER


class EcoCloudEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = EcoCloudEnv()

    def test_reset_returns_expected_task(self) -> None:
        observation = self.env.reset(TASK_ORDER[0])
        self.assertEqual(observation.task_id, TASK_ORDER[0])
        self.assertEqual(observation.step_number, 0)

    def test_step_reward_stays_in_range(self) -> None:
        self.env.reset(TASK_ORDER[0])
        result = self.env.step(
            Action(
                action_type="shutdown_server",
                task_id=TASK_ORDER[0],
                server_id="us-east-gpu-b",
            )
        )
        self.assertGreaterEqual(result.reward.total, 0.0)
        self.assertLessEqual(result.reward.total, 1.0)

    def test_graders_are_deterministic(self) -> None:
        self.env.reset(TASK_ORDER[1])
        payload = self.env.export_payload()
        first = grade_payload(TASK_ORDER[1], payload)
        second = grade_payload(TASK_ORDER[1], payload)
        self.assertEqual(first, second)

    def test_heuristic_policy_solves_all_tasks(self) -> None:
        scores: list[float] = []
        for task_id in TASK_ORDER:
            observation = self.env.reset(task_id)
            done = False
            while not done:
                action = Action(**choose_heuristic_action(observation))
                result = self.env.step(action)
                observation = result.observation
                done = result.done
            final_grade = self.env.grade_current_task()
            scores.append(final_grade["score"])

        self.assertGreaterEqual(scores[0], 0.9)
        self.assertGreaterEqual(scores[1], 0.85)
        self.assertGreaterEqual(scores[2], 0.74)


class ApiSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_validate_endpoint(self) -> None:
        response = self.client.get("/validate")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["valid"])
        self.assertIn("tasks", data)

    def test_reset_and_state_endpoints(self) -> None:
        reset_response = self.client.post("/reset", json={"task_id": TASK_ORDER[0]})
        self.assertEqual(reset_response.status_code, 200)
        state_response = self.client.get("/state")
        self.assertEqual(state_response.status_code, 200)
        self.assertEqual(state_response.json()["task_id"], TASK_ORDER[0])

    def test_reset_accepts_empty_body(self) -> None:
        reset_response = self.client.post("/reset", json={})
        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(reset_response.json()["task"]["task_id"], TASK_ORDER[0])


class InferenceFormattingTests(unittest.TestCase):
    def test_emit_format(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            emit_start("task_1_idle_capacity_cleanup")
            emit_step(1, "noop()", 0.0, False, None)
            emit_end(True, 1, 0.5, [0.0])
        lines = [line.strip() for line in buffer.getvalue().splitlines()]
        self.assertEqual(
            lines[0],
            "[START] task=task_1_idle_capacity_cleanup env=ecocloud-openenv model=meta-llama/Meta-Llama-3.1-8B-Instruct",
        )
        self.assertEqual(lines[1], "[STEP] step=1 action=noop() reward=0.00 done=false error=null")
        self.assertEqual(lines[2], "[END] success=true steps=1 score=0.50 rewards=0.00")


if __name__ == "__main__":
    unittest.main()
