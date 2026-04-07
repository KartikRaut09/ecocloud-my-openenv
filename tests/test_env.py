from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import app
from env.baseline_policy import choose_heuristic_action
from env.environment import EcoCloudEnv
from env.models import Action
from graders.graders import grade_payload
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


if __name__ == "__main__":
    unittest.main()
