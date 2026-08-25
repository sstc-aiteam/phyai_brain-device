import unittest

from orchestrator.planner import DryRunPlanner


class FakeRegistry:
    def safe_planning_observation(self, cell_id):
        if cell_id not in {"left", "right"}:
            return None
        return {
            "pose": [0.1, 0.2, 0.6 if cell_id == "left" else 0.5, 0, 0, 0]
        }


class OrchestratorPlannerTest(unittest.TestCase):
    def setUp(self):
        self.planner = DryRunPlanner(FakeRegistry())

    def test_parallel_two_arm_plan_is_non_executing(self):
        result = self.planner.plan_parallel_stages([
            {"parallel": [
                {"cell_id": "left", "relative_z": 0.03},
                {"cell_id": "right", "relative_z": -0.02},
            ]},
            {"parallel": [
                {"cell_id": "left", "return_to_start": True},
                {"cell_id": "right", "return_to_start": True},
            ]},
        ])
        self.assertFalse(result["will_execute"])
        self.assertEqual(result["backend_calls_made"], 0)
        self.assertEqual(result["execution_mode"], "parallel_stages")
        self.assertEqual(len(result["stages"]), 2)

    def test_rejects_distance_over_limit(self):
        with self.assertRaises(ValueError):
            self.planner.plan([
                {"cell_id": "left", "relative_z": 0.051},
            ])

    def test_rejects_duplicate_cell_in_parallel_stage(self):
        with self.assertRaises(ValueError):
            self.planner.plan_parallel_stages([
                {"parallel": [
                    {"cell_id": "left", "relative_z": 0.01},
                    {"cell_id": "left", "relative_z": -0.01},
                ]},
            ])


if __name__ == "__main__":
    unittest.main()
