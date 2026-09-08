"""Simulation scenario catalog preserved from the V5.4 benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .coordination_schema import CoordinationSpec, DeviceRule, MaintainFactUntilGoal
from .dispatch_executor import FakeDispatchExecutor
from .goal_schema import GoalCondition, GoalSet
from .tool_catalogs import simulation_runtime_tools
from .tool_model import ToolCatalog
from .world_state import WorldState

_DATA = json.loads(Path(__file__).with_name("_scenario_data.json").read_text())
ALIASES = {
    "logistics": "logistics_single_arm",
    "warehouse": "logistics_single_arm",
    "dual_arm": "dual_arm_cabinet",
}


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    data: dict

    @property
    def description(self):
        return self.data["description"]

    @property
    def max_parallel_actions(self):
        return int(self.data["max_parallel_actions"])

    @property
    def max_iterations(self):
        return int(self.data["max_iterations"])

    def build_world(self):
        return WorldState(self.data["world"])

    def build_goals(self, world: WorldState):
        rows = self.data["goals"]
        if self.name == "inspection":
            rows, previous = [], None
            for point in ("machine_A", "machine_B", "machine_C"):
                inspect_id = f"inspect_{point}"
                rows.append({
                    "goal_id": inspect_id, "subject": point,
                    "field": "inspected", "operator": "eq", "value": True,
                    "depends_on": [previous] if previous else [],
                })
                if world.get(point, "health_state") == "anomaly":
                    evidence_id, report_id = f"evidence_{point}", f"report_{point}"
                    rows += [
                        {"goal_id": evidence_id, "subject": point,
                         "field": "evidence_captured", "operator": "eq", "value": True,
                         "depends_on": [inspect_id]},
                        {"goal_id": report_id, "subject": point,
                         "field": "anomaly_reported", "operator": "eq", "value": True,
                         "depends_on": [evidence_id]},
                    ]
                    previous = report_id
                else:
                    previous = inspect_id
            rows.append({
                "goal_id": "return_to_charging_station", "subject": "quadruped_1",
                "field": "location", "operator": "eq", "value": "charging_station",
                "depends_on": [previous] if previous else [],
            })
        return GoalSet([
            GoalCondition(
                goal_id=x["goal_id"], subject=x["subject"], field=x["field"],
                value=x["value"], operator=x.get("operator", "eq"),
                depends_on=tuple(x.get("depends_on") or ()),
            )
            for x in rows
        ])

    def build_coordination(self):
        raw = self.data.get("coordination") or {}
        return CoordinationSpec(
            maintain_until=[MaintainFactUntilGoal(**x) for x in raw.get("maintain_until") or []],
            device_rules=[DeviceRule(**x) for x in raw.get("device_rules") or []],
        )

    def build_tools(self):
        catalog = simulation_runtime_tools()
        return ToolCatalog([catalog.get(name) for name in self.data["tools"]])

    def build_executor(self, *, failure_probability=0.0, seed=0):
        if self.name == "inspection":
            return _InspectionExecutor(
                hidden_truth=self.data["hidden_truth"],
                failure_probability=failure_probability,
                seed=seed,
            )
        return FakeDispatchExecutor(failure_probability=failure_probability, seed=seed)

    def final_check(self, world: WorldState):
        for fact, expected in self.data["expected_final"].items():
            entity, field = fact.split(".", 1)
            current = world.get(entity, field)
            if current != expected:
                return False, f"{fact}={current!r}, expected={expected!r}"
        return True, "ok"


class _InspectionExecutor(FakeDispatchExecutor):
    def __init__(self, *, hidden_truth, failure_probability, seed):
        super().__init__(failure_probability=failure_probability, seed=seed)
        self.hidden_truth = dict(hidden_truth)

    def execute(self, dispatch, *, world, tools):
        updated, report = super().execute(dispatch, world=world, tools=tools)
        for result in report.results:
            action = result.bound_action.action
            if result.verified_success and action.function_name == "inspect":
                point = action.arguments["point_id"]
                updated.set(point, "health_state", self.hidden_truth[point])
        return updated, report


def get_scenario(name: str):
    canonical = ALIASES.get(name, name)
    if canonical not in _DATA:
        raise KeyError(f"Unknown simulation scenario: {name}")
    return SimulationScenario(canonical, dict(_DATA[canonical]))


def list_scenarios():
    return list(_DATA)
