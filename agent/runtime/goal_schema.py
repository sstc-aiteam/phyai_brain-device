from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .world_state import WorldState


@dataclass(frozen=True)
class GoalCondition:
    goal_id: str
    subject: str
    field: str
    value: Any
    operator: str = "eq"
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def is_satisfied(self, world: WorldState) -> bool:
        current = world.get(self.subject, self.field)
        if self.operator == "eq":
            return current == self.value
        if self.operator == "ne":
            return current != self.value
        if self.operator == "in":
            return current in self.value
        raise ValueError(f"Unsupported goal operator: {self.operator}")


class GoalSet:
    def __init__(self, conditions: Iterable[GoalCondition]):
        self.conditions = list(conditions)
        ids = [g.goal_id for g in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate goal_id")
        known = set(ids)
        for goal in self.conditions:
            unknown = set(goal.depends_on) - known
            if unknown:
                raise ValueError(
                    f"Goal {goal.goal_id} depends on unknown goals: "
                    f"{sorted(unknown)}"
                )

    def by_id(self, goal_id: str) -> GoalCondition:
        for goal in self.conditions:
            if goal.goal_id == goal_id:
                return goal
        raise KeyError(goal_id)

    def satisfied_ids(self, world: WorldState) -> set[str]:
        return {
            goal.goal_id
            for goal in self.conditions
            if goal.is_satisfied(world)
        }

    def is_satisfied(self, world: WorldState) -> bool:
        return all(g.is_satisfied(world) for g in self.conditions)

    def dependencies_satisfied(
        self,
        goal: GoalCondition,
        world: WorldState,
    ) -> bool:
        satisfied = self.satisfied_ids(world)
        return set(goal.depends_on).issubset(satisfied)

    def summary(self, world: WorldState) -> list[dict[str, Any]]:
        satisfied = self.satisfied_ids(world)
        rows: list[dict[str, Any]] = []
        for goal in self.conditions:
            rows.append({
                "goal_id": goal.goal_id,
                "subject": goal.subject,
                "field": goal.field,
                "operator": goal.operator,
                "current": world.get(goal.subject, goal.field),
                "desired": goal.value,
                "satisfied": goal.goal_id in satisfied,
                "depends_on": list(goal.depends_on),
                "dependencies_satisfied": (
                    set(goal.depends_on).issubset(satisfied)
                ),
            })
        return rows
