"""
Runtime semantic goal definitions.

This module defines:
- GoalCondition: one desired semantic world-state condition
- GoalSet: a collection of goals and their dependencies

This module does NOT:
- call an LLM
- interpret natural language
- execute robot actions
- mutate WorldState
- decide which goal should be executed next

It only defines and evaluates task goals against the current WorldState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .world_state import WorldState


@dataclass(frozen=True)
class GoalCondition:
    """
    One semantic condition that should become true in WorldState.

    Example:
        GoalCondition(
            goal_id="G001",
            subject="saline_1",
            field="location",
            operator="eq",
            value="drawer_1",
        )

    means:
        saline_1.location == "drawer_1"
    """

    goal_id: str
    subject: str
    field: str
    value: Any
    operator: str = "eq"
    depends_on: tuple[str, ...] = field(
        default_factory=tuple
    )

    def is_satisfied(
        self,
        world: WorldState,
    ) -> bool:
        """
        Evaluate this goal against the current WorldState.
        """

        current = world.get(
            self.subject,
            self.field,
        )

        if self.operator == "eq":
            return current == self.value

        if self.operator == "ne":
            return current != self.value

        if self.operator == "in":
            return current in self.value

        raise ValueError(
            "Unsupported goal operator: "
            f"{self.operator}"
        )


class GoalSet:
    """
    Collection of semantic goals for one task.

    GoalSet is responsible for:
    - keeping goal IDs unique
    - validating dependency references
    - looking up goals by ID
    - evaluating physical satisfaction
    - reporting dependency satisfaction
    - producing a compact runtime summary

    Task credit / ordered milestone history belongs to the progress monitor,
    not to GoalSet.
    """

    def __init__(
        self,
        conditions: Iterable[GoalCondition],
    ):
        self.conditions = list(
            conditions
        )

        ids = [
            goal.goal_id
            for goal in self.conditions
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate goal_id"
            )

        known = set(
            ids
        )

        for goal in self.conditions:
            unknown = (
                set(goal.depends_on)
                - known
            )

            if unknown:
                raise ValueError(
                    f"Goal {goal.goal_id} "
                    "depends on unknown goals: "
                    f"{sorted(unknown)}"
                )

    def by_id(
        self,
        goal_id: str,
    ) -> GoalCondition:
        """
        Return one goal by its unique ID.
        """

        for goal in self.conditions:
            if goal.goal_id == goal_id:
                return goal

        raise KeyError(
            goal_id
        )

    def satisfied_ids(
        self,
        world: WorldState,
    ) -> set[str]:
        """
        Return IDs whose physical conditions are currently satisfied.
        """

        return {
            goal.goal_id
            for goal in self.conditions
            if goal.is_satisfied(
                world
            )
        }

    def is_satisfied(
        self,
        world: WorldState,
    ) -> bool:
        """
        Return True only when every goal condition is currently satisfied.
        """

        return all(
            goal.is_satisfied(
                world
            )
            for goal in self.conditions
        )

    def dependencies_satisfied(
        self,
        goal: GoalCondition,
        world: WorldState,
    ) -> bool:
        """
        Check whether all dependency goals are physically satisfied now.

        Ordered task-credit semantics are handled separately by
        progress_monitor.py.
        """

        satisfied = self.satisfied_ids(
            world
        )

        return set(
            goal.depends_on
        ).issubset(
            satisfied
        )

    def summary(
        self,
        world: WorldState,
    ) -> list[dict[str, Any]]:
        """
        Build a compact goal-state view for the runtime brain / debugging.
        """

        satisfied = self.satisfied_ids(
            world
        )

        rows: list[
            dict[str, Any]
        ] = []

        for goal in self.conditions:
            rows.append({
                "goal_id":
                    goal.goal_id,

                "subject":
                    goal.subject,

                "field":
                    goal.field,

                "operator":
                    goal.operator,

                "current":
                    world.get(
                        goal.subject,
                        goal.field,
                    ),

                "desired":
                    goal.value,

                "satisfied":
                    goal.goal_id
                    in satisfied,

                "depends_on":
                    list(
                        goal.depends_on
                    ),

                "dependencies_satisfied":
                    set(
                        goal.depends_on
                    ).issubset(
                        satisfied
                    ),
            })

        return rows


__all__ = [
    "GoalCondition",
    "GoalSet",
]
