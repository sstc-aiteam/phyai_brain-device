from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain.task import CoordinationSpec
from ..domain.task import GoalSet
from ..domain.tool import ToolCatalog
from ..domain.world import WorldState



@dataclass(frozen=True)
class PlannerContext:
    """
    Complete decision-time context passed from RuntimeEngine to the planner.

    This is a transport object only. It does not perform planning or mutate
    runtime state.
    """

    world: WorldState
    goals: GoalSet
    tools: ToolCatalog

    coordination: CoordinationSpec | None = None
    device_context: Any = None

    rejected_dispatches: list[dict[str, Any]] | None = None

    # Historical name retained for compatibility.
    # RuntimeEngine currently fills this with ExecutionMemory.brain_view().
    task_state: dict[str, Any] | None = None

    task_progress_state: dict[str, Any] | None = None
    repair_context: dict[str, Any] | None = None
    loop_context: dict[str, Any] | None = None

    max_parallel_actions: int = 2

    def as_kwargs(self) -> dict[str, Any]:
        """
        Convert to the legacy decide_dispatch() keyword interface.
        """

        return {
            "world": self.world,
            "goals": self.goals,
            "tools": self.tools,
            "coordination": self.coordination,
            "device_context": self.device_context,
            "rejected_dispatches": self.rejected_dispatches,
            "task_state": self.task_state,
            "task_progress_state": self.task_progress_state,
            "repair_context": self.repair_context,
            "loop_context": self.loop_context,
            "max_parallel_actions": self.max_parallel_actions,
        }

_COMMITMENT_FIELDS = (
    "holding",
    "carrying",
    "supporting",
    "grasping",
    "controlling",
)


def build_critical_state(
    world: WorldState,
    goals: GoalSet,
    coordination: CoordinationSpec,
) -> dict[str, Any]:
    """Extract per-device commitments and their relevant goal context.

    This is salience extraction, NOT action selection.
    """

    satisfied_ids = goals.satisfied_ids(world)
    commitments: list[dict[str, Any]] = []

    active_maintain = [
        c for c in coordination.maintain_until
        if c.is_active(world, goals)
    ]

    for resource_id in world.entity_ids():
        entity = world.entity(resource_id)

        for relation in _COMMITMENT_FIELDS:
            engaged = entity.get(relation)
            if not isinstance(engaged, str) or not engaged:
                continue

            related_goals = []
            for goal in goals.conditions:
                if goal.subject != engaged:
                    continue
                if goal.goal_id in satisfied_ids:
                    continue
                related_goals.append({
                    "goal_id": goal.goal_id,
                    "subject": goal.subject,
                    "field": goal.field,
                    "current": world.get(goal.subject, goal.field),
                    "desired": goal.value,
                    "depends_on": list(goal.depends_on),
                    "dependencies_satisfied": (
                        set(goal.depends_on).issubset(satisfied_ids)
                    ),
                })

            protected_by = [
                {
                    "subject": c.subject,
                    "field": c.field,
                    "value": c.value,
                    "until_goal": c.until_goal,
                }
                for c in active_maintain
                if c.subject == engaged
            ]

            commitments.append({
                "device_id": resource_id,
                "relation": relation,
                "engaged_entity_id": engaged,
                "engaged_entity_state": (
                    world.entity(engaged)
                    if world.has_entity(engaged)
                    else None
                ),
                "related_unsatisfied_goals": related_goals,
                "active_maintain_constraints": protected_by,
            })

    return {
        "has_active_commitment": bool(commitments),
        "resource_commitments": commitments,
        "active_maintain_constraints": [
            {
                "subject": c.subject,
                "field": c.field,
                "value": c.value,
                "until_goal": c.until_goal,
            }
            for c in active_maintain
        ],
        "note": (
            "Commitments are PER DEVICE, not a global lock. "
            "One committed arm does not block another free arm."
        ),
    }
