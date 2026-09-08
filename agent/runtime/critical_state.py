from __future__ import annotations

from typing import Any

from .coordination_schema import CoordinationSpec
from .goal_schema import GoalSet
from .world_state import WorldState


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
