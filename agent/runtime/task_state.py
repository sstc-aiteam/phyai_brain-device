"""
Runtime TaskState.

TaskState is short-term execution memory for the closed-loop runtime.

Important rule:
    WorldState is authoritative for current physical reality.
    TaskState never overwrites WorldState.

TaskState only summarizes:
- which goals are completed / pending / regressed
- what the runtime is currently pursuing
- current blockers derived from the latest WorldState
- recent failure / rejection context
- the primary task target

Typical use:

    state = get_task_state(
        world=world,
        goals=goals,
        task_progress_state=progress_monitor.brain_view(goals, world),
        repair_context=repair_tracker.brain_view(world, iteration=iteration),
        loop_context=loop_monitor.brain_view(),
        current_dispatch=last_dispatch,
        rejected_dispatches=rejected,
    )

    state.as_dict()

No camera, service, LLM, robot, or executor call is performed here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _world_snapshot(
    world: Any,
) -> dict[str, Any]:
    if hasattr(
        world,
        "snapshot",
    ):
        value = world.snapshot()
        if isinstance(
            value,
            dict,
        ):
            return value

    if isinstance(
        world,
        dict,
    ):
        return deepcopy(
            world
        )

    raise TypeError(
        "world must provide snapshot() or be a dict"
    )


def _world_entities(
    world: Any,
) -> dict[str, dict[str, Any]]:
    snapshot = _world_snapshot(
        world
    )

    entities = snapshot.get(
        "entities",
        {},
    )

    if not isinstance(
        entities,
        dict,
    ):
        return {}

    return {
        str(entity_id):
            entity
        for entity_id, entity
        in entities.items()
        if isinstance(
            entity,
            dict,
        )
    }


def _goal_conditions(
    goals: Any,
) -> list[Any]:
    conditions = getattr(
        goals,
        "conditions",
        None,
    )

    if conditions is None:
        return []

    return list(
        conditions
    )


def _goal_id(
    goal: Any,
) -> str:
    return str(
        getattr(
            goal,
            "goal_id",
            "",
        )
    )


def _goal_subject(
    goal: Any,
) -> str:
    return str(
        getattr(
            goal,
            "subject",
            "",
        )
    )


def _goal_field(
    goal: Any,
) -> str:
    return str(
        getattr(
            goal,
            "field",
            "",
        )
    )


def _goal_operator(
    goal: Any,
) -> str:
    return str(
        getattr(
            goal,
            "operator",
            getattr(
                goal,
                "op",
                "eq",
            ),
        )
    )


def _goal_value(
    goal: Any,
) -> Any:
    for name in (
        "value",
        "target",
        "expected",
        "required",
    ):
        if hasattr(
            goal,
            name,
        ):
            return deepcopy(
                getattr(
                    goal,
                    name,
                )
            )

    return None


def _goal_depends_on(
    goal: Any,
) -> list[str]:
    value = getattr(
        goal,
        "depends_on",
        (),
    )

    if value is None:
        return []

    return [
        str(item)
        for item in value
    ]


def _goal_is_satisfied(
    goal: Any,
    world: Any,
) -> bool:
    method = getattr(
        goal,
        "is_satisfied",
        None,
    )

    if callable(
        method
    ):
        try:
            return bool(
                method(
                    world
                )
            )
        except Exception:
            return False

    return False


def _current_world_value(
    entities: dict[str, dict[str, Any]],
    *,
    subject: str,
    field_name: str,
) -> Any:
    entity = entities.get(
        subject
    )

    if not isinstance(
        entity,
        dict,
    ):
        return None

    return deepcopy(
        entity.get(
            field_name
        )
    )


def _goal_row(
    goal: Any,
    *,
    entities: dict[str, dict[str, Any]],
    world: Any,
    credited: set[str],
) -> dict[str, Any]:
    goal_id = _goal_id(
        goal
    )

    subject = _goal_subject(
        goal
    )

    field_name = _goal_field(
        goal
    )

    physically_satisfied = (
        _goal_is_satisfied(
            goal,
            world,
        )
    )

    credited_now = (
        goal_id in credited
    )

    return {
        "goal_id":
            goal_id,
        "subject":
            subject,
        "field":
            field_name,
        "operator":
            _goal_operator(
                goal
            ),
        "target":
            _goal_value(
                goal
            ),
        "current":
            _current_world_value(
                entities,
                subject=
                    subject,
                field_name=
                    field_name,
            ),
        "depends_on":
            _goal_depends_on(
                goal
            ),
        "physically_satisfied":
            physically_satisfied,
        "credited":
            credited_now,
    }


def _extract_action(
    dispatch: Any,
) -> dict[str, Any] | None:
    if dispatch is None:
        return None

    if hasattr(
        dispatch,
        "as_dict",
    ):
        try:
            dispatch = dispatch.as_dict()
        except Exception:
            return None

    if not isinstance(
        dispatch,
        dict,
    ):
        return None

    actions = dispatch.get(
        "actions",
        [],
    )

    if not isinstance(
        actions,
        list,
    ) or not actions:
        return None

    action = actions[0]

    if not isinstance(
        action,
        dict,
    ):
        return None

    return deepcopy(
        action
    )


def _extract_failures(
    *,
    task_progress_state: dict[str, Any] | None,
    repair_context: dict[str, Any] | None,
    rejected_dispatches: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    failures: list[
        dict[str, Any]
    ] = []

    if isinstance(
        task_progress_state,
        dict,
    ):
        last_feedback = (
            task_progress_state.get(
                "last_execution_feedback"
            )
        )

        if isinstance(
            last_feedback,
            dict,
        ):
            failed_count = (
                last_feedback.get(
                    "failed_action_count",
                    0,
                )
            )

            if failed_count:
                failures.append({
                    "source":
                        "execution",
                    "type":
                        "execution_failure",
                    "failed_action_count":
                        failed_count,
                    "progress_class":
                        last_feedback.get(
                            "progress_class"
                        ),
                })

    if rejected_dispatches:
        rows = list(
            rejected_dispatches
        )

        for row in rows[-5:]:
            if not isinstance(
                row,
                dict,
            ):
                continue

            failures.append({
                "source":
                    "decision_validation",
                "type":
                    (
                        row.get(
                            "structured_feedback",
                            {},
                        )
                        .get(
                            "rejection_type"
                        )
                        if isinstance(
                            row.get(
                                "structured_feedback"
                            ),
                            dict,
                        )
                        else None
                    ),
                "reason":
                    row.get(
                        "reason"
                    ),
                "dispatch":
                    deepcopy(
                        row.get(
                            "dispatch"
                        )
                    ),
            })

    if isinstance(
        repair_context,
        dict,
    ):
        # Keep repair context available without assuming one exact schema.
        pending_intent = (
            repair_context.get(
                "pending_intent"
            )
        )

        if pending_intent:
            failures.append({
                "source":
                    "repair_context",
                "type":
                    "pending_intent",
                "details":
                    deepcopy(
                        pending_intent
                    ),
            })

    return failures[-8:]


def _entity_pickable(
    entity: dict[str, Any],
) -> bool:
    value = entity.get(
        "pickable"
    )

    if value is True:
        return True

    grasp = entity.get(
        "grasp"
    )

    return (
        isinstance(
            grasp,
            dict,
        )
        and grasp.get(
            "pickable"
        )
        is True
    )


def _blocked_info(
    entity: dict[str, Any],
) -> tuple[bool, list[str]]:
    grasp = entity.get(
        "grasp"
    )

    if isinstance(
        grasp,
        dict,
    ):
        blocked = bool(
            grasp.get(
                "blocked",
                False,
            )
        )

        blocked_by = (
            grasp.get(
                "blocked_by"
            )
            or []
        )
    else:
        blocked = bool(
            entity.get(
                "blocked",
                False,
            )
        )

        blocked_by = (
            entity.get(
                "blocked_by"
            )
            or []
        )

    return (
        blocked,
        [
            str(item)
            for item
            in blocked_by
            if isinstance(
                item,
                str,
            )
            and item
        ],
    )


def _build_blocking_conditions(
    *,
    pending_goals: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    blockers: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[Any, ...]
    ] = set()

    def add(
        row: dict[str, Any],
        key: tuple[Any, ...],
    ) -> None:
        if key in seen:
            return

        seen.add(
            key
        )

        blockers.append(
            row
        )

    for goal in pending_goals:
        goal_id = str(
            goal.get(
                "goal_id",
                "",
            )
        )

        subject = str(
            goal.get(
                "subject",
                "",
            )
        )

        field_name = str(
            goal.get(
                "field",
                "",
            )
        )

        target = goal.get(
            "target"
        )

        subject_entity = (
            entities.get(
                subject
            )
        )

        if subject and subject_entity is None:
            add(
                {
                    "goal_id":
                        goal_id,
                    "subject":
                        subject,
                    "type":
                        "target_not_observed",
                },
                (
                    goal_id,
                    subject,
                    "target_not_observed",
                ),
            )

            continue

        if isinstance(
            subject_entity,
            dict,
        ):
            blocked, blocked_by = (
                _blocked_info(
                    subject_entity
                )
            )

            if blocked:
                add(
                    {
                        "goal_id":
                            goal_id,
                        "subject":
                            subject,
                        "type":
                            "occluded",
                        "blocked_by":
                            blocked_by,
                        "pickable_blockers": [
                            blocker_id
                            for blocker_id
                            in blocked_by
                            if (
                                blocker_id
                                in entities
                                and _entity_pickable(
                                    entities[
                                        blocker_id
                                    ]
                                )
                            )
                        ],
                    },
                    (
                        goal_id,
                        subject,
                        "occluded",
                        tuple(
                            blocked_by
                        ),
                    ),
                )

            if (
                subject_entity.get(
                    "pickable"
                )
                is False
                and not blocked
                and subject_entity.get(
                    "type"
                )
                == "object"
            ):
                add(
                    {
                        "goal_id":
                            goal_id,
                        "subject":
                            subject,
                        "type":
                            "not_pickable",
                    },
                    (
                        goal_id,
                        subject,
                        "not_pickable",
                    ),
                )

        # For object.location == destination, the destination can impose a
        # physical prerequisite such as container open_state == open.
        if (
            field_name
            == "location"
            and isinstance(
                target,
                str,
            )
            and target
        ):
            destination = (
                entities.get(
                    target
                )
            )

            if isinstance(
                destination,
                dict,
            ):
                open_state = (
                    destination.get(
                        "open_state"
                    )
                )

                if open_state == "closed":
                    add(
                        {
                            "goal_id":
                                goal_id,
                            "subject":
                                subject,
                            "type":
                                "destination_closed",
                            "destination_id":
                                target,
                            "open_state":
                                open_state,
                        },
                        (
                            goal_id,
                            target,
                            "destination_closed",
                        ),
                    )

    return blockers


def _select_target_object(
    *,
    pending_goals: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
) -> str | None:
    # Prefer pending goal subjects that are physical objects.
    for goal in pending_goals:
        subject = goal.get(
            "subject"
        )

        entity = entities.get(
            subject
        )

        if (
            isinstance(
                entity,
                dict,
            )
            and entity.get(
                "type"
            )
            == "object"
        ):
            return str(
                subject
            )

    # Fall back to any pending goal subject.
    for goal in pending_goals:
        subject = goal.get(
            "subject"
        )

        if isinstance(
            subject,
            str,
        ) and subject:
            return subject

    return None


def _current_step(
    *,
    target_object: str | None,
    pending_goals: list[dict[str, Any]],
    blocking_conditions: list[dict[str, Any]],
    current_dispatch: Any,
    repair_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    action = _extract_action(
        current_dispatch
    )

    if action is not None:
        return {
            "phase":
                "executing",
            "action":
                action,
        }

    if isinstance(
        repair_context,
        dict,
    ):
        pending_intent = (
            repair_context.get(
                "pending_intent"
            )
        )

        if pending_intent:
            return {
                "phase":
                    "repair",
                "intent":
                    deepcopy(
                        pending_intent
                    ),
            }

    if blocking_conditions:
        first = blocking_conditions[
            0
        ]

        return {
            "phase":
                "blocked",
            "target":
                first.get(
                    "subject",
                    target_object,
                ),
            "condition":
                deepcopy(
                    first
                ),
        }

    if pending_goals:
        return {
            "phase":
                "pursuing_goal",
            "target":
                target_object,
            "goal_id":
                pending_goals[0].get(
                    "goal_id"
                ),
        }

    return None


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------

@dataclass
class TaskState:
    status: str
    current_step: dict[str, Any] | None
    completed: list[dict[str, Any]] = field(
        default_factory=list
    )
    pending: list[dict[str, Any]] = field(
        default_factory=list
    )
    regressed: list[dict[str, Any]] = field(
        default_factory=list
    )
    failures: list[dict[str, Any]] = field(
        default_factory=list
    )
    target_object: str | None = None
    blocking_conditions: list[dict[str, Any]] = field(
        default_factory=list
    )
    last_action: dict[str, Any] | None = None
    loop_context: dict[str, Any] | None = None

    def as_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "status":
                self.status,
            "current_step":
                deepcopy(
                    self.current_step
                ),
            "completed":
                deepcopy(
                    self.completed
                ),
            "pending":
                deepcopy(
                    self.pending
                ),
            "regressed":
                deepcopy(
                    self.regressed
                ),
            "failures":
                deepcopy(
                    self.failures
                ),
            "target_object":
                self.target_object,
            "blocking_conditions":
                deepcopy(
                    self.blocking_conditions
                ),
            "last_action":
                deepcopy(
                    self.last_action
                ),
            "loop_context":
                deepcopy(
                    self.loop_context
                ),
        }


def get_task_state(
    *,
    world: Any,
    goals: Any,
    task_progress_state: dict[str, Any] | None = None,
    repair_context: dict[str, Any] | None = None,
    loop_context: dict[str, Any] | None = None,
    current_dispatch: Any = None,
    last_dispatch: Any = None,
    rejected_dispatches: Iterable[dict[str, Any]] | None = None,
) -> TaskState:
    """
    Build current short-term TaskState from authoritative runtime inputs.

    Reality-first semantics:
    - Fresh WorldState decides whether a goal is CURRENTLY physically true.
    - Historical credit alone never makes a currently-false goal "completed".
    - A previously completed goal that is no longer true becomes "regressed".
    - TaskState does not write anything back to WorldState.

    This means that if the runtime remembers a successful pick, but a fresh
    observation shows that the object is still at the original location, the
    current WorldState wins and the corresponding goal remains pending /
    regressed.
    """

    entities = _world_entities(
        world
    )

    progress_state_available = isinstance(
        task_progress_state,
        dict,
    )

    credited = set()

    if progress_state_available:
        credited = {
            str(item)
            for item
            in (
                task_progress_state.get(
                    "credited_goals"
                )
                or []
            )
        }

    rows = [
        _goal_row(
            goal,
            entities=
                entities,
            world=
                world,
            credited=
                credited,
        )
        for goal
        in _goal_conditions(
            goals
        )
    ]

    completed = []
    pending = []
    regressed = []

    for row in rows:
        physically_satisfied = bool(
            row.get(
                "physically_satisfied"
            )
        )

        was_credited = bool(
            row.get(
                "credited"
            )
        )

        # Reality wins:
        # credited history cannot keep a physically false goal completed.
        if (
            physically_satisfied
            and (
                was_credited
                or not progress_state_available
            )
        ):
            completed.append(
                row
            )
            continue

        pending.append(
            row
        )

        if (
            was_credited
            and not physically_satisfied
        ):
            regressed.append({
                **deepcopy(
                    row
                ),
                "reason":
                    "fresh_world_state_disagrees_with_previous_credit",
            })

    blocking_conditions = (
        _build_blocking_conditions(
            pending_goals=
                pending,
            entities=
                entities,
        )
    )

    target_object = (
        _select_target_object(
            pending_goals=
                pending,
            entities=
                entities,
        )
    )

    failures = (
        _extract_failures(
            task_progress_state=
                task_progress_state,
            repair_context=
                repair_context,
            rejected_dispatches=
                rejected_dispatches,
        )
    )

    progress_complete = (
        isinstance(
            task_progress_state,
            dict,
        )
        and task_progress_state.get(
            "task_complete"
        )
        is True
    )

    physically_complete = (
        bool(
            rows
        )
        and all(
            bool(
                row.get(
                    "physically_satisfied"
                )
            )
            for row in rows
        )
    )

    if (
        progress_complete
        and physically_complete
        and not regressed
    ):
        status = "completed"
    elif regressed:
        status = "running"
    elif failures:
        status = "running"
    else:
        status = "running"

    step = (
        _current_step(
            target_object=
                target_object,
            pending_goals=
                pending,
            blocking_conditions=
                blocking_conditions,
            current_dispatch=
                current_dispatch,
            repair_context=
                repair_context,
        )
    )

    return TaskState(
        status=status,
        current_step=step,
        completed=completed,
        pending=pending,
        regressed=regressed,
        failures=failures,
        target_object=target_object,
        blocking_conditions=
            blocking_conditions,
        last_action=
            _extract_action(
                last_dispatch
            ),
        loop_context=(
            deepcopy(
                loop_context
            )
            if isinstance(
                loop_context,
                dict,
            )
            else None
        ),
    )


__all__ = [
    "TaskState",
    "get_task_state",
]
