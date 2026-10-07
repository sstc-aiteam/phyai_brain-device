"""
Runtime dispatch validator.

The validator is the final deterministic gate before physical execution.

It answers only:

    "May this already-selected, already-bound dispatch start NOW?"

It does NOT:
- choose the next task action
- choose a device
- call an LLM
- execute robot commands
- treat ToolSpec effects as observed truth

Tool effects are used here only for conflict / constraint prediction.
Canonical WorldState is updated only by fresh observation after execution.
"""

from __future__ import annotations

from typing import Any

from .coordination import CoordinationSpec
from .dispatch import (
    BoundAction,
    BoundDispatch,
)
from .goals import GoalSet
from .tool_model import (
    ToolCatalog,
    ToolValidationError,
)
from .world_state import WorldState


class RuntimeValidationError(RuntimeError):
    """
    Hard runtime rejection with optional machine-readable feedback.
    """

    def __init__(
        self,
        message: str,
        *,
        feedback: dict[str, Any] | None = None,
    ):
        super().__init__(
            message
        )
        self.feedback = (
            feedback
            or {}
        )


def _reject(
    message: str,
    *,
    rejection_type: str,
    **feedback: Any,
) -> None:
    raise RuntimeValidationError(
        message,
        feedback={
            "rejection_type":
                rejection_type,
            **feedback,
        },
    )


def _validate_required_device_rules(
    bound: BoundAction,
    coordination: CoordinationSpec,
) -> None:
    """
    Device binding normally handles DeviceRule.

    This is a defensive final check so a required rule cannot be bypassed by
    an incorrect/custom binder.
    """

    action = bound.action

    required_devices = {
        rule.device_id
        for rule in coordination.device_rules
        if (
            rule.mode == "required"
            and rule.device_id
            and rule.matches(
                action.function_name,
                action.arguments,
            )
        )
    }

    if len(
        required_devices
    ) > 1:
        _reject(
            "conflicting required device rules",
            rejection_type=
                "COORDINATION_DEVICE_CONFLICT",
            attempted_action=
                action.as_dict(),
            required_devices=
                sorted(
                    required_devices
                ),
        )

    if (
        required_devices
        and bound.device_id
        not in required_devices
    ):
        _reject(
            f"{action.function_name} must use "
            f"{next(iter(required_devices))}, "
            f"not {bound.device_id}",
            rejection_type=
                "COORDINATION_DEVICE_REQUIRED",
            attempted_action=
                action.as_dict(),
            bound_device_id=
                bound.device_id,
            required_devices=
                sorted(
                    required_devices
                ),
        )


def _validate_action_gates(
    bound: BoundAction,
    *,
    world: WorldState,
    goals: GoalSet,
    coordination: CoordinationSpec,
) -> None:
    """
    Validate goal-state gates attached to this action.
    """

    satisfied = goals.satisfied_ids(
        world
    )

    for gate in coordination.action_gates:
        if not gate.matches(
            bound.action
        ):
            continue

        missing_required = (
            set(
                gate.requires_goals
            )
            - satisfied
        )

        still_satisfied_but_required_unsatisfied = (
            set(
                gate.requires_unsatisfied_goals
            )
            & satisfied
        )

        if missing_required:
            _reject(
                "action gate requirements are not satisfied",
                rejection_type=
                    "COORDINATION_ACTION_GATE",
                attempted_action=
                    bound.action.as_dict(),
                device_id=
                    bound.device_id,
                missing_required_goals=
                    sorted(
                        missing_required
                    ),
            )

        if still_satisfied_but_required_unsatisfied:
            _reject(
                "action gate requires goals to remain unsatisfied",
                rejection_type=
                    "COORDINATION_ACTION_GATE",
                attempted_action=
                    bound.action.as_dict(),
                device_id=
                    bound.device_id,
                goals_that_must_be_unsatisfied=
                    sorted(
                        still_satisfied_but_required_unsatisfied
                    ),
            )


def _validate_maintain_constraints(
    bound: BoundAction,
    *,
    world: WorldState,
    goals: GoalSet,
    tools: ToolCatalog,
    coordination: CoordinationSpec,
) -> None:
    """
    Reject an action whose declared semantic effects would break an ACTIVE
    MaintainFactUntilGoal constraint.

    Effects are predictive metadata here only.  `predicted` is never promoted
    to canonical WorldState.
    """

    active_constraints = [
        constraint
        for constraint
        in coordination.maintain_until
        if constraint.is_active(
            world,
            goals,
        )
    ]

    if not active_constraints:
        return

    tool = tools.get(
        bound.action.function_name
    )

    write_facts = tool.write_facts(
        bound
    )

    relevant = [
        constraint
        for constraint
        in active_constraints
        if (
            constraint.subject,
            constraint.field,
        )
        in write_facts
    ]

    if not relevant:
        return

    predicted = tool.apply_effects(
        world,
        bound,
    )

    for constraint in relevant:
        predicted_value = predicted.get(
            constraint.subject,
            constraint.field,
        )

        if (
            predicted_value
            != constraint.value
        ):
            _reject(
                "action would violate active maintain constraint",
                rejection_type=
                    "COORDINATION_MAINTAIN_VIOLATION",
                attempted_action=
                    bound.action.as_dict(),
                device_id=
                    bound.device_id,
                constraint={
                    "subject":
                        constraint.subject,
                    "field":
                        constraint.field,
                    "required_value":
                        constraint.value,
                    "until_goal":
                        constraint.until_goal,
                    "current_value":
                        world.get(
                            constraint.subject,
                            constraint.field,
                        ),
                    "predicted_value":
                        predicted_value,
                },
            )



_NUDGE_RECOVERY_REASONS = {
    "missing_grasp_point",
    "unreachable",
    "out_of_workspace",
    "no_reachable_arm",
    "grasp_unavailable",
    "perception_uncertain",
    "low_confidence",
}


def _validate_nudge_relevance(
    bound: BoundAction,
    *,
    world: WorldState,
    goals: GoalSet,
) -> None:
    """
    Keep nudge_arm available, but only as a recovery primitive.

    A nudge is accepted when an UNSATISFIED goal has concrete evidence that
    a small arm adjustment may help, for example:
      - the goal directly concerns the bound arm's pose/location
      - the pending target has a grasp/reachability recovery reason
      - the pending target is occluded, but there is no known pickable blocker

    If a target is occluded by a known pickable object, moving that blocker is
    a more direct semantic recovery than nudging the arm, so nudge_arm is not
    accepted for that condition.

    This gate intentionally does not choose the replacement action. It only
    rejects an irrelevant escape action so RuntimeBrain must decide again.
    """

    action = bound.action

    if action.function_name != "nudge_arm":
        return

    pending_goals = [
        goal
        for goal in goals.conditions
        if not goal.is_satisfied(
            world
        )
    ]

    pending_goal_ids = [
        goal.goal_id
        for goal in pending_goals
    ]

    recovery_evidence: list[
        dict[str, Any]
    ] = []

    direct_blockers: list[
        dict[str, Any]
    ] = []

    for goal in pending_goals:
        # A task that directly targets arm pose/location may legitimately use
        # a Cartesian nudge.
        if (
            goal.subject
            == bound.device_id
            and goal.field
            in {
                "pose",
                "location",
            }
        ):
            recovery_evidence.append({
                "goal_id":
                    goal.goal_id,
                "subject":
                    goal.subject,
                "reason":
                    "arm_motion_goal",
            })
            continue

        if not world.has_entity(
            goal.subject
        ):
            continue

        entity = world.entity(
            goal.subject
        )

        if not isinstance(
            entity,
            dict,
        ):
            continue

        grasp = entity.get(
            "grasp"
        )

        if not isinstance(
            grasp,
            dict,
        ):
            grasp = {}

        reasons = {
            str(
                value
            )
            for value
            in (
                grasp.get(
                    "reasons"
                )
                or []
            )
        }

        matched_reasons = sorted(
            reasons
            & _NUDGE_RECOVERY_REASONS
        )

        if matched_reasons:
            recovery_evidence.append({
                "goal_id":
                    goal.goal_id,
                "subject":
                    goal.subject,
                "reason":
                    "grasp_or_reachability_recovery",
                "details":
                    matched_reasons,
            })

        blocked = bool(
            grasp.get(
                "blocked",
                entity.get(
                    "blocked",
                    False,
                ),
            )
        )

        blocked_by = (
            grasp.get(
                "blocked_by"
            )
            or entity.get(
                "blocked_by"
            )
            or []
        )

        blocked_by = [
            blocker_id
            for blocker_id in blocked_by
            if isinstance(
                blocker_id,
                str,
            )
            and blocker_id
        ]

        if not blocked:
            continue

        pickable_blockers = [
            blocker_id
            for blocker_id in blocked_by
            if (
                world.has_entity(
                    blocker_id
                )
                and world.get(
                    blocker_id,
                    "pickable",
                    False,
                )
                is True
            )
        ]

        if pickable_blockers:
            direct_blockers.append({
                "goal_id":
                    goal.goal_id,
                "subject":
                    goal.subject,
                "pickable_blockers":
                    pickable_blockers,
            })
            continue

        # Occlusion exists but there is no known directly-manipulable blocker.
        # A small viewpoint/reachability adjustment can be a legitimate
        # recovery attempt.
        recovery_evidence.append({
            "goal_id":
                goal.goal_id,
            "subject":
                goal.subject,
            "reason":
                "occluded_without_pickable_blocker",
            "blocked_by":
                blocked_by,
        })

    if recovery_evidence:
        return

    _reject(
        "nudge_arm is recovery-only and no pending goal currently "
        "provides a valid recovery reason",
        rejection_type=
            "ACTION_NOT_RELEVANT",
        attempted_action=
            action.as_dict(),
        device_id=
            bound.device_id,
        pending_goal_ids=
            pending_goal_ids,
        direct_pickable_blockers=
            direct_blockers,
        guidance=(
            "Choose an action that directly advances a pending goal, "
            "satisfies a prerequisite, or removes a known blocker."
        ),
    )


def _validate_one_action(
    bound: BoundAction,
    *,
    world: WorldState,
    goals: GoalSet,
    tools: ToolCatalog,
    coordination: CoordinationSpec,
) -> None:
    """
    Validate one BoundAction against current world and coordination.
    """

    if not isinstance(
        bound,
        BoundAction,
    ):
        raise TypeError(
            "bound must be BoundAction"
        )

    tool = tools.get(
        bound.action.function_name
    )

    try:
        tool.validate(
            bound,
            world,
        )

    except ToolValidationError as exc:
        feedback = dict(
            getattr(
                exc,
                "feedback",
                {},
            )
            or {}
        )

        feedback.setdefault(
            "rejection_type",
            "TOOL_VALIDATION_FAILED",
        )

        feedback.setdefault(
            "attempted_action",
            bound.action.as_dict(),
        )

        feedback.setdefault(
            "device_id",
            bound.device_id,
        )

        raise RuntimeValidationError(
            str(
                exc
            ),
            feedback=
                feedback,
        ) from exc

    _validate_nudge_relevance(
        bound,
        world=world,
        goals=goals,
    )

    _validate_required_device_rules(
        bound,
        coordination,
    )

    _validate_action_gates(
        bound,
        world=world,
        goals=goals,
        coordination=coordination,
    )

    _validate_maintain_constraints(
        bound,
        world=world,
        goals=goals,
        tools=tools,
        coordination=coordination,
    )


def _validate_distinct_devices(
    dispatch: BoundDispatch,
) -> None:
    """
    One concurrent dispatch cannot start two atomic actions on one device.
    """

    devices = [
        bound.device_id
        for bound
        in dispatch.actions
    ]

    if len(
        devices
    ) != len(
        set(
            devices
        )
    ):
        _reject(
            "concurrent dispatch assigns more than one action "
            "to the same device",
            rejection_type=
                "PARALLEL_DEVICE_CONFLICT",
            device_ids=
                devices,
        )


def _validate_exclusive_entities(
    dispatch: BoundDispatch,
    *,
    tools: ToolCatalog,
) -> None:
    """
    Detect two concurrent actions claiming the same exclusive entity.
    """

    owners: dict[
        str,
        list[int],
    ] = {}

    for index, bound in enumerate(
        dispatch.actions
    ):
        tool = tools.get(
            bound.action.function_name
        )

        for entity_id in tool.exclusive_entities(
            bound.action
        ):
            owners.setdefault(
                entity_id,
                [],
            ).append(
                index
            )

    conflicts = {
        entity_id:
            indexes
        for entity_id, indexes
        in owners.items()
        if len(
            indexes
        ) > 1
    }

    if conflicts:
        _reject(
            "concurrent actions claim the same exclusive entity",
            rejection_type=
                "PARALLEL_ENTITY_CONFLICT",
            conflicts=
                conflicts,
        )


def _capacity_value(
    world: WorldState,
    entity_id: str,
    capacity_field: str,
) -> int:
    """
    Missing/invalid capacity is conservatively treated as 1.
    """

    raw = world.get(
        entity_id,
        capacity_field,
        1,
    )

    try:
        value = int(
            raw
        )
    except (
        TypeError,
        ValueError,
    ):
        return 1

    return max(
        1,
        value,
    )


def _validate_capacity_claims(
    dispatch: BoundDispatch,
    *,
    world: WorldState,
    tools: ToolCatalog,
) -> None:
    """
    Enforce ToolSpec capacity_limited_entity_args across parallel actions.
    """

    claims: dict[
        tuple[str, str],
        int,
    ] = {}

    for bound in dispatch.actions:
        tool = tools.get(
            bound.action.function_name
        )

        for (
            entity_id,
            capacity_field,
        ) in tool.capacity_limited_entities(
            bound.action
        ):
            key = (
                entity_id,
                capacity_field,
            )

            claims[key] = (
                claims.get(
                    key,
                    0,
                )
                + 1
            )

    violations = []

    for (
        entity_id,
        capacity_field,
    ), count in claims.items():

        capacity = _capacity_value(
            world,
            entity_id,
            capacity_field,
        )

        if count > capacity:
            violations.append({
                "entity_id":
                    entity_id,
                "capacity_field":
                    capacity_field,
                "requested":
                    count,
                "capacity":
                    capacity,
            })

    if violations:
        _reject(
            "parallel entity capacity exceeded",
            rejection_type=
                "PARALLEL_CAPACITY_EXCEEDED",
            violations=
                violations,
        )


def _validate_read_write_conflicts(
    dispatch: BoundDispatch,
    *,
    tools: ToolCatalog,
) -> None:
    """
    Concurrent actions must not have semantic read/write or write/write races.

    Read/read overlap is allowed.
    """

    rows = []

    for bound in dispatch.actions:
        tool = tools.get(
            bound.action.function_name
        )

        rows.append({
            "bound":
                bound,
            "reads":
                tool.read_facts(
                    bound
                ),
            "writes":
                tool.write_facts(
                    bound
                ),
        })

    conflicts = []

    for left_index in range(
        len(
            rows
        )
    ):
        for right_index in range(
            left_index + 1,
            len(
                rows
            )
        ):
            left = rows[
                left_index
            ]

            right = rows[
                right_index
            ]

            left_write_right_read = (
                left["writes"]
                & right["reads"]
            )

            right_write_left_read = (
                right["writes"]
                & left["reads"]
            )

            write_write = (
                left["writes"]
                & right["writes"]
            )

            conflict_facts = (
                left_write_right_read
                | right_write_left_read
                | write_write
            )

            if not conflict_facts:
                continue

            conflicts.append({
                "left_action":
                    left["bound"].action.as_dict(),
                "left_device_id":
                    left["bound"].device_id,
                "right_action":
                    right["bound"].action.as_dict(),
                "right_device_id":
                    right["bound"].device_id,
                "facts": [
                    {
                        "subject":
                            subject,
                        "field":
                            field,
                    }
                    for subject, field
                    in sorted(
                        conflict_facts
                    )
                ],
            })

    if conflicts:
        _reject(
            "parallel semantic read/write conflict",
            rejection_type=
                "PARALLEL_STATE_CONFLICT",
            conflicts=
                conflicts,
        )


class RuntimeValidator:
    """
    Deterministic final validation stage before Executor.
    """

    def validate(
        self,
        dispatch: BoundDispatch,
        *,
        world: WorldState,
        goals: GoalSet,
        tools: ToolCatalog,
        coordination: CoordinationSpec | None = None,
    ) -> None:
        if not isinstance(
            dispatch,
            BoundDispatch,
        ):
            raise TypeError(
                "dispatch must be BoundDispatch"
            )

        if not isinstance(
            world,
            WorldState,
        ):
            raise TypeError(
                "world must be WorldState"
            )

        if not isinstance(
            goals,
            GoalSet,
        ):
            raise TypeError(
                "goals must be GoalSet"
            )

        if not isinstance(
            tools,
            ToolCatalog,
        ):
            raise TypeError(
                "tools must be ToolCatalog"
            )

        if coordination is None:
            coordination = (
                CoordinationSpec()
            )

        if not isinstance(
            coordination,
            CoordinationSpec,
        ):
            raise TypeError(
                "coordination must be CoordinationSpec"
            )

        # Each action must be legal against the SAME CURRENT world.
        for bound in dispatch.actions:
            _validate_one_action(
                bound,
                world=world,
                goals=goals,
                tools=tools,
                coordination=
                    coordination,
            )

        # Parallel dispatch constraints.
        if len(
            dispatch.actions
        ) > 1:
            _validate_distinct_devices(
                dispatch
            )

            _validate_exclusive_entities(
                dispatch,
                tools=tools,
            )

            _validate_capacity_claims(
                dispatch,
                world=world,
                tools=tools,
            )

            _validate_read_write_conflicts(
                dispatch,
                tools=tools,
            )


def validate_dispatch(
    dispatch: BoundDispatch,
    *,
    world: WorldState,
    goals: GoalSet,
    tools: ToolCatalog,
    coordination: CoordinationSpec | None = None,
) -> None:
    """
    Functional convenience wrapper.
    """

    RuntimeValidator().validate(
        dispatch,
        world=world,
        goals=goals,
        tools=tools,
        coordination=coordination,
    )


__all__ = [
    "RuntimeValidationError",
    "RuntimeValidator",
    "validate_dispatch",
]
