"""
Runtime device binding.
 assigns concrete runtime devices to semantic ActionIntent objects.

Input:
    DispatchDecision
    + current WorldState
    + ToolCatalog
    + CoordinationSpec
Output:
    BoundDispatch
 decides WHO performs it.

"""

from __future__ import annotations

from copy import deepcopy
from ...domain.task import TaskSpec, DeviceRule

from typing import Any, Protocol

from ...domain.task import CoordinationSpec
from ...domain.action import (
    ActionIntent,
    BoundAction,
    BoundDispatch,
    DispatchDecision,
)
from ...domain.tool import (
    ToolCatalog,
    ToolSpec,
)
from ...domain.world import WorldState


class DeviceBindingError(RuntimeError):
    """Hard binding/feasibility error with optional structured feedback."""

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


class DeviceBinder(Protocol):
    """
    Common runtime binder contract.
    """

    def for_brain(
        self,
        world: WorldState,
    ) -> Any:
        ...

    def bind(
        self,
        dispatch: DispatchDecision,
        *,
        world: WorldState,
        tools: ToolCatalog,
        coordination: CoordinationSpec,
    ) -> BoundDispatch:
        ...


def _explicit_rule_device(
    action: ActionIntent,
    coordination: CoordinationSpec,
) -> tuple[
    str | None,
    str | None,
]:
    """
    Resolve explicit required/preferred DeviceRule constraints.
    """

    required: list[str] = []
    preferred: list[str] = []

    for rule in coordination.device_rules:
        if not rule.matches(
            action.function_name,
            action.arguments,
        ):
            continue

        if not rule.device_id:
            continue

        if rule.mode == "required":
            required.append(
                rule.device_id
            )
        else:
            preferred.append(
                rule.device_id
            )

    required_unique = sorted(
        set(
            required
        )
    )

    if len(
        required_unique
    ) > 1:
        raise DeviceBindingError(
            "conflicting required devices: "
            f"{required_unique}"
        )

    return (
        (
            required_unique[0]
            if required_unique
            else None
        ),
        (
            preferred[0]
            if preferred
            else None
        ),
    )


def infer_required_device(
    action: ActionIntent,
    tool: ToolSpec,
    world: WorldState,
    coordination: CoordinationSpec,
) -> tuple[
    str | None,
    str | None,
]:
    """
    Infer hard/preferred device binding without asking an LLM.

    Priority:
    1. explicit DeviceRule
    2. ToolSpec.fixed_device_id
    3. ownership continuity
    4. control continuity
    5. exact reachability
    """

    required, preferred = (
        _explicit_rule_device(
            action,
            coordination,
        )
    )

    if required:
        return (
            required,
            preferred,
        )

    if tool.fixed_device_id:
        return (
            tool.fixed_device_id,
            preferred,
        )

    # --------------------------------------------------------
    # Ownership continuity
    # --------------------------------------------------------

    object_id = action.arguments.get(
        "object_id"
    )

    if (
        isinstance(
            object_id,
            str,
        )
        and world.has_entity(
            object_id
        )
    ):
        held_by = world.get(
            object_id,
            "held_by",
        )

        if (
            isinstance(
                held_by,
                str,
            )
            and held_by
        ):
            return (
                held_by,
                preferred,
            )

    # --------------------------------------------------------
    # Control continuity
    # --------------------------------------------------------

    container_id = (
        action.arguments.get(
            "container_id"
        )
        or tool.control_entity_id
    )

    if (
        isinstance(
            container_id,
            str,
        )
        and container_id
    ):
        controlling_devices = [
            device_id
            for device_id
            in world.entity_ids()
            if (
                world.get(
                    device_id,
                    "controlling",
                )
                == container_id
            )
        ]

        if (
            len(
                controlling_devices
            )
            == 1
        ):
            return (
                controlling_devices[0],
                preferred,
            )

    # --------------------------------------------------------
    # Exact reachability
    # --------------------------------------------------------

    entity_id = (
        action.arguments.get(
            tool.primary_entity_arg
        )
        if tool.primary_entity_arg
        else tool.primary_entity_id
    )

    if (
        isinstance(
            entity_id,
            str,
        )
        and entity_id
    ):
        reachable_by = world.get(
            entity_id,
            "reachable_by",
        )

        if isinstance(
            reachable_by,
            list,
        ):
            reachable = [
                value
                for value
                in reachable_by
                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value
                )
            ]

            if len(
                reachable
            ) == 1:
                return (
                    reachable[0],
                    preferred,
                )

    # --------------------------------------------------------
    # Camera-side soft preference
    # --------------------------------------------------------

    if (
        preferred is None
        and isinstance(
            entity_id,
            str,
        )
        and entity_id
        and world.has_entity(
            entity_id
        )
    ):
        camera_source = world.get(
            entity_id,
            "camera_source",
        )

        if camera_source == "left":
            preferred = "left_arm"

        elif camera_source == "right":
            preferred = "right_arm"

    return (
        None,
        preferred,
    )


class StaticWorldDeviceBinder:
    """
    Binder based only on canonical device entities in WorldState.

    Suitable for:
    - offline tests
    - regression tests
    - simulation
    - runtime environments where WorldState already contains authoritative
      device compatibility/reachability information

    Production runtime uses this binder with canonical WorldState device data.
    """

    def for_brain(
        self,
        world: WorldState,
    ) -> Any:
        devices = []

        for entity_id in world.entity_ids():
            tags = world.tags(
                entity_id
            )

            if not (
                "manipulator" in tags
                or "mobile_base" in tags
            ):
                continue

            devices.append({
                "device_id":
                    entity_id,

                "type":
                    world.get(
                        entity_id,
                        "type",
                    ),

                "tags":
                    sorted(
                        tags
                    ),

                "holding":
                    world.get(
                        entity_id,
                        "holding",
                    ),

                "controlling":
                    world.get(
                        entity_id,
                        "controlling",
                    ),

                "location":
                    world.get(
                        entity_id,
                        "location",
                    ),
            })

        return {
            "devices":
                devices,
        }

    def _candidates(
        self,
        action: ActionIntent,
        *,
        tool: ToolSpec,
        world: WorldState,
        coordination: CoordinationSpec,
    ) -> tuple[
        list[str],
        str | None,
    ]:
        required, preferred = (
            infer_required_device(
                action,
                tool,
                world,
                coordination,
            )
        )

        required_tags = set(
            tool.required_device_tags
        )

        candidates: list[str] = []

        for entity_id in world.entity_ids():
            # Device type is part of the ToolSpec contract.
            if (
                tool.device_target_type
                and world.get(
                    entity_id,
                    "type",
                )
                != tool.device_target_type
            ):
                continue

            if not required_tags.issubset(
                world.tags(
                    entity_id
                )
            ):
                continue

            candidates.append(
                entity_id
            )

        if required:
            if required not in candidates:
                raise DeviceBindingError(
                    f"required device {required} "
                    f"is not compatible with "
                    f"{action.function_name}",
                    feedback={
                        "rejection_type":
                            "REQUIRED_DEVICE_INCOMPATIBLE",

                        "attempted_action":
                            action.as_dict(),

                        "required_device_id":
                            required,
                    },
                )

            candidates = [
                required
            ]

        # ----------------------------------------------------
        # Reachability filter
        # ----------------------------------------------------

        entity_id = (
            action.arguments.get(
                tool.primary_entity_arg
            )
            if tool.primary_entity_arg
            else tool.primary_entity_id
        )

        if (
            isinstance(
                entity_id,
                str,
            )
            and entity_id
        ):
            reachable_by = world.get(
                entity_id,
                "reachable_by",
            )

            if (
                isinstance(
                    reachable_by,
                    list,
                )
                and reachable_by
            ):
                allowed = {
                    value
                    for value
                    in reachable_by
                    if isinstance(
                        value,
                        str,
                    )
                }

                candidates = [
                    device_id
                    for device_id
                    in candidates
                    if device_id
                    in allowed
                ]

        candidates = sorted(
            set(
                candidates
            )
        )

        # ----------------------------------------------------
        # Current precondition feasibility
        # ----------------------------------------------------

        feasible: list[str] = []

        infeasible_reasons: dict[
            str,
            str,
        ] = {}

        infeasible_feedback: dict[
            str,
            dict[str, Any],
        ] = {}

        for device_id in candidates:
            probe = BoundAction(
                action=action,
                device_id=device_id,
            )

            try:
                tool.validate(
                    probe,
                    world,
                )

                feasible.append(
                    device_id
                )

            except Exception as exc:
                infeasible_reasons[
                    device_id
                ] = str(
                    exc
                )

                feedback = getattr(
                    exc,
                    "feedback",
                    None,
                )

                if (
                    isinstance(
                        feedback,
                        dict,
                    )
                    and feedback
                ):
                    infeasible_feedback[
                        device_id
                    ] = feedback

        candidates = feasible

        if (
            preferred
            and preferred in candidates
        ):
            candidates.remove(
                preferred
            )

            candidates.insert(
                0,
                preferred,
            )

        if not candidates:
            state_differences: list[
                dict[str, Any]
            ] = []

            seen: set[
                tuple[Any, ...]
            ] = set()

            for (
                device_id,
                feedback,
            ) in infeasible_feedback.items():

                differences = feedback.get(
                    "state_differences",
                    [],
                )

                if not isinstance(
                    differences,
                    list,
                ):
                    continue

                for difference in differences:
                    if not isinstance(
                        difference,
                        dict,
                    ):
                        continue

                    enriched = dict(
                        difference
                    )

                    enriched.setdefault(
                        "candidate_device_id",
                        device_id,
                    )

                    key = (
                        enriched.get(
                            "subject"
                        ),
                        enriched.get(
                            "field"
                        ),
                        repr(
                            enriched.get(
                                "current"
                            )
                        ),
                        repr(
                            enriched.get(
                                "required"
                            )
                        ),
                        enriched.get(
                            "operator"
                        ),
                        enriched.get(
                            "candidate_device_id"
                        ),
                    )

                    if key in seen:
                        continue

                    seen.add(
                        key
                    )

                    state_differences.append(
                        enriched
                    )

            rejection_type = (
                "PHYSICAL_PRECONDITION_FAILED"
                if state_differences
                else "NO_FEASIBLE_DEVICE"
            )

            detail = (
                "; per-device precondition "
                f"failures={infeasible_reasons}"
                if infeasible_reasons
                else ""
            )

            raise DeviceBindingError(
                "no currently feasible device "
                f"for {action}{detail}",
                feedback={
                    "rejection_type":
                        rejection_type,

                    "attempted_action":
                        action.as_dict(),

                    "state_differences":
                        state_differences,

                    "per_device_failures": {
                        device_id: {
                            "reason":
                                infeasible_reasons.get(
                                    device_id,
                                    "",
                                ),

                            "details":
                                infeasible_feedback.get(
                                    device_id,
                                    {},
                                ),
                        }
                        for device_id
                        in sorted(
                            set(
                                infeasible_reasons
                            )
                            | set(
                                infeasible_feedback
                            )
                        )
                    },
                },
            )

        return (
            candidates,
            required,
        )

    def bind(
        self,
        dispatch: DispatchDecision,
        *,
        world: WorldState,
        tools: ToolCatalog,
        coordination: CoordinationSpec,
    ) -> BoundDispatch:
        candidate_rows: list[
            tuple[
                ActionIntent,
                list[str],
            ]
        ] = []

        for action in dispatch.actions:
            tool = tools.get(
                action.function_name
            )

            candidates, _ = (
                self._candidates(
                    action,
                    tool=tool,
                    world=world,
                    coordination=
                        coordination,
                )
            )

            candidate_rows.append(
                (
                    action,
                    candidates,
                )
            )

        assignment: list[
            tuple[
                ActionIntent,
                str,
            ]
        ] = []

        def search(
            index: int,
            used: set[str],
        ) -> bool:
            if (
                index
                >= len(
                    candidate_rows
                )
            ):
                return True

            (
                action,
                candidates,
            ) = candidate_rows[
                index
            ]

            for device_id in candidates:
                # Actions in one dispatch start concurrently.
                # One physical device can execute at most one
                # atomic action in that dispatch.
                if device_id in used:
                    continue

                assignment.append(
                    (
                        action,
                        device_id,
                    )
                )

                used.add(
                    device_id
                )

                if search(
                    index + 1,
                    used,
                ):
                    return True

                used.remove(
                    device_id
                )

                assignment.pop()

            return False

        if not search(
            0,
            set(),
        ):
            raise DeviceBindingError(
                "cannot assign distinct devices "
                "to this concurrent dispatch",
                feedback={
                    "rejection_type":
                        "DEVICE_ASSIGNMENT_CONFLICT",
                },
            )

        return BoundDispatch(
            tuple(
                BoundAction(
                    action=action,
                    device_id=device_id,
                    role_id=(
                        f"runtime_role_"
                        f"{index + 1}"
                    ),
                )
                for (
                    index,
                    (
                        action,
                        device_id,
                    ),
                )
                in enumerate(
                    assignment
                )
            )
        )



# Compatibility name for old runtime imports.


__all__ = [
    "DeviceBindingError",
    "DeviceBinder",
    "infer_required_device",
    "StaticWorldDeviceBinder",
]


# ============================================================
# Explicit Runtime device constraint
# ============================================================

def apply_force_device_constraint(
    task_spec: TaskSpec,
    *,
    world: WorldState,
    tools,
    force_device_id: str | None,
) -> tuple[TaskSpec, list[str]]:
    """
    Apply an explicit hard device constraint to a Runtime TaskSpec.

    Free-text instructions are not treated as a hardware guarantee.  This
    helper converts the UI/API force_device_id flag into required DeviceRules
    for every runtime tool whose target type is robot_arm.

    Existing LLM-produced device rules for those robot-arm functions are
    replaced so the explicit flag cannot conflict with an inferred rule.
    """

    if force_device_id is None:
        return task_spec, []

    force_device_id = str(
        force_device_id
    ).strip()

    if not force_device_id:
        return task_spec, []

    if not world.has_entity(
        force_device_id
    ):
        raise ValueError(
            "force_device_id references unknown WorldState entity: "
            f"{force_device_id!r}"
        )

    device_type = world.get(
        force_device_id,
        "type",
    )

    if device_type != "robot_arm":
        raise ValueError(
            "force_device_id 必須指向 robot_arm entity；"
            f"{force_device_id!r} type={device_type!r}"
        )

    forced_functions = sorted({
        tool.name
        for tool
        in tools.values()
        if getattr(
            tool,
            "device_target_type",
            None,
        ) == "robot_arm"
    })

    if not forced_functions:
        raise RuntimeError(
            "runtime tool catalog 沒有 robot_arm tools，"
            "無法套用 force_device_id"
        )

    forced_set = set(
        forced_functions
    )

    preserved_rules = [
        rule
        for rule
        in task_spec.coordination.device_rules
        if rule.function_name
        not in forced_set
    ]

    forced_rules = [
        DeviceRule(
            function_name=
                function_name,
            subject_arg=
                None,
            subject_id=
                None,
            device_id=
                force_device_id,
            mode=
                "required",
        )
        for function_name
        in forced_functions
    ]

    coordination = CoordinationSpec(
        maintain_until=
            list(
                task_spec.coordination.maintain_until
            ),
        device_rules=
            preserved_rules
            + forced_rules,
    )

    raw = deepcopy(
        task_spec.raw
    )

    if not isinstance(
        raw,
        dict,
    ):
        raw = {}

    raw[
        "manual_runtime_force_device"
    ] = {
        "device_id":
            force_device_id,
        "mode":
            "required",
        "functions":
            forced_functions,
    }

    return (
        TaskSpec(
            goals=
                task_spec.goals,
            coordination=
                coordination,
            motion_request=
                deepcopy(
                    task_spec.motion_request
                ),
            summary=
                task_spec.summary,
            raw=
                raw,
        ),
        forced_functions,
    )
