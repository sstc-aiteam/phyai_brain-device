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

from typing import Any, Protocol

from .coordination import CoordinationSpec
from .dispatch import (
    ActionIntent,
    BoundAction,
    BoundDispatch,
    DispatchDecision,
)
from .tool_model import (
    ToolCatalog,
    ToolSpec,
)
from .world_state import WorldState


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

    Production may still use PlanningDeviceBinder while the existing
    DeviceRegistry allocator remains the source of hardware allocation truth.
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


class PlanningDeviceBinder:
    """
    Adapter to the existing agent.planning DeviceRegistry allocator.

    This preserves the current production hardware-allocation path while the
    closed-loop runtime is being refactored.

    Imports are lazy so offline runtime tests do not require hardware config.
    """

    def __init__(
        self,
        device_registry=None,
        *,
        probe_hardware: bool = True,
        device_statuses: dict[
            str,
            dict[str, Any],
        ] | None = None,
    ):
        self.device_registry = (
            device_registry
        )

        self.probe_hardware = (
            probe_hardware
        )

        self.device_statuses = (
            device_statuses
        )

    def _registry(
        self,
    ):
        if (
            self.device_registry
            is not None
        ):
            return (
                self.device_registry
            )

        from agent.planning.device_registry import (
            DeviceRegistry,
        )

        self.device_registry = (
            DeviceRegistry.from_arm_config()
        )

        return (
            self.device_registry
        )

    def for_brain(
        self,
        world: WorldState,
    ) -> Any:
        _ = world

        registry = (
            self._registry()
        )

        try:
            return (
                registry.for_prompt()
            )

        except Exception:
            return {
                "device_registry":
                    "available",
            }

    def bind(
        self,
        dispatch: DispatchDecision,
        *,
        world: WorldState,
        tools: ToolCatalog,
        coordination: CoordinationSpec,
    ) -> BoundDispatch:
        _ = world

        from agent.planning.plan_pipeline import (
            prepare_plan,
        )

        roles = []
        steps = []
        required_roles = []

        for (
            index,
            action,
        ) in enumerate(
            dispatch.actions,
            start=1,
        ):
            tool = tools.get(
                action.function_name
            )

            role_id = (
                f"runtime_role_{index}"
            )

            step_id = (
                f"runtime_step_{index}"
            )

            (
                required_device,
                preferred_device,
            ) = infer_required_device(
                action,
                tool,
                world,
                coordination,
            )

            if required_device:
                binding = {
                    "mode":
                        "required",
                    "target_id":
                        required_device,
                }

            elif preferred_device:
                binding = {
                    "mode":
                        "preferred",
                    "target_id":
                        preferred_device,
                }

            else:
                binding = {
                    "mode":
                        "automatic",
                }

            roles.append({
                "id":
                    role_id,

                "target_type":
                    tool.device_target_type,

                "required_capabilities":
                    list(
                        tool.required_capabilities
                    ),

                "required_zones":
                    list(
                        tool.required_zones
                    ),

                "binding":
                    binding,
            })

            steps.append({
                "id":
                    step_id,

                "role":
                    role_id,

                "action":
                    action.as_dict(),

                "depends_on":
                    [],

                "satisfies":
                    [],
            })

            required_roles.append(
                role_id
            )

        constraints = []

        if len(
            required_roles
        ) > 1:
            constraints.append({
                "type":
                    "distinct_assignment",

                "roles":
                    required_roles,

                "hard":
                    True,
            })

        micro_plan = {
            "schema_version":
                "1.0",

            "roles":
                roles,

            "steps":
                steps,

            "requirements":
                [],

            "constraints":
                constraints,

            "assignment_preferences":
                [],
        }

        prepared = prepare_plan(
            micro_plan,
            tools.legacy_view(),
            self._registry(),
            probe_hardware=
                self.probe_hardware,
            device_statuses=
                self.device_statuses,
        )

        resolved = prepared[
            "resolved_plan"
        ]

        resolved_steps = (
            resolved.get(
                "resolved_steps"
            )
            or []
        )

        by_step = {
            str(
                step["id"]
            ):
                step
            for step
            in resolved_steps
        }

        bound: list[
            BoundAction
        ] = []

        for (
            index,
            action,
        ) in enumerate(
            dispatch.actions,
            start=1,
        ):
            step_id = (
                f"runtime_step_{index}"
            )

            row = by_step.get(
                step_id
            )

            if not row:
                raise DeviceBindingError(
                    "resolved plan missing "
                    f"{step_id}"
                )

            target = (
                row.get(
                    "target"
                )
                or {}
            )

            device_id = target.get(
                "id"
            )

            if (
                not isinstance(
                    device_id,
                    str,
                )
                or not device_id
            ):
                raise DeviceBindingError(
                    f"{step_id} missing "
                    "resolved target id"
                )

            bound.append(
                BoundAction(
                    action=action,
                    device_id=device_id,
                    role_id=(
                        f"runtime_role_"
                        f"{index}"
                    ),
                )
            )

        return BoundDispatch(
            tuple(
                bound
            )
        )


# Compatibility name for old runtime imports.
LegacyPlanningDeviceBinder = (
    PlanningDeviceBinder
)


__all__ = [
    "DeviceBindingError",
    "DeviceBinder",
    "infer_required_device",
    "StaticWorldDeviceBinder",
    "PlanningDeviceBinder",
    "LegacyPlanningDeviceBinder",
]
