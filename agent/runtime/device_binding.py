from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .coordination_schema import CoordinationSpec
from .dispatch_schema import (
    ActionIntent,
    BoundAction,
    BoundDispatch,
    DispatchDecision,
)
from .tool_model import ToolCatalog, ToolSpec
from .world_state import WorldState


class DeviceBindingError(RuntimeError):
    """Hard binding/feasibility error with optional structured feedback."""

    def __init__(
        self,
        message: str,
        *,
        feedback: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.feedback = feedback or {}


class DeviceBinder(Protocol):
    def for_brain(self, world: WorldState) -> Any:
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
) -> tuple[str | None, str | None]:
    required: list[str] = []
    preferred: list[str] = []

    for rule in coordination.device_rules:
        if not rule.matches(action.function_name, action.arguments):
            continue
        if not rule.device_id:
            continue
        if rule.mode == "required":
            required.append(rule.device_id)
        else:
            preferred.append(rule.device_id)

    if len(set(required)) > 1:
        raise DeviceBindingError(
            f"conflicting required devices: {sorted(set(required))}"
        )
    return (
        required[0] if required else None,
        preferred[0] if preferred else None,
    )


def infer_required_device(
    action: ActionIntent,
    tool: ToolSpec,
    world: WorldState,
    coordination: CoordinationSpec,
) -> tuple[str | None, str | None]:
    """Infer hard/preferred device binding without asking an LLM."""

    required, preferred = _explicit_rule_device(
        action,
        coordination,
    )
    if required:
        return required, preferred
    if tool.fixed_device_id:
        return tool.fixed_device_id, preferred

    # Ownership continuity: place must use the arm currently holding the object.
    object_id = action.arguments.get("object_id")
    if object_id and world.has_entity(object_id):
        held_by = world.get(object_id, "held_by")
        if isinstance(held_by, str) and held_by:
            return held_by, preferred

    # Container close must use the device currently controlling that resource.
    container_id = action.arguments.get("container_id") or tool.control_entity_id
    if container_id:
        controlling_devices = [
            device_id
            for device_id in world.entity_ids()
            if world.get(device_id, "controlling") == container_id
        ]
        if len(controlling_devices) == 1:
            return controlling_devices[0], preferred

    # Geometry / adapter can expose exact reachability in canonical WorldState.
    entity_id = (
        action.arguments.get(tool.primary_entity_arg)
        if tool.primary_entity_arg else tool.primary_entity_id
    )
    if entity_id:
        reachable_by = (
            world.get(entity_id, "reachable_by")
            if entity_id
            else None
        )
        if isinstance(reachable_by, list):
            reachable = [
                str(x) for x in reachable_by
                if isinstance(x, str) and x
            ]
            if len(reachable) == 1:
                return reachable[0], preferred

    return None, preferred


class StaticWorldDeviceBinder:
    """Offline/test binder using canonical device entities in WorldState.

    This is NOT intended to replace your production DeviceRegistry allocator.
    It exists so runtime core and regression scenarios can run independently.
    """

    def for_brain(self, world: WorldState) -> Any:
        devices = []
        for entity_id in world.entity_ids():
            tags = world.tags(entity_id)
            if "manipulator" in tags or "mobile_base" in tags:
                devices.append({
                    "device_id": entity_id,
                    "type": world.get(entity_id, "type"),
                    "tags": sorted(tags),
                    "holding": world.get(entity_id, "holding"),
                    "controlling": world.get(entity_id, "controlling"),
                    "location": world.get(entity_id, "location"),
                })
        return {"devices": devices}

    def _candidates(
        self,
        action: ActionIntent,
        *,
        tool: ToolSpec,
        world: WorldState,
        coordination: CoordinationSpec,
    ) -> tuple[list[str], str | None]:
        required, preferred = infer_required_device(
            action,
            tool,
            world,
            coordination,
        )

        candidates = []
        required_tags = set(tool.required_device_tags)
        for entity_id in world.entity_ids():
            if not required_tags.issubset(world.tags(entity_id)):
                continue
            candidates.append(entity_id)

        if required:
            if required not in candidates:
                raise DeviceBindingError(
                    f"required device {required} is not compatible "
                    f"with {action.function_name}"
                )
            candidates = [required]

        # Multi-candidate reachability filtering.
        entity_id = (
            action.arguments.get(tool.primary_entity_arg)
            if tool.primary_entity_arg else tool.primary_entity_id
        )
        if entity_id:
            reachable_by = (
                world.get(entity_id, "reachable_by")
                if entity_id
                else None
            )
            if isinstance(reachable_by, list) and reachable_by:
                allowed = {str(x) for x in reachable_by}
                candidates = [
                    d for d in candidates if d in allowed
                ]

        candidates = sorted(set(candidates))

        # Runtime precondition-aware device filtering.
        #
        # Device allocation must not only ask "which device has the right
        # capability/reachability?"  It must also ask "which device can execute
        # THIS action in the CURRENT world state?"
        #
        # Example:
        #   left_arm.holding = towel_1
        #   right_arm.holding = None
        #   action = open_container(cabinet_1)
        #
        # Both arms are geometrically compatible, but only right_arm satisfies
        # open_container's current device preconditions.  Without this filter,
        # deterministic sorting could repeatedly choose left_arm and the brain
        # would be blamed for an allocator mistake it cannot fix (the brain does
        # not choose device IDs by design).
        feasible: list[str] = []
        infeasible_reasons: dict[str, str] = {}
        infeasible_feedback: dict[str, dict[str, Any]] = {}

        for device_id in candidates:
            probe = BoundAction(
                action=action,
                device_id=device_id,
            )
            try:
                tool.validate(probe, world)
                feasible.append(device_id)
            except Exception as exc:
                infeasible_reasons[device_id] = str(exc)
                feedback = getattr(exc, "feedback", None)
                if isinstance(feedback, dict) and feedback:
                    infeasible_feedback[device_id] = feedback

        candidates = feasible

        if preferred and preferred in candidates:
            candidates.remove(preferred)
            candidates.insert(0, preferred)

        if not candidates:
            detail = (
                f"; per-device precondition failures={infeasible_reasons}"
                if infeasible_reasons
                else ""
            )

            # Merge machine-readable CURRENT-vs-REQUIRED differences.
            # No repair action/tool is generated here: the LLM must still infer
            # a repairing action from available_tools and their effects.
            state_differences: list[dict[str, Any]] = []
            seen: set[tuple[Any, ...]] = set()

            for device_id, feedback in infeasible_feedback.items():
                for difference in feedback.get(
                    "state_differences",
                    [],
                ):
                    if not isinstance(difference, dict):
                        continue
                    enriched = dict(difference)
                    enriched.setdefault(
                        "candidate_device_id",
                        device_id,
                    )
                    key = (
                        enriched.get("subject"),
                        enriched.get("field"),
                        repr(enriched.get("current")),
                        repr(enriched.get("required")),
                        enriched.get("operator"),
                        enriched.get("candidate_device_id"),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    state_differences.append(enriched)

            rejection_type = (
                "PHYSICAL_PRECONDITION_FAILED"
                if state_differences
                else "NO_FEASIBLE_DEVICE"
            )

            raise DeviceBindingError(
                f"no currently feasible device for {action}{detail}",
                feedback={
                    "rejection_type": rejection_type,
                    "attempted_action": action.as_dict(),
                    "state_differences": state_differences,
                    "per_device_failures": {
                        device_id: {
                            "reason": infeasible_reasons.get(
                                device_id,
                                "",
                            ),
                            "details": infeasible_feedback.get(
                                device_id,
                                {},
                            ),
                        }
                        for device_id in sorted(
                            set(infeasible_reasons)
                            | set(infeasible_feedback)
                        )
                    },
                },
            )

        return candidates, required

    def bind(
        self,
        dispatch: DispatchDecision,
        *,
        world: WorldState,
        tools: ToolCatalog,
        coordination: CoordinationSpec,
    ) -> BoundDispatch:
        candidate_rows = []
        for action in dispatch.actions:
            tool = tools.get(action.function_name)
            candidates, _ = self._candidates(
                action,
                tool=tool,
                world=world,
                coordination=coordination,
            )
            candidate_rows.append((action, candidates))

        assignment: list[tuple[ActionIntent, str]] = []

        def search(index: int, used: set[str]) -> bool:
            if index >= len(candidate_rows):
                return True
            action, candidates = candidate_rows[index]
            for device_id in candidates:
                # One dispatch means concurrent start; one physical device may
                # execute at most one exclusive atomic action in that dispatch.
                if device_id in used:
                    continue
                assignment.append((action, device_id))
                used.add(device_id)
                if search(index + 1, used):
                    return True
                used.remove(device_id)
                assignment.pop()
            return False

        if not search(0, set()):
            raise DeviceBindingError(
                "cannot assign distinct devices to this concurrent dispatch"
            )

        return BoundDispatch(tuple(
            BoundAction(
                action=action,
                device_id=device_id,
                role_id=f"runtime_role_{index + 1}",
            )
            for index, (action, device_id)
            in enumerate(assignment)
        ))


class LegacyPlanningDeviceBinder:
    """Adapter to your EXISTING agent.planning allocator.

    It builds a one-dispatch micro Plan and calls the current
    plan_pipeline.prepare_plan(), which already performs catalog validation,
    status probing, allocation, assignment validation and resolution.

    Existing DeviceRegistry / allocator remain the production source of device
    truth.  This class intentionally uses lazy imports so offline runtime tests
    do not depend on hardware configuration.
    """

    def __init__(
        self,
        device_registry=None,
        *,
        probe_hardware: bool = True,
        device_statuses: dict[str, dict[str, Any]] | None = None,
    ):
        self.device_registry = device_registry
        self.probe_hardware = probe_hardware
        self.device_statuses = device_statuses

    def _registry(self):
        if self.device_registry is not None:
            return self.device_registry
        from agent.planning.device_registry import DeviceRegistry
        self.device_registry = DeviceRegistry.from_arm_config()
        return self.device_registry

    def for_brain(self, world: WorldState) -> Any:
        registry = self._registry()
        try:
            return registry.for_prompt()
        except Exception:
            return {"device_registry": "available"}

    def bind(
        self,
        dispatch: DispatchDecision,
        *,
        world: WorldState,
        tools: ToolCatalog,
        coordination: CoordinationSpec,
    ) -> BoundDispatch:
        from agent.planning.plan_pipeline import prepare_plan

        roles = []
        steps = []
        required_roles = []

        for index, action in enumerate(dispatch.actions, start=1):
            tool = tools.get(action.function_name)
            role_id = f"runtime_role_{index}"
            step_id = f"runtime_step_{index}"

            required_device, preferred_device = infer_required_device(
                action,
                tool,
                world,
                coordination,
            )

            if required_device:
                binding = {
                    "mode": "required",
                    "target_id": required_device,
                }
            elif preferred_device:
                binding = {
                    "mode": "preferred",
                    "target_id": preferred_device,
                }
            else:
                binding = {"mode": "automatic"}

            roles.append({
                "id": role_id,
                "target_type": tool.device_target_type,
                "required_capabilities": [],
                "required_zones": [],
                "binding": binding,
            })
            steps.append({
                "id": step_id,
                "role": role_id,
                "action": action.as_dict(),
                "depends_on": [],
                "satisfies": [],
            })
            required_roles.append(role_id)

        constraints = []
        if len(required_roles) > 1:
            constraints.append({
                "type": "distinct_assignment",
                "roles": required_roles,
                "hard": True,
            })

        micro_plan = {
            "schema_version": "1.0",
            "roles": roles,
            "steps": steps,
            "requirements": [],
            "constraints": constraints,
            "assignment_preferences": [],
        }

        prepared = prepare_plan(
            micro_plan,
            tools.legacy_view(),
            self._registry(),
            probe_hardware=self.probe_hardware,
            device_statuses=self.device_statuses,
        )

        resolved = prepared["resolved_plan"]
        resolved_steps = resolved.get("resolved_steps") or []
        by_step = {
            str(step["id"]): step
            for step in resolved_steps
        }

        bound = []
        for index, action in enumerate(dispatch.actions, start=1):
            step_id = f"runtime_step_{index}"
            row = by_step.get(step_id)
            if not row:
                raise DeviceBindingError(
                    f"resolved plan missing {step_id}"
                )
            target = row.get("target") or {}
            device_id = target.get("id")
            if not isinstance(device_id, str) or not device_id:
                raise DeviceBindingError(
                    f"{step_id} missing resolved target id"
                )
            bound.append(BoundAction(
                action=action,
                device_id=device_id,
                role_id=f"runtime_role_{index}",
            ))

        return BoundDispatch(tuple(bound))
