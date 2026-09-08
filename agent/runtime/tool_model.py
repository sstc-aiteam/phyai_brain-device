from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .dispatch_schema import ActionIntent, BoundAction
from .world_state import WorldState


@dataclass(frozen=True)
class ArgRef:
    name: str


@dataclass(frozen=True)
class DeviceRef:
    pass


DEVICE = DeviceRef()


@dataclass(frozen=True)
class FactValueRef:
    subject: str | ArgRef | DeviceRef
    field: str


@dataclass(frozen=True)
class InteractionLocationRef:
    """Resolve a known physical interaction location.

    If entity.location is known, use it. For destination-like entities,
    fallback_to_entity_id=True lets the entity ID itself act as the location.
    Otherwise unknown location stays unknown and must not create a false hard
    rejection.
    """
    subject: str | ArgRef | DeviceRef
    fallback_to_entity_id: bool = False


@dataclass(frozen=True)
class FactRef:
    subject: str | ArgRef | DeviceRef
    field: str


@dataclass(frozen=True)
class ParameterSpec:
    description: str
    required_tags: tuple[str, ...] = field(default_factory=tuple)
    must_exist: bool = True
    allowed_values: tuple[Any, ...] = field(default_factory=tuple)
    required: bool = True
    schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Condition:
    fact: FactRef
    operator: str
    value: Any


@dataclass(frozen=True)
class Effect:
    fact: FactRef
    value: Any


class ToolValidationError(ValueError):
    """Hard tool validation error with optional machine-readable feedback."""

    def __init__(
        self,
        message: str,
        *,
        feedback: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.feedback = feedback or {}


def _resolve_subject(
    ref: str | ArgRef | DeviceRef,
    action: ActionIntent,
    device_id: str,
) -> str:
    if isinstance(ref, ArgRef):
        try:
            return action.arguments[ref.name]
        except KeyError as exc:
            raise ToolValidationError(
                f"missing argument {ref.name}"
            ) from exc
    if isinstance(ref, DeviceRef):
        return device_id
    return ref


def _resolve_value(
    value: Any,
    action: ActionIntent,
    device_id: str,
    world: WorldState,
) -> Any:
    if isinstance(value, ArgRef):
        return action.arguments[value.name]
    if isinstance(value, DeviceRef):
        return device_id
    if isinstance(value, FactValueRef):
        subject = _resolve_subject(value.subject, action, device_id)
        return world.get(subject, value.field)
    if isinstance(value, InteractionLocationRef):
        subject = _resolve_subject(value.subject, action, device_id)
        location = world.get(subject, "location")
        if isinstance(location, str) and location:
            return location
        if value.fallback_to_entity_id:
            return subject
        return None
    return value



def _condition_difference(
    condition: Condition,
    *,
    action: ActionIntent,
    device_id: str,
    world: WorldState,
) -> dict[str, Any] | None:
    """Return a machine-readable state difference when a condition is false.

    This is evaluator/runtime information only.  It does NOT identify which
    tool should be chosen to repair the state.
    """
    subject = _resolve_subject(
        condition.fact.subject,
        action,
        device_id,
    )
    expected_value = _resolve_value(
        condition.value,
        action,
        device_id,
        world,
    )
    exists = world.has_field(
        subject,
        condition.fact.field,
    )
    current = world.get(
        subject,
        condition.fact.field,
    )

    if condition.operator == "eq":
        ok = current == expected_value
    elif condition.operator == "ne":
        ok = current != expected_value
    elif condition.operator == "if_present_eq":
        ok = (not exists) or current == expected_value
    elif condition.operator == "if_known_eq":
        ok = (
            (not exists)
            or expected_value is None
            or current == expected_value
        )
    else:
        raise ToolValidationError(
            f"unsupported condition operator "
            f"{condition.operator}"
        )

    if ok:
        return None

    return {
        "subject": subject,
        "field": condition.fact.field,
        "operator": condition.operator,
        "current": current,
        "required": expected_value,
        "current_known": bool(exists),
    }


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    preconditions: tuple[Condition, ...]
    effects: tuple[Effect, ...]
    device_target_type: str
    required_device_tags: tuple[str, ...]
    required_capabilities: tuple[str, ...] = field(default_factory=tuple)
    required_zones: tuple[str, ...] = field(default_factory=tuple)
    primary_entity_arg: str | None = None
    primary_entity_id: str | None = None
    control_entity_id: str | None = None
    fixed_device_id: str | None = None
    exclusive_entity_args: tuple[str, ...] = field(default_factory=tuple)
    capacity_limited_entity_args: dict[str, str] = field(
        default_factory=dict
    )

    @property
    def required_arguments(self) -> tuple[str, ...]:
        return tuple(
            name for name, spec in self.parameters.items()
            if spec.required
        )

    def brain_view(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "arguments": {
                name: {
                    "description": spec.description,
                    "required_tags": list(spec.required_tags),
                    "entity_reference": spec.must_exist,
                    "allowed_values": list(spec.allowed_values),
                    "required": spec.required,
                    "schema": dict(spec.schema),
                }
                for name, spec in self.parameters.items()
            },
            "device_requirements": {
                "target_type": self.device_target_type,
                "required_tags": list(self.required_device_tags),
                "required_capabilities": list(self.required_capabilities),
            },
            "preconditions": [
                _condition_text(c) for c in self.preconditions
            ],
            "effects": [
                _effect_text(e) for e in self.effects
            ],
        }

    def legacy_view(self) -> dict[str, Any]:
        """Convert to the existing agent.planning tool-catalog shape."""
        return {
            "api_function": self.name,
            "module": "agent.runtime",
            "description": self.description,
            "parameters": {
                name: {
                    **(dict(spec.schema) if spec.schema else {"type": "string"}),
                    "description": spec.description,
                }
                for name, spec in self.parameters.items()
            },
            "required": list(self.required_arguments),
            "target_types": [self.device_target_type],
            "required_capabilities": list(self.required_capabilities),
            "required_zones": list(self.required_zones),
        }

    def precondition_differences(
        self,
        bound: BoundAction,
        world: WorldState,
    ) -> list[dict[str, Any]]:
        """Return all false preconditions as CURRENT-vs-REQUIRED facts."""
        result: list[dict[str, Any]] = []
        for condition in self.preconditions:
            difference = _condition_difference(
                condition,
                action=bound.action,
                device_id=bound.device_id,
                world=world,
            )
            if difference is not None:
                result.append(difference)
        return result

    def validate(
        self,
        bound: BoundAction,
        world: WorldState,
    ) -> None:
        action = bound.action
        if action.function_name != self.name:
            raise ToolValidationError(
                f"tool mismatch: {action.function_name} != {self.name}"
            )

        expected = set(self.parameters)
        actual = set(action.arguments)
        missing = set(self.required_arguments) - actual
        extra = actual - expected
        if missing:
            raise ToolValidationError(
                f"{self.name} missing arguments: {sorted(missing)}"
            )
        if extra:
            raise ToolValidationError(
                f"{self.name} unsupported arguments: {sorted(extra)}"
            )

        if not world.has_entity(bound.device_id):
            raise ToolValidationError(
                f"unknown device: {bound.device_id}"
            )

        device_tags = world.tags(bound.device_id)
        missing_device_tags = (
            set(self.required_device_tags) - device_tags
        )
        if missing_device_tags:
            raise ToolValidationError(
                f"device {bound.device_id} missing tags "
                f"{sorted(missing_device_tags)}"
            )

        for arg_name, spec in self.parameters.items():
            value = action.arguments[arg_name]

            if spec.allowed_values and value not in spec.allowed_values:
                raise ToolValidationError(
                    f"{self.name}.{arg_name} must be one of "
                    f"{list(spec.allowed_values)!r}; current={value!r}"
                )

            if spec.must_exist:
                entity_id = value
                if not world.has_entity(entity_id):
                    raise ToolValidationError(
                        f"{arg_name} references unknown entity {entity_id}"
                    )
                if spec.required_tags:
                    missing_tags = (
                        set(spec.required_tags) - world.tags(entity_id)
                    )
                    if missing_tags:
                        raise ToolValidationError(
                            f"{entity_id} missing tags "
                            f"{sorted(missing_tags)}"
                        )

        differences = self.precondition_differences(
            bound,
            world,
        )
        if differences:
            first = differences[0]
            raise ToolValidationError(
                f"{self.name} precondition failed: "
                f"{first['subject']}.{first['field']} "
                f"{first['operator']} {first['required']!r}; "
                f"current={first['current']!r}",
                feedback={
                    "rejection_type": (
                        "PHYSICAL_PRECONDITION_FAILED"
                    ),
                    "attempted_action": action.as_dict(),
                    "device_id": bound.device_id,
                    "state_differences": differences,
                },
            )

    def read_facts(
        self,
        bound: BoundAction,
    ) -> set[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for condition in self.preconditions:
            result.add((
                _resolve_subject(
                    condition.fact.subject,
                    bound.action,
                    bound.device_id,
                ),
                condition.fact.field,
            ))
        return result

    def write_facts(
        self,
        bound: BoundAction,
    ) -> set[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for effect in self.effects:
            result.add((
                _resolve_subject(
                    effect.fact.subject,
                    bound.action,
                    bound.device_id,
                ),
                effect.fact.field,
            ))
        return result

    def exclusive_entities(
        self,
        action: ActionIntent,
    ) -> set[str]:
        return {
            action.arguments[arg]
            for arg in self.exclusive_entity_args
            if arg in action.arguments
        }

    def capacity_limited_entities(
        self,
        action: ActionIntent,
    ) -> list[tuple[str, str]]:
        """Return (entity_id, capacity_field) claims for this action.

        Example:
          place_object(destination_id=container_02)
          -> ("container_02", "parallel_place_capacity")

        A missing/invalid capacity field is treated conservatively as 1 by
        ParallelValidator.
        """
        result: list[tuple[str, str]] = []
        for arg_name, capacity_field in (
            self.capacity_limited_entity_args.items()
        ):
            if arg_name in action.arguments:
                result.append((
                    action.arguments[arg_name],
                    capacity_field,
                ))
        return result

    def apply_effects(
        self,
        world: WorldState,
        bound: BoundAction,
    ) -> WorldState:
        updated = world.clone()
        for effect in self.effects:
            subject = _resolve_subject(
                effect.fact.subject,
                bound.action,
                bound.device_id,
            )
            value = _resolve_value(
                effect.value,
                bound.action,
                bound.device_id,
                updated,
            )
            updated.set(subject, effect.fact.field, value)
        return updated


def _subject_text(value: str | ArgRef | DeviceRef) -> str:
    if isinstance(value, ArgRef):
        return f"arg:{value.name}"
    if isinstance(value, DeviceRef):
        return "device"
    return value


def _value_text(value: Any) -> str:
    if isinstance(value, ArgRef):
        return f"arg:{value.name}"
    if isinstance(value, DeviceRef):
        return "device"
    if isinstance(value, FactValueRef):
        return (
            f"{_subject_text(value.subject)}.{value.field}"
        )
    if isinstance(value, InteractionLocationRef):
        suffix = ", fallback=entity_id" if value.fallback_to_entity_id else ""
        return (
            f"interaction_location({_subject_text(value.subject)}{suffix})"
        )
    return repr(value)


def _condition_text(condition: Condition) -> str:
    return (
        f"{_subject_text(condition.fact.subject)}."
        f"{condition.fact.field} {condition.operator} "
        f"{_value_text(condition.value)}"
    )


def _effect_text(effect: Effect) -> str:
    return (
        f"{_subject_text(effect.fact.subject)}."
        f"{effect.fact.field} := {_value_text(effect.value)}"
    )


class ToolCatalog:
    def __init__(self, tools: Iterable[ToolSpec]):
        tool_list = list(tools)
        self._tools = {tool.name: tool for tool in tool_list}
        if len(self._tools) != len(tool_list):
            raise ValueError("Duplicate runtime tool name")

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolValidationError(
                f"Unknown runtime tool: {name}"
            ) from exc

    def values(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def brain_view(self) -> dict[str, Any]:
        return {
            tool.name: tool.brain_view()
            for tool in self.values()
        }

    def legacy_view(self) -> dict[str, dict[str, Any]]:
        return {
            tool.name: tool.legacy_view()
            for tool in self.values()
        }
