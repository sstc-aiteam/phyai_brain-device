from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dispatch_schema import ActionIntent

from .goal_schema import GoalSet
from .world_state import WorldState


@dataclass(frozen=True)
class MaintainFactUntilGoal:
    """Once the maintained fact becomes true, preserve it until a goal is done.

    Example:
      cabinet_1.open_state == "open"
      until_goal = "towel_inside"

    This does NOT require the cabinet to be open initially.  The constraint only
    becomes active after the fact reaches the requested value.
    """

    subject: str
    field: str
    value: Any
    until_goal: str

    def is_active(self, world: WorldState, goals: GoalSet) -> bool:
        if goals.by_id(self.until_goal).is_satisfied(world):
            return False
        return world.get(self.subject, self.field) == self.value


@dataclass(frozen=True)
class ActionGate:
    function_name: str
    argument_name: str | None = None
    argument_value: str | None = None
    requires_goals: tuple[str, ...] = field(default_factory=tuple)
    requires_unsatisfied_goals: tuple[str, ...] = field(default_factory=tuple)

    def matches(self, action: ActionIntent) -> bool:
        if action.function_name != self.function_name:
            return False
        if self.argument_name is None:
            return True
        if self.argument_name not in action.arguments:
            return False
        if self.argument_value is None:
            return True
        return action.arguments[self.argument_name] == self.argument_value


@dataclass(frozen=True)
class DeviceRule:

    """Explicit device restriction from user/task or deterministic geometry.

    Prefer geometry/reachability in WorldState for normal operation.  Use this
    when the task itself explicitly requires a named device.
    """

    function_name: str
    subject_arg: str | None = None
    subject_id: str | None = None
    device_id: str | None = None
    mode: str = "required"  # required | preferred

    def matches(self, function_name: str, arguments: dict[str, str]) -> bool:
        if self.function_name != function_name:
            return False
        if self.subject_arg is None:
            return True
        if self.subject_arg not in arguments:
            return False
        if self.subject_id is None:
            return True
        return arguments[self.subject_arg] == self.subject_id


@dataclass
class CoordinationSpec:
    maintain_until: list[MaintainFactUntilGoal] = field(default_factory=list)
    device_rules: list[DeviceRule] = field(default_factory=list)
    action_gates: list[ActionGate] = field(default_factory=list)

    def brain_view(self, world: WorldState, goals: GoalSet) -> dict[str, Any]:
        return {
            "maintain_until": [
                {
                    "subject": c.subject,
                    "field": c.field,
                    "value": c.value,
                    "until_goal": c.until_goal,
                    "active": c.is_active(world, goals),
                }
                for c in self.maintain_until
            ],
            "device_rules": [
                {
                    "function_name": r.function_name,
                    "subject_arg": r.subject_arg,
                    "subject_id": r.subject_id,
                    "device_id": r.device_id,
                    "mode": r.mode,
                }
                for r in self.device_rules
            ],
            "action_gates": [
                {
                    "function_name": g.function_name,
                    "argument_name": g.argument_name,
                    "argument_value": g.argument_value,
                    "requires_goals": list(g.requires_goals),
                    "requires_unsatisfied_goals": list(
                        g.requires_unsatisfied_goals
                    ),
                    "allowed_now": (
                        set(g.requires_goals).issubset(
                            goals.satisfied_ids(world)
                        )
                        and not (
                            set(g.requires_unsatisfied_goals)
                            & goals.satisfied_ids(world)
                        )
                    ),
                }
                for g in self.action_gates
            ],
        }
