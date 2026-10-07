"""
Runtime coordination constraints.

This module defines constraints that apply across multiple runtime steps.

It does NOT:
- interpret natural language
- choose the next action
- choose a robot arm by itself
- execute robot actions
- mutate WorldState

Main contracts:
- MaintainFactUntilGoal
- ActionGate
- DeviceRule
- CoordinationSpec
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# dispatch.py has not been migrated yet.
# Keep this compatibility import until dispatch_schema.py is renamed.
from .dispatch import ActionIntent

from .goals import GoalSet
from .world_state import WorldState


@dataclass(frozen=True)
class MaintainFactUntilGoal:
    """
    Preserve one semantic fact until a specified goal is completed.

    Example:

        drawer_1.open_state == "open"
        until_goal = "G003"

    Important:
    This does NOT require the fact to be true initially.

    The constraint becomes active only after:
        world[subject][field] == value

    and remains active until:
        until_goal is physically satisfied.
    """

    subject: str
    field: str
    value: Any
    until_goal: str

    def is_active(
        self,
        world: WorldState,
        goals: GoalSet,
    ) -> bool:
        """
        Return True when this maintain constraint is currently active.
        """

        if goals.by_id(
            self.until_goal
        ).is_satisfied(
            world
        ):
            return False

        return (
            world.get(
                self.subject,
                self.field,
            )
            == self.value
        )


@dataclass(frozen=True)
class ActionGate:
    """
    Gate one action behind goal-state conditions.

    Example:

        close_drawer(drawer_1)

    may require:
        G001 and G002 satisfied

    before the action is allowed.

    `requires_unsatisfied_goals` can be used when an action is allowed only
    while specific goals are still incomplete.
    """

    function_name: str
    argument_name: str | None = None
    argument_value: str | None = None

    requires_goals: tuple[str, ...] = field(
        default_factory=tuple
    )

    requires_unsatisfied_goals: tuple[str, ...] = field(
        default_factory=tuple
    )

    def matches(
        self,
        action: ActionIntent,
    ) -> bool:
        """
        Return True if this gate applies to the given action.
        """

        if (
            action.function_name
            != self.function_name
        ):
            return False

        if self.argument_name is None:
            return True

        if (
            self.argument_name
            not in action.arguments
        ):
            return False

        if self.argument_value is None:
            return True

        return (
            action.arguments[
                self.argument_name
            ]
            == self.argument_value
        )


@dataclass(frozen=True)
class DeviceRule:
    """
    Explicit device restriction from the user/task or deterministic geometry.

    Normal arm selection should still prefer current WorldState facts such as:

        reachable_by
        holding
        controlling

    DeviceRule is mainly for explicit restrictions such as:

        "用左手拿 towel_1"

    mode:
        required
        preferred
    """

    function_name: str
    subject_arg: str | None = None
    subject_id: str | None = None
    device_id: str | None = None
    mode: str = "required"

    def matches(
        self,
        function_name: str,
        arguments: dict[str, str],
    ) -> bool:
        """
        Return True if this device rule applies to the action request.
        """

        if (
            self.function_name
            != function_name
        ):
            return False

        if self.subject_arg is None:
            return True

        if (
            self.subject_arg
            not in arguments
        ):
            return False

        if self.subject_id is None:
            return True

        return (
            arguments[
                self.subject_arg
            ]
            == self.subject_id
        )


@dataclass
class CoordinationSpec:
    """
    Coordination constraints attached to one interpreted task.

    maintain_until:
        Facts that must continue to hold once activated.

    device_rules:
        Explicit device restrictions / preferences.

    action_gates:
        Goal-state requirements that gate specific actions.
    """

    maintain_until: list[
        MaintainFactUntilGoal
    ] = field(
        default_factory=list
    )

    device_rules: list[
        DeviceRule
    ] = field(
        default_factory=list
    )

    action_gates: list[
        ActionGate
    ] = field(
        default_factory=list
    )

    def brain_view(
        self,
        world: WorldState,
        goals: GoalSet,
    ) -> dict[str, Any]:
        """
        Return a JSON-friendly view for Runtime Brain / debugging.
        """

        satisfied_ids = (
            goals.satisfied_ids(
                world
            )
        )

        return {
            "maintain_until": [
                {
                    "subject":
                        constraint.subject,

                    "field":
                        constraint.field,

                    "value":
                        constraint.value,

                    "until_goal":
                        constraint.until_goal,

                    "active":
                        constraint.is_active(
                            world,
                            goals,
                        ),
                }
                for constraint
                in self.maintain_until
            ],

            "device_rules": [
                {
                    "function_name":
                        rule.function_name,

                    "subject_arg":
                        rule.subject_arg,

                    "subject_id":
                        rule.subject_id,

                    "device_id":
                        rule.device_id,

                    "mode":
                        rule.mode,
                }
                for rule
                in self.device_rules
            ],

            "action_gates": [
                {
                    "function_name":
                        gate.function_name,

                    "argument_name":
                        gate.argument_name,

                    "argument_value":
                        gate.argument_value,

                    "requires_goals":
                        list(
                            gate.requires_goals
                        ),

                    "requires_unsatisfied_goals":
                        list(
                            gate.requires_unsatisfied_goals
                        ),

                    "allowed_now": (
                        set(
                            gate.requires_goals
                        ).issubset(
                            satisfied_ids
                        )
                        and not (
                            set(
                                gate.requires_unsatisfied_goals
                            )
                            & satisfied_ids
                        )
                    ),
                }
                for gate
                in self.action_gates
            ],
        }


__all__ = [
    "MaintainFactUntilGoal",
    "ActionGate",
    "DeviceRule",
    "CoordinationSpec",
]
