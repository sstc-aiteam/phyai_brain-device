"""
Runtime dispatch contracts.

This module contains only the data exchanged between:
- runtime brain
- device binding
- validator
- executor
- progress / repair / loop monitors

It does not:
- choose an action
- choose a device
- validate physical feasibility
- execute robot commands
- mutate WorldState
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionIntent:
    """
    One atomic action selected by the runtime brain.

    `device_id` does not belong here. Device selection happens later in
    the binding stage.
    """

    function_name: str
    arguments: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "arguments": dict(self.arguments),
        }

    def __str__(self) -> str:
        args = ", ".join(
            f"{key}={value}"
            for key, value in self.arguments.items()
        )
        return f"{self.function_name}({args})"


@dataclass(frozen=True)
class DispatchDecision:
    """
    One closed-loop runtime decision.

    A dispatch may contain more than one action only when those actions may
    start concurrently in the current world state.

    The pending-intent fields belong to the repair/executive-memory protocol.
    """

    actions: tuple[ActionIntent, ...]

    pending_intent_resolution: str | None = None
    pending_intent_abandon_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError(
                "DispatchDecision must contain at least one action"
            )

        if self.pending_intent_resolution not in {
            None,
            "continue",
            "abandon",
        }:
            raise ValueError(
                "pending_intent_resolution must be "
                "None, 'continue', or 'abandon'"
            )

        if (
            self.pending_intent_resolution == "abandon"
            and not str(
                self.pending_intent_abandon_reason or ""
            ).strip()
        ):
            raise ValueError(
                "pending_intent_abandon_reason is required "
                "when pending_intent_resolution='abandon'"
            )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "actions": [
                action.as_dict()
                for action in self.actions
            ],
        }

        if self.pending_intent_resolution is not None:
            result["pending_intent_resolution"] = (
                self.pending_intent_resolution
            )

        if self.pending_intent_abandon_reason is not None:
            result["pending_intent_abandon_reason"] = (
                self.pending_intent_abandon_reason
            )

        return result


@dataclass(frozen=True)
class BoundAction:
    """
    ActionIntent after a concrete runtime device has been assigned.
    """

    action: ActionIntent
    device_id: str
    role_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.action.as_dict(),
            "device_id": self.device_id,
            "role_id": self.role_id,
        }


@dataclass(frozen=True)
class BoundDispatch:
    """
    One dispatch after all actions have concrete device bindings.
    """

    actions: tuple[BoundAction, ...]

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError(
                "BoundDispatch must contain at least one action"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "actions": [
                action.as_dict()
                for action in self.actions
            ],
        }


@dataclass(frozen=True)
class ActionExecutionResult:
    """
    Execution result for one bound action.

    command_success:
        The command/callback itself completed successfully.

    verified_success:
        Fresh observation confirmed that the intended physical result holds.

    These are intentionally separate because command success must not be
    treated as semantic/world-state success.
    """

    bound_action: BoundAction
    command_success: bool
    verified_success: bool
    error: str | None = None
    raw_result: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.bound_action.as_dict(),
            "command_success": self.command_success,
            "verified_success": self.verified_success,
            "error": self.error,
            "raw_result": self.raw_result,
        }


@dataclass
class DispatchExecutionReport:
    """
    Aggregate execution report for one dispatch.
    """

    results: list[ActionExecutionResult] = field(
        default_factory=list
    )

    @property
    def any_success(self) -> bool:
        return any(
            result.verified_success
            for result in self.results
        )

    @property
    def all_success(self) -> bool:
        return (
            bool(self.results)
            and all(
                result.verified_success
                for result in self.results
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_success": self.all_success,
            "any_success": self.any_success,
            "results": [
                result.as_dict()
                for result in self.results
            ],
        }


__all__ = [
    "ActionIntent",
    "DispatchDecision",
    "BoundAction",
    "BoundDispatch",
    "ActionExecutionResult",
    "DispatchExecutionReport",
]
