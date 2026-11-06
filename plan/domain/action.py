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


ACTION_DECISION_PURPOSES = (
    "complete_goal",
    "satisfy_prerequisite",
    "repair_precondition",
    "remove_blocker",
    "maintain_commitment",
    "recovery",
)


@dataclass(frozen=True)
class ActionDecisionContext:
    """Compact Brain-declared purpose for one ActionIntent.

    This is executive/task context only.  It is deliberately kept outside
    ActionIntent so action identity/signatures remain stable for binding,
    repair tracking, loop detection, and validation.
    """

    purpose: str
    related_goal_ids: tuple[str, ...] = ()
    target_fact: dict[str, Any] | None = None
    reason_summary: str = ""

    def __post_init__(self) -> None:
        if self.purpose not in ACTION_DECISION_PURPOSES:
            raise ValueError(
                "unsupported action decision purpose: "
                f"{self.purpose!r}"
            )

        if self.target_fact is not None and not isinstance(
            self.target_fact,
            dict,
        ):
            raise TypeError(
                "target_fact must be object or None"
            )

        if not str(self.reason_summary).strip():
            raise ValueError(
                "reason_summary must be non-empty"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "related_goal_ids": list(
                self.related_goal_ids
            ),
            "target_fact": (
                dict(self.target_fact)
                if self.target_fact is not None
                else None
            ),
            "reason_summary": self.reason_summary,
        }


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

    # Keep the legacy pending-intent fields in their original positional
    # order for backward compatibility with external/custom callers.
    pending_intent_resolution: str | None = None
    pending_intent_abandon_reason: str | None = None

    # Optional for backward compatibility with custom/tests that construct
    # DispatchDecision directly. RuntimeBrain always supplies one context per
    # action.
    action_contexts: tuple[ActionDecisionContext, ...] = ()

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError(
                "DispatchDecision must contain at least one action"
            )

        if (
            self.action_contexts
            and len(self.action_contexts) != len(self.actions)
        ):
            raise ValueError(
                "action_contexts must align one-to-one with actions"
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

        if self.action_contexts:
            result["action_contexts"] = [
                context.as_dict()
                for context in self.action_contexts
            ]

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
    "ACTION_DECISION_PURPOSES",
    "ActionDecisionContext",
    "ActionIntent",
    "DispatchDecision",
    "BoundAction",
    "BoundDispatch",
    "ActionExecutionResult",
    "DispatchExecutionReport",
]
