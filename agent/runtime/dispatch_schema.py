from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionIntent:
    function_name: str
    arguments: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "arguments": dict(self.arguments),
        }

    def __str__(self) -> str:
        args = ", ".join(
            f"{k}={v}" for k, v in self.arguments.items()
        )
        return f"{self.function_name}({args})"


@dataclass(frozen=True)
class DispatchDecision:
    actions: tuple[ActionIntent, ...]
    rationale_tag: str | None = None

    # Executive-memory protocol.  These fields do NOT choose an action.
    # They only let the brain explicitly resolve its OWN previously blocked
    # action intent when that intent is ready again.
    pending_intent_resolution: str | None = None
    pending_intent_abandon_reason: str | None = None

    def __post_init__(self):
        if not self.actions:
            raise ValueError("DispatchDecision must contain at least one action")
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
        result = {
            "actions": [a.as_dict() for a in self.actions],
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
    actions: tuple[BoundAction, ...]

    def __post_init__(self):
        if not self.actions:
            raise ValueError("BoundDispatch must contain at least one action")

    def as_dict(self) -> dict[str, Any]:
        return {
            "actions": [a.as_dict() for a in self.actions],
        }


@dataclass(frozen=True)
class ActionExecutionResult:
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
    results: list[ActionExecutionResult] = field(default_factory=list)

    @property
    def any_success(self) -> bool:
        return any(r.verified_success for r in self.results)

    @property
    def all_success(self) -> bool:
        return bool(self.results) and all(
            r.verified_success for r in self.results
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_success": self.all_success,
            "any_success": self.any_success,
            "results": [r.as_dict() for r in self.results],
        }
