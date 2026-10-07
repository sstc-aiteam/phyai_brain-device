"""
Semantic loop detection for the closed-loop runtime.

This module is intentionally small.  It never blocks execution and never
chooses an action.  It only summarizes recent executed transitions for Brain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .dispatch import (
    BoundDispatch,
    DispatchExecutionReport,
)
from .progress_monitor import ProgressFeedback
from .world_state import WorldState


def _world_hash(world: WorldState) -> str:
    payload = json.dumps(
        world.snapshot(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(
        payload.encode("utf-8")
    ).hexdigest()[:16]


def _action_signature(bound_action) -> str:
    return json.dumps(
        bound_action.action.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass
class LoopEvent:
    iteration: int
    action_signatures: list[str]
    world_before_hash: str
    world_after_hash: str
    progress_class: str
    newly_credited_goals: list[str]
    verified_success: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "action_signatures": list(self.action_signatures),
            "world_before_hash": self.world_before_hash,
            "world_after_hash": self.world_after_hash,
            "progress_class": self.progress_class,
            "newly_credited_goals": list(self.newly_credited_goals),
            "verified_success": self.verified_success,
        }


class SemanticLoopMonitor:
    """
    Detect recent legal-but-unproductive semantic cycles.

    It reports evidence only:
        - repeated semantic World states
        - no task credit
        - no-progress streak

    Validator / Coordinator decide what to do with that evidence.
    """

    def __init__(
        self,
        *,
        history_limit: int = 8,
    ):
        if history_limit < 4:
            raise ValueError(
                "history_limit must be >= 4"
            )

        self.history_limit = int(history_limit)
        self._history: list[LoopEvent] = []

        self.loop_event_count = 0
        self.no_progress_streak = 0
        self._last_cycle_key: tuple[str, ...] | None = None

    def observe(
        self,
        *,
        iteration: int,
        dispatch: BoundDispatch,
        report: DispatchExecutionReport,
        progress: ProgressFeedback,
        world_before: WorldState,
        world_after: WorldState,
    ) -> dict[str, Any]:
        event = LoopEvent(
            iteration=iteration,
            action_signatures=[
                _action_signature(bound)
                for bound in dispatch.actions
            ],
            world_before_hash=_world_hash(world_before),
            world_after_hash=_world_hash(world_after),
            progress_class=progress.progress_class,
            newly_credited_goals=list(
                progress.newly_credited_goals
            ),
            verified_success=report.any_success,
        )

        self._history.append(event)
        self._history = self._history[
            -self.history_limit:
        ]

        if (
            event.verified_success
            and not event.newly_credited_goals
            and event.progress_class != "execution_failure"
        ):
            self.no_progress_streak += 1
        else:
            self.no_progress_streak = 0

        view = self.brain_view()
        cycle = view["detected_cycle"]

        new_loop_event = False

        if cycle is not None:
            key = (
                str(cycle["cycle_length"]),
                *cycle["world_hash_cycle"],
            )

            if key != self._last_cycle_key:
                self.loop_event_count += 1
                self._last_cycle_key = key
                new_loop_event = True
        else:
            self._last_cycle_key = None

        return {
            **view,
            # Keep detailed history in trace/debug output, but do not duplicate
            # it in the Brain payload. ExecutionMemory is the canonical
            # Brain-facing execution history.
            "recent_executions": [
                event.as_dict()
                for event in self._history[-6:]
            ],
            "new_loop_event": new_loop_event,
        }

    def _detect_cycle(
        self,
    ) -> dict[str, Any] | None:
        """
        Detect contiguous ABAB or ABCABC world-state cycles.

        Every event inside the cycle must:
        - have at least one verified-success action;
        - have no execution_failure classification;
        - earn no task credit.
        """

        for cycle_len in (2, 3):
            needed = cycle_len * 2

            if len(self._history) < needed:
                continue

            events = self._history[-needed:]

            if any(
                not event.verified_success
                or event.progress_class == "execution_failure"
                or event.newly_credited_goals
                for event in events
            ):
                continue

            hashes = [
                event.world_after_hash
                for event in events
            ]

            if hashes[:cycle_len] != hashes[cycle_len:]:
                continue

            return {
                "cycle_length": cycle_len,
                "world_hash_cycle": hashes[:cycle_len],
                "iterations": [
                    event.iteration
                    for event in events
                ],
                "actions": [
                    list(event.action_signatures)
                    for event in events
                ],
                "task_credit_during_cycle": [],
            }

        return None

    def brain_view(
        self,
    ) -> dict[str, Any]:
        cycle = self._detect_cycle()

        return {
            "loop_detected": cycle is not None,
            "detected_cycle": cycle,
            "no_progress_streak": self.no_progress_streak,
            "instruction": (
                "Execution-history evidence only. "
                "If a loop is detected, avoid repeating the same "
                "state/action cycle unless required by an active repair."
            ),
        }

    def clear(
        self,
    ) -> None:
        self._history.clear()
        self.no_progress_streak = 0
        self._last_cycle_key = None


__all__ = [
    "LoopEvent",
    "SemanticLoopMonitor",
]
