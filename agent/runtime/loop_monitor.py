from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .dispatch_schema import BoundDispatch, DispatchExecutionReport
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
            "action_signatures": list(
                self.action_signatures
            ),
            "world_before_hash": self.world_before_hash,
            "world_after_hash": self.world_after_hash,
            "progress_class": self.progress_class,
            "newly_credited_goals": list(
                self.newly_credited_goals
            ),
            "verified_success": self.verified_success,
        }


class SemanticLoopMonitor:
    """Remember legal-but-unproductive execution history.

    This monitor NEVER blocks an action.  It only tells the brain when recent
    successful actions revisited semantic World states without task credit.
    """

    def __init__(
        self,
        *,
        history_limit: int = 8,
    ):
        self.history_limit = history_limit
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
        signatures = [
            str(action.action)
            for action in dispatch.actions
        ]
        verified_success = report.any_success

        event = LoopEvent(
            iteration=iteration,
            action_signatures=signatures,
            world_before_hash=_world_hash(world_before),
            world_after_hash=_world_hash(world_after),
            progress_class=progress.progress_class,
            newly_credited_goals=list(
                progress.newly_credited_goals
            ),
            verified_success=verified_success,
        )
        self._history.append(event)
        self._history = self._history[-self.history_limit:]

        if (
            verified_success
            and not progress.newly_credited_goals
            and progress.progress_class
            != "execution_failure"
        ):
            self.no_progress_streak += 1
        elif progress.newly_credited_goals:
            self.no_progress_streak = 0

        view = self.brain_view()
        cycle = view.get("detected_cycle")
        new_loop_event = False
        if isinstance(cycle, dict):
            key = tuple(cycle.get("world_hash_cycle", []))
            if key and key != self._last_cycle_key:
                self.loop_event_count += 1
                self._last_cycle_key = key
                new_loop_event = True
        else:
            self._last_cycle_key = None

        return {
            **view,
            "new_loop_event": new_loop_event,
        }

    def _detect_cycle(self) -> dict[str, Any] | None:
        # Detect recent ABAB or ABCABC semantic-state cycles.
        successful = [
            e
            for e in self._history
            if e.verified_success
            and e.progress_class != "execution_failure"
        ]
        if len(successful) < 4:
            return None

        hashes = [
            e.world_after_hash
            for e in successful
        ]

        for cycle_len in (2, 3):
            needed = cycle_len * 2
            if len(hashes) < needed:
                continue
            tail = hashes[-needed:]
            if tail[:cycle_len] != tail[cycle_len:]:
                continue

            events = successful[-needed:]
            if any(
                e.newly_credited_goals
                for e in events
            ):
                continue

            return {
                "cycle_length": cycle_len,
                "world_hash_cycle": tail[:cycle_len],
                "iterations": [
                    e.iteration
                    for e in events
                ],
                "actions": [
                    list(e.action_signatures)
                    for e in events
                ],
                "task_credit_during_cycle": [],
            }

        return None

    def brain_view(self) -> dict[str, Any]:
        cycle = self._detect_cycle()
        return {
            "loop_detected": cycle is not None,
            "detected_cycle": cycle,
            "no_progress_streak": self.no_progress_streak,
            "recent_executions": [
                event.as_dict()
                for event in self._history[-6:]
            ],
            "instruction": (
                "This is execution-history feedback only. Legal actions are "
                "not blocked. If a loop is detected, choose a materially "
                "different strategy unless the repetition is required to "
                "repair an explicit pending physical precondition."
            ),
        }
