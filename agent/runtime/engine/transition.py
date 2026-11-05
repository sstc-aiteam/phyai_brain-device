"""
Runtime transition facade.

Owns task-progress, repair-context, and semantic-loop tracking so the
runtime engine does not need to coordinate individual monitor classes.
"""

from __future__ import annotations

from typing import Any

from .monitors.progress import TaskProgressMonitor
from .monitors.repair import (
    PendingIntentProtocolError,
    RepairContextTracker,
)
from .monitors.loop import SemanticLoopMonitor


class TransitionTracker:
    """
    Facade over runtime transition monitors.

    This class intentionally preserves the existing monitor behavior.
    It only centralizes ownership and access.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._progress = TaskProgressMonitor()
        self._repair = RepairContextTracker()
        self._loop = SemanticLoopMonitor()

    # ---------------------------------------------------------
    # Decision context
    # ---------------------------------------------------------

    def initialize(
        self,
        goals,
        world,
    ) -> None:
        self._progress.initialize(
            goals,
            world,
        )

    def progress_view(
        self,
        goals,
        world,
    ) -> dict[str, Any]:
        return self._progress.brain_view(
            goals,
            world,
        )

    def repair_view(
        self,
        world,
        *,
        iteration: int,
    ) -> dict[str, Any]:
        return self._repair.brain_view(
            world,
            iteration=iteration,
        )

    def loop_view(self) -> dict[str, Any]:
        return self._loop.brain_view()

    # ---------------------------------------------------------
    # Rejection / continuity
    # ---------------------------------------------------------

    def record_rejection(
        self,
        feedback: dict[str, Any],
        *,
        iteration: int,
    ) -> None:
        self._repair.record_rejection(
            feedback,
            iteration=iteration,
        )

    def validate_decision_continuity(
        self,
        dispatch,
        *,
        world,
        iteration: int,
    ) -> None:
        self._repair.validate_decision_continuity(
            dispatch,
            world=world,
            iteration=iteration,
        )

    # ---------------------------------------------------------
    # Post-execution transition
    # ---------------------------------------------------------

    def observe_transition(self, **kwargs):
        return self._progress.observe_transition(
            **kwargs
        )

    def observe_execution(
        self,
        report,
        *,
        world,
        iteration: int,
    ) -> None:
        self._repair.observe_execution(
            report,
            world=world,
            iteration=iteration,
        )

    def observe_loop(self, **kwargs):
        return self._loop.observe(
            **kwargs
        )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    def is_complete(
        self,
        goals,
        world,
    ) -> bool:
        return self._progress.is_complete(
            goals,
            world,
        )


__all__ = [
    "PendingIntentProtocolError",
    "TransitionTracker",
]
