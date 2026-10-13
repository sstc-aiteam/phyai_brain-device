"""
Runtime dispatch guard facade.

Owns the deterministic pre-execution pipeline:

    planner decision
        -> device binding
        -> runtime validation
        -> executable BoundDispatch

A relevance guard can be inserted here later without changing the
RuntimeEngine orchestration interface.
"""

from __future__ import annotations

from ..domain.action import (
    BoundDispatch,
    DispatchDecision,
)
from ..domain.task import (
    CoordinationSpec,
    GoalSet,
)
from ..domain.tool import ToolCatalog
from ..domain.world import WorldState

from .guards.binding import DeviceBinder
from .guards.validation import RuntimeValidator


class DispatchGuard:
    """
    Facade for deterministic dispatch preparation.
    """

    def __init__(
        self,
        *,
        binder: DeviceBinder,
        validator: RuntimeValidator | None = None,
    ) -> None:
        self._binder = binder
        self._validator = (
            validator
            if validator is not None
            else RuntimeValidator()
        )

    def device_context(
        self,
        world: WorldState,
    ):
        """
        Device information exposed to RuntimeBrain.
        """
        return self._binder.for_brain(
            world
        )

    def prepare(
        self,
        dispatch: DispatchDecision,
        *,
        world: WorldState,
        goals: GoalSet,
        tools: ToolCatalog,
        coordination: CoordinationSpec,
    ) -> BoundDispatch:
        """
        Bind a planner decision to devices and validate it against
        the current observed WorldState.
        """

        bound = self._binder.bind(
            dispatch,
            world=world,
            tools=tools,
            coordination=coordination,
        )

        self._validator.validate(
            bound,
            world=world,
            goals=goals,
            tools=tools,
            coordination=coordination,
        )

        return bound


__all__ = [
    "DispatchGuard",
]
