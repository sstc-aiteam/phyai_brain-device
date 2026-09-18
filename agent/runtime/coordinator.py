"""
Closed-loop runtime coordinator.

The coordinator owns orchestration, not domain logic.

Per cycle:

    CURRENT WorldState
        ↓
    Brain decides one dispatch
        ↓
    Device Binder assigns concrete devices
        ↓
    Validator performs final deterministic checks
        ↓
    Executor executes callbacks and returns FRESH observed WorldState
        ↓
    Progress Monitor evaluates task progress
        ↓
    Repair / Loop monitors update executive context
        ↓
    next cycle

The coordinator never:
- calls perception/camera logic directly
- chooses a robot arm itself
- defines tool preconditions/effects
- treats callback success as WorldState truth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .coordination import CoordinationSpec
from .device_binding import DeviceBinder
from .dispatch import (
    BoundDispatch,
    DispatchDecision,
)
from .goals import GoalSet
from .loop_monitor import SemanticLoopMonitor
from .progress_monitor import TaskProgressMonitor
from .repair_context import (
    PendingIntentProtocolError,
    RepairContextTracker,
)
from .tool_model import ToolCatalog
from .validator import validate_dispatch
from .world_state import WorldState


@dataclass
class RuntimeTraceRow:
    iteration: int
    world_before: dict[str, Any]
    goal_state_before: list[dict[str, Any]]

    dispatch: dict[str, Any] | None = None
    bound_dispatch: dict[str, Any] | None = None

    rejected_dispatches: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    execution: dict[str, Any] | None = None
    progress_feedback: dict[str, Any] | None = None

    repair_context_before: dict[str, Any] | None = None
    repair_context_after: dict[str, Any] | None = None

    loop_context_before: dict[str, Any] | None = None
    loop_context_after: dict[str, Any] | None = None

    world_after: dict[str, Any] | None = None

    executive_protocol_retries: int = 0
    decision_parallel_limit: int | None = None
    parallel_fallback_used: bool = False


@dataclass
class RuntimeRunResult:
    success: bool
    world: WorldState
    iterations: int
    trace: list[RuntimeTraceRow]


class RuntimeCoordinator:
    """
    Highest-level closed-loop executive.

    `brain`, `binder`, and `executor` are injected so runtime orchestration is
    independent from a specific LLM provider, hardware allocator, or robot
    callback implementation.
    """

    def __init__(
        self,
        *,
        brain,
        binder: DeviceBinder,
        executor,
        tools: ToolCatalog,
        max_parallel_actions: int = 2,
        max_decision_attempts: int = 5,
        parallel_fallback_after_rejections: int = 2,
        fallback_max_parallel_actions: int = 1,
    ):
        if max_parallel_actions < 1:
            raise ValueError(
                "max_parallel_actions must be >= 1"
            )

        if max_decision_attempts < 1:
            raise ValueError(
                "max_decision_attempts must be >= 1"
            )

        if parallel_fallback_after_rejections < 0:
            raise ValueError(
                "parallel_fallback_after_rejections must be >= 0"
            )

        if fallback_max_parallel_actions < 1:
            raise ValueError(
                "fallback_max_parallel_actions must be >= 1"
            )

        self.brain = brain
        self.binder = binder
        self.executor = executor
        self.tools = tools

        self.max_parallel_actions = int(
            max_parallel_actions
        )

        self.max_decision_attempts = int(
            max_decision_attempts
        )

        self.parallel_fallback_after_rejections = int(
            parallel_fallback_after_rejections
        )

        self.fallback_max_parallel_actions = min(
            int(
                fallback_max_parallel_actions
            ),
            self.max_parallel_actions,
        )

        self._reset_monitors()

    def _reset_monitors(
        self,
    ) -> None:
        self.progress_monitor = (
            TaskProgressMonitor()
        )

        self.repair_tracker = (
            RepairContextTracker()
        )

        self.loop_monitor = (
            SemanticLoopMonitor()
        )

    def reset(
        self,
    ) -> None:
        """
        Clear task-local runtime memory before starting an unrelated task.
        """

        self._reset_monitors()

    def _parallel_limit(
        self,
        *,
        hard_rejected_count: int,
    ) -> tuple[int, bool]:
        """
        Reduce parallelism after repeated hard bind/validation rejections.

        Executive-protocol retries and malformed Brain output do not trigger
        this fallback because they are not evidence of a parallel resource
        conflict.
        """

        fallback_active = (
            hard_rejected_count
            >= self.parallel_fallback_after_rejections
        )

        if not fallback_active:
            return (
                self.max_parallel_actions,
                False,
            )

        return (
            self.fallback_max_parallel_actions,
            True,
        )

    def _decide_and_bind(
        self,
        *,
        world: WorldState,
        goals: GoalSet,
        coordination: CoordinationSpec,
        iteration: int,
    ) -> tuple[
        DispatchDecision,
        BoundDispatch,
        list[dict[str, Any]],
        bool,
        int,
    ]:
        """
        Ask Brain for a dispatch and deterministically turn it into a valid
        BoundDispatch.

        Rejections are fed back to Brain inside the SAME world iteration.
        No physical execution occurs until this method returns successfully.
        """

        rejected: list[
            dict[str, Any]
        ] = []

        hard_rejected_count = 0
        fallback_used = False

        for attempt in range(
            1,
            self.max_decision_attempts + 1,
        ):
            (
                effective_parallel_limit,
                fallback_active,
            ) = self._parallel_limit(
                hard_rejected_count=
                    hard_rejected_count,
            )

            fallback_used = (
                fallback_used
                or fallback_active
            )

            # ------------------------------------------------
            # 1. Brain: WHAT should start now?
            # ------------------------------------------------

            try:
                dispatch = (
                    self.brain.decide_dispatch(
                        world=world,
                        goals=goals,
                        tools=self.tools,
                        coordination=coordination,
                        device_context=
                            self.binder.for_brain(
                                world
                            ),
                        rejected_dispatches=
                            rejected,
                        task_progress_state=
                            self.progress_monitor.brain_view(
                                goals,
                                world,
                            ),
                        repair_context=
                            self.repair_tracker.brain_view(
                                world,
                                iteration=iteration,
                            ),
                        loop_context=
                            self.loop_monitor.brain_view(),
                        max_parallel_actions=
                            effective_parallel_limit,
                    )
                )

            except Exception as exc:
                rejected.append({
                    "dispatch":
                        None,
                    "reason":
                        (
                            "BRAIN_OUTPUT_ERROR: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    "structured_feedback":
                        {},
                    "hard_reject":
                        False,
                    "stage":
                        "brain",
                    "attempt":
                        attempt,
                    "max_parallel_actions":
                        effective_parallel_limit,
                    "parallel_fallback_active":
                        fallback_active,
                })

                continue

            # ------------------------------------------------
            # 2. Binder: WHO performs each action?
            # 3. Validator: may the bound dispatch start NOW?
            # ------------------------------------------------
            #
            # Important ordering:
            # bind/validate happens BEFORE pending-intent resolution is
            # committed.  Therefore an explicit "abandon" does not erase a
            # repair frame when the replacement dispatch itself is invalid.
            # ------------------------------------------------

            try:
                bound = self.binder.bind(
                    dispatch,
                    world=world,
                    tools=self.tools,
                    coordination=coordination,
                )

                validate_dispatch(
                    bound,
                    world=world,
                    goals=goals,
                    tools=self.tools,
                    coordination=coordination,
                )

            except Exception as exc:
                raw_feedback = getattr(
                    exc,
                    "feedback",
                    None,
                )

                structured_feedback = (
                    dict(
                        raw_feedback
                    )
                    if isinstance(
                        raw_feedback,
                        dict,
                    )
                    else {}
                )

                self.repair_tracker.record_rejection(
                    structured_feedback,
                    iteration=iteration,
                )

                rejected.append({
                    "dispatch":
                        dispatch.as_dict(),
                    "reason":
                        f"{type(exc).__name__}: {exc}",
                    "structured_feedback":
                        structured_feedback,
                    "hard_reject":
                        True,
                    "stage":
                        "bind_or_validate",
                    "attempt":
                        attempt,
                    "max_parallel_actions":
                        effective_parallel_limit,
                    "parallel_fallback_active":
                        fallback_active,
                })

                hard_rejected_count += 1
                continue

            # ------------------------------------------------
            # 4. Executive pending-intent continuity
            # ------------------------------------------------
            #
            # This is not a physical validation rule.  It protects the Brain's
            # own persistent repair memory.
            # ------------------------------------------------

            try:
                self.repair_tracker.validate_decision_continuity(
                    dispatch,
                    world=world,
                    iteration=iteration,
                )

            except PendingIntentProtocolError as exc:
                rejected.append({
                    "dispatch":
                        dispatch.as_dict(),
                    "reason":
                        (
                            "EXECUTIVE_PROTOCOL_RETRY: "
                            f"{exc}"
                        ),
                    "structured_feedback":
                        dict(
                            exc.feedback
                        ),
                    "hard_reject":
                        False,
                    "stage":
                        "executive_protocol",
                    "attempt":
                        attempt,
                    "max_parallel_actions":
                        effective_parallel_limit,
                    "parallel_fallback_active":
                        fallback_active,
                })

                continue

            return (
                dispatch,
                bound,
                rejected,
                fallback_used,
                effective_parallel_limit,
            )

        raise RuntimeError(
            "runtime brain failed to produce a bindable/valid "
            f"dispatch after {self.max_decision_attempts} attempts; "
            f"rejected={rejected!r}"
        )

    def run_cycle(
        self,
        *,
        iteration: int,
        world: WorldState,
        goals: GoalSet,
        coordination: CoordinationSpec | None = None,
    ) -> tuple[
        WorldState,
        RuntimeTraceRow,
    ]:
        """
        Execute exactly one closed-loop runtime cycle.

        The executor must return a FRESH observed WorldState.  Production
        CallbackExecutor already enforces re-observation after callbacks.
        """

        if iteration < 1:
            raise ValueError(
                "iteration must be >= 1"
            )

        if not isinstance(
            world,
            WorldState,
        ):
            raise TypeError(
                "world must be WorldState"
            )

        if not isinstance(
            goals,
            GoalSet,
        ):
            raise TypeError(
                "goals must be GoalSet"
            )

        coordination = (
            coordination
            or CoordinationSpec()
        )

        self.progress_monitor.initialize(
            goals,
            world,
        )

        row = RuntimeTraceRow(
            iteration=iteration,
            world_before=
                world.snapshot(),
            goal_state_before=
                goals.summary(
                    world
                ),
            repair_context_before=
                self.repair_tracker.brain_view(
                    world,
                    iteration=iteration,
                ),
            loop_context_before=
                self.loop_monitor.brain_view(),
        )

        (
            dispatch,
            bound,
            rejected,
            fallback_used,
            decision_parallel_limit,
        ) = self._decide_and_bind(
            world=world,
            goals=goals,
            coordination=coordination,
            iteration=iteration,
        )

        row.dispatch = (
            dispatch.as_dict()
        )

        row.bound_dispatch = (
            bound.as_dict()
        )

        row.rejected_dispatches = (
            rejected
        )

        row.parallel_fallback_used = (
            fallback_used
        )

        row.decision_parallel_limit = (
            decision_parallel_limit
        )

        # ----------------------------------------------------
        # Execute + fresh observation
        # ----------------------------------------------------

        updated, report = (
            self.executor.execute(
                bound,
                world=world,
                tools=self.tools,
            )
        )

        if not isinstance(
            updated,
            WorldState,
        ):
            raise TypeError(
                "executor.execute() must return "
                "(WorldState, DispatchExecutionReport)"
            )

        # ----------------------------------------------------
        # Evaluate task progress from OBSERVED before/after
        # ----------------------------------------------------

        feedback = (
            self.progress_monitor.observe_transition(
                goals_before=goals,
                goals_after=goals,
                world_before=world,
                world_after=updated,
                dispatch=bound,
                report=report,
                tools=self.tools,
            )
        )

        # ----------------------------------------------------
        # Update persistent executive memories
        # ----------------------------------------------------

        self.repair_tracker.observe_execution(
            report,
            world=updated,
            iteration=iteration,
        )

        loop_feedback = (
            self.loop_monitor.observe(
                iteration=iteration,
                dispatch=bound,
                report=report,
                progress=feedback,
                world_before=world,
                world_after=updated,
            )
        )

        # ----------------------------------------------------
        # Trace
        # ----------------------------------------------------

        row.execution = (
            report.as_dict()
        )

        row.progress_feedback = (
            feedback.as_dict()
        )

        row.repair_context_after = (
            self.repair_tracker.brain_view(
                updated,
                iteration=iteration,
            )
        )

        row.loop_context_after = (
            loop_feedback
        )

        row.executive_protocol_retries = sum(
            1
            for item
            in rejected
            if item.get(
                "stage"
            )
            == "executive_protocol"
        )

        row.world_after = (
            updated.snapshot()
        )

        return (
            updated,
            row,
        )

    def run_until_done(
        self,
        *,
        world: WorldState,
        goals: GoalSet,
        coordination: CoordinationSpec | None = None,
        max_iterations: int = 50,
        reset_runtime_state: bool = True,
    ) -> RuntimeRunResult:
        """
        Run closed-loop cycles until all goals are physically satisfied AND
        task-credited, or until max_iterations is reached.

        `world` must be an already observed initial WorldState.  Subsequent
        fresh observations are owned by the production executor callback path.
        """

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be >= 1"
            )

        if reset_runtime_state:
            self.reset()

        coordination = (
            coordination
            or CoordinationSpec()
        )

        current = (
            world
        )

        trace: list[
            RuntimeTraceRow
        ] = []

        self.progress_monitor.initialize(
            goals,
            current,
        )

        if self.progress_monitor.is_complete(
            goals,
            current,
        ):
            return RuntimeRunResult(
                success=True,
                world=current,
                iterations=0,
                trace=trace,
            )

        for iteration in range(
            1,
            max_iterations + 1,
        ):
            current, row = (
                self.run_cycle(
                    iteration=iteration,
                    world=current,
                    goals=goals,
                    coordination=coordination,
                )
            )

            trace.append(
                row
            )

            if self.progress_monitor.is_complete(
                goals,
                current,
            ):
                return RuntimeRunResult(
                    success=True,
                    world=current,
                    iterations=iteration,
                    trace=trace,
                )

        return RuntimeRunResult(
            success=False,
            world=current,
            iterations=max_iterations,
            trace=trace,
        )


__all__ = [
    "RuntimeTraceRow",
    "RuntimeRunResult",
    "RuntimeCoordinator",
]
