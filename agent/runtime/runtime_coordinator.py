from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .coordination_schema import CoordinationSpec
from .decision_retry_policy import DecisionRetryPolicy
from .device_binding import DeviceBinder
from .dispatch_schema import BoundDispatch, DispatchDecision
from .goal_schema import GoalSet
from .parallel_validator import validate_dispatch
from .loop_monitor import SemanticLoopMonitor
from .progress_monitor import TaskProgressMonitor
from .repair_context import (
    PendingIntentProtocolError,
    RepairContextTracker,
)
from .tool_model import ToolCatalog
from .world_state import WorldState


@dataclass
class RuntimeTraceRow:
    iteration: int
    world_before: dict[str, Any]
    goal_state_before: list[dict[str, Any]]
    dispatch: dict[str, Any] | None = None
    bound_dispatch: dict[str, Any] | None = None
    rejected_dispatches: list[dict[str, Any]] = field(default_factory=list)
    execution: dict[str, Any] | None = None
    progress_feedback: dict[str, Any] | None = None
    repair_context_before: dict[str, Any] | None = None
    repair_context_after: dict[str, Any] | None = None
    loop_context_before: dict[str, Any] | None = None
    loop_context_after: dict[str, Any] | None = None
    executive_protocol_retries: int = 0
    world_after: dict[str, Any] | None = None
    decision_parallel_limit: int | None = None
    parallel_fallback_used: bool = False


@dataclass
class RuntimeRunResult:
    success: bool
    world: WorldState
    iterations: int
    trace: list[RuntimeTraceRow]


class RuntimeCoordinator:
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
        self.brain = brain
        self.binder = binder
        self.executor = executor
        self.tools = tools
        self.max_parallel_actions = max_parallel_actions
        self.max_decision_attempts = max_decision_attempts
        self.retry_policy = DecisionRetryPolicy(
            max_attempts=max_decision_attempts,
            parallel_fallback_after_rejections=(
                parallel_fallback_after_rejections
            ),
            fallback_max_parallel_actions=(
                fallback_max_parallel_actions
            ),
        )
        self.progress_monitor = TaskProgressMonitor()
        self.repair_tracker = RepairContextTracker()
        self.loop_monitor = SemanticLoopMonitor()

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
        rejected: list[dict[str, Any]] = []
        hard_rejected_count = 0
        fallback_used = False

        for _attempt in range(1, self.retry_policy.max_attempts + 1):
            effective_parallel_limit, fallback_active = (
                self.retry_policy.parallel_limit(
                    configured_max_parallel_actions=(
                        self.max_parallel_actions
                    ),
                    rejected_count=hard_rejected_count,
                )
            )
            fallback_used = fallback_used or fallback_active

            try:
                dispatch = self.brain.decide_dispatch(
                    world=world,
                    goals=goals,
                    tools=self.tools,
                    coordination=coordination,
                    device_context=self.binder.for_brain(world),
                    rejected_dispatches=rejected,
                    task_progress_state=(
                        self.progress_monitor.brain_view(
                            goals,
                            world,
                        )
                    ),
                    repair_context=(
                        self.repair_tracker.brain_view(
                            world,
                            iteration=iteration,
                        )
                    ),
                    loop_context=self.loop_monitor.brain_view(),
                    max_parallel_actions=effective_parallel_limit,
                )
            except Exception as exc:
                rejected.append({
                    "dispatch": None,
                    "reason": (
                        "BRAIN_OUTPUT_ERROR: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "hard_reject": True,
                    "max_parallel_actions": effective_parallel_limit,
                    "parallel_fallback_active": fallback_active,
                })
                hard_rejected_count += 1
                continue

            try:
                self.repair_tracker.validate_decision_continuity(
                    dispatch,
                    world=world,
                    iteration=iteration,
                )
            except PendingIntentProtocolError as exc:
                rejected.append({
                    "dispatch": dispatch.as_dict(),
                    "reason": (
                        "EXECUTIVE_PROTOCOL_RETRY: "
                        f"{exc}"
                    ),
                    "structured_feedback": exc.feedback,
                    "hard_reject": False,
                    "stage": "executive_protocol",
                    "max_parallel_actions": (
                        effective_parallel_limit
                    ),
                    "parallel_fallback_active": False,
                })
                continue

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
                return (
                    dispatch,
                    bound,
                    rejected,
                    fallback_used,
                    effective_parallel_limit,
                )
            except Exception as exc:
                structured_feedback = getattr(
                    exc,
                    "feedback",
                    None,
                )
                clean_feedback = (
                    structured_feedback
                    if isinstance(structured_feedback, dict)
                    else {}
                )
                self.repair_tracker.record_rejection(
                    clean_feedback,
                    iteration=iteration,
                )
                rejected.append({
                    "dispatch": dispatch.as_dict(),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "structured_feedback": clean_feedback,
                    "hard_reject": True,
                    "max_parallel_actions": effective_parallel_limit,
                    "parallel_fallback_active": fallback_active,
                })
                hard_rejected_count += 1

        raise RuntimeError(
            "runtime brain failed to produce a bindable/valid "
            f"dispatch after {self.retry_policy.max_attempts} attempts; "
            f"rejected={rejected!r}"
        )

    def run_cycle(
        self,
        *,
        iteration: int,
        world: WorldState,
        goals: GoalSet,
        coordination: CoordinationSpec | None = None,
    ) -> tuple[WorldState, RuntimeTraceRow]:
        coordination = coordination or CoordinationSpec()
        self.progress_monitor.initialize(goals, world)

        row = RuntimeTraceRow(
            iteration=iteration,
            world_before=world.snapshot(),
            goal_state_before=goals.summary(world),
            repair_context_before=(
                self.repair_tracker.brain_view(
                    world,
                    iteration=iteration,
                )
            ),
            loop_context_before=(
                self.loop_monitor.brain_view()
            ),
        )

        dispatch, bound, rejected, fallback_used, decision_parallel_limit = (
            self._decide_and_bind(
                world=world,
                goals=goals,
                coordination=coordination,
                iteration=iteration,
            )
        )

        row.dispatch = dispatch.as_dict()
        row.bound_dispatch = bound.as_dict()
        row.rejected_dispatches = rejected
        row.parallel_fallback_used = fallback_used
        row.decision_parallel_limit = decision_parallel_limit

        updated, report = self.executor.execute(
            bound,
            world=world,
            tools=self.tools,
        )
        goals_after = goals
        feedback = self.progress_monitor.observe_transition(
            goals_before=goals,
            goals_after=goals_after,
            world_before=world,
            world_after=updated,
            dispatch=bound,
            report=report,
            tools=self.tools,
        )

        credited_goal_facts = []
        for goal_id in feedback.newly_credited_goals:
            try:
                goal = goals_after.by_id(goal_id)
            except KeyError:
                continue
            credited_goal_facts.append({
                "goal_id": goal.goal_id,
                "subject": goal.subject,
                "field": goal.field,
            })

        self.repair_tracker.observe_execution(
            report,
            world=updated,
            iteration=iteration,
            credited_goal_facts=credited_goal_facts,
        )
        loop_feedback = self.loop_monitor.observe(
            iteration=iteration,
            dispatch=bound,
            report=report,
            progress=feedback,
            world_before=world,
            world_after=updated,
        )

        row.execution = report.as_dict()
        row.progress_feedback = feedback.as_dict()
        row.repair_context_after = (
            self.repair_tracker.brain_view(
                updated,
                iteration=iteration,
            )
        )
        row.loop_context_after = loop_feedback
        row.executive_protocol_retries = sum(
            1
            for item in row.rejected_dispatches
            if item.get("stage") == "executive_protocol"
        )
        row.world_after = updated.snapshot()
        return updated, row

    def run_until_done(
        self,
        *,
        world: WorldState,
        goals: GoalSet,
        coordination: CoordinationSpec | None = None,
        max_iterations: int = 50,
    ) -> RuntimeRunResult:
        coordination = coordination or CoordinationSpec()
        current = world
        trace: list[RuntimeTraceRow] = []
        self.progress_monitor.initialize(goals, current)

        for iteration in range(1, max_iterations + 1):
            if self.progress_monitor.is_complete(goals, current):
                return RuntimeRunResult(
                    success=True,
                    world=current,
                    iterations=iteration - 1,
                    trace=trace,
                )

            current, row = self.run_cycle(
                iteration=iteration,
                world=current,
                goals=goals,
                coordination=coordination,
            )
            trace.append(row)

        return RuntimeRunResult(
            success=self.progress_monitor.is_complete(goals, current),
            world=current,
            iterations=max_iterations,
            trace=trace,
        )
