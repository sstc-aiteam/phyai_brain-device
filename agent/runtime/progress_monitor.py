from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dispatch_schema import BoundDispatch, DispatchExecutionReport
from .goal_schema import GoalCondition, GoalSet
from .tool_model import ToolCatalog
from .world_state import WorldState


def _flatten_world(world: WorldState) -> dict[tuple[str, str], Any]:
    flat: dict[tuple[str, str], Any] = {}
    snapshot = world.snapshot().get("entities", {})
    for entity_id, entity in snapshot.items():
        if not isinstance(entity, dict):
            continue
        for field_name, value in entity.items():
            flat[(str(entity_id), str(field_name))] = value
    return flat


def _changed_facts(
    before: WorldState,
    after: WorldState,
) -> list[dict[str, Any]]:
    left = _flatten_world(before)
    right = _flatten_world(after)
    keys = sorted(set(left) | set(right))
    rows: list[dict[str, Any]] = []
    for key in keys:
        old = left.get(key)
        new = right.get(key)
        if old == new:
            continue
        rows.append({
            "subject": key[0],
            "field": key[1],
            "before": old,
            "after": new,
        })
    return rows


@dataclass
class ProgressFeedback:
    world_changed: bool
    changed_facts: list[dict[str, Any]] = field(default_factory=list)
    newly_credited_goals: list[str] = field(default_factory=list)
    regressed_goals: list[str] = field(default_factory=list)
    out_of_order_goal_events: list[dict[str, Any]] = field(default_factory=list)
    credited_goals: list[str] = field(default_factory=list)
    uncredited_physically_satisfied_goals: list[str] = field(default_factory=list)
    task_complete: bool = False
    progress_class: str = "none"
    verified_success_count: int = 0
    failed_action_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "world_changed": self.world_changed,
            "changed_facts": self.changed_facts,
            "newly_credited_goals": self.newly_credited_goals,
            "regressed_goals": self.regressed_goals,
            "out_of_order_goal_events": self.out_of_order_goal_events,
            "credited_goals": self.credited_goals,
            "uncredited_physically_satisfied_goals": (
                self.uncredited_physically_satisfied_goals
            ),
            "task_complete": self.task_complete,
            "progress_class": self.progress_class,
            "verified_success_count": self.verified_success_count,
            "failed_action_count": self.failed_action_count,
        }


class TaskProgressMonitor:
    """Observe task progress without blocking legal actions.

    The validator answers only: "can this action physically/safely execute now?"
    This monitor answers a separate question after execution:
    "did the task make progress, regress, or complete an ordered milestone?"

    Goal dependencies are interpreted as temporal credit dependencies for
    evaluation/feedback.  A goal event completed before its dependencies is
    physically allowed, but it is not credited as ordered task progress until
    the milestone is performed again after prerequisites are credited.
    """

    def __init__(self):
        self.credited_goal_ids: set[str] = set()
        self._known_goals: dict[str, GoalCondition] = {}
        self._initialized = False
        self.last_feedback: ProgressFeedback | None = None

    def _register(self, goals: GoalSet) -> None:
        for goal in goals.conditions:
            self._known_goals[goal.goal_id] = goal

    def initialize(self, goals: GoalSet, world: WorldState) -> None:
        self._register(goals)
        if self._initialized:
            return

        # Credit facts that are already true at task start, but only when
        # their dependency chain is also initially true/credited.
        changed = True
        while changed:
            changed = False
            for goal in goals.conditions:
                if goal.goal_id in self.credited_goal_ids:
                    continue
                if not goal.is_satisfied(world):
                    continue
                if not set(goal.depends_on).issubset(
                    self.credited_goal_ids
                ):
                    continue
                self.credited_goal_ids.add(goal.goal_id)
                changed = True

        self._initialized = True

    def _remove_invalid_credits(
        self,
        goals: GoalSet,
        world: WorldState,
    ) -> set[str]:
        known_current = {g.goal_id: g for g in goals.conditions}
        removed: set[str] = set()

        # First remove credits whose physical goal fact regressed.
        for goal_id in list(self.credited_goal_ids):
            goal = known_current.get(goal_id) or self._known_goals.get(goal_id)
            if goal is None:
                continue
            if not goal.is_satisfied(world):
                self.credited_goal_ids.remove(goal_id)
                removed.add(goal_id)

        # Then remove downstream credits whose prerequisite credit was lost.
        changed = True
        while changed:
            changed = False
            for goal_id in list(self.credited_goal_ids):
                goal = known_current.get(goal_id) or self._known_goals.get(goal_id)
                if goal is None:
                    continue
                if not set(goal.depends_on).issubset(
                    self.credited_goal_ids
                ):
                    self.credited_goal_ids.remove(goal_id)
                    removed.add(goal_id)
                    changed = True

        return removed

    def is_complete(self, goals: GoalSet, world: WorldState) -> bool:
        self._register(goals)
        goal_ids = {g.goal_id for g in goals.conditions}
        return (
            goal_ids.issubset(self.credited_goal_ids)
            and all(g.is_satisfied(world) for g in goals.conditions)
        )

    def brain_view(self, goals: GoalSet, world: WorldState) -> dict[str, Any]:
        self._register(goals)
        satisfied = goals.satisfied_ids(world)
        goal_ids = {g.goal_id for g in goals.conditions}
        return {
            "credited_goals": sorted(
                self.credited_goal_ids & goal_ids
            ),
            "uncredited_physically_satisfied_goals": sorted(
                satisfied - self.credited_goal_ids
            ),
            "task_complete": self.is_complete(goals, world),
            "last_execution_feedback": (
                self.last_feedback.as_dict()
                if self.last_feedback is not None
                else None
            ),
        }

    def observe_transition(
        self,
        *,
        goals_before: GoalSet,
        goals_after: GoalSet,
        world_before: WorldState,
        world_after: WorldState,
        dispatch: BoundDispatch,
        report: DispatchExecutionReport,
        tools: ToolCatalog,
    ) -> ProgressFeedback:
        self.initialize(goals_before, world_before)
        self._register(goals_after)

        credited_before = set(self.credited_goal_ids)
        changed_facts = _changed_facts(world_before, world_after)
        changed_fact_keys = {
            (row["subject"], row["field"])
            for row in changed_facts
        }

        successful_write_facts: set[tuple[str, str]] = set()
        for result in report.results:
            if not result.verified_success:
                continue
            tool = tools.get(
                result.bound_action.action.function_name
            )
            successful_write_facts |= tool.write_facts(
                result.bound_action
            )

        regressed = self._remove_invalid_credits(
            goals_after,
            world_after,
        )

        before_by_id = {
            g.goal_id: g for g in goals_before.conditions
        }
        after_by_id = {
            g.goal_id: g for g in goals_after.conditions
        }

        newly_credited: set[str] = set()
        out_of_order: list[dict[str, Any]] = []

        # Credit only goals that had a task-relevant event in THIS dispatch:
        # their fact changed, their fact was written by a verified action, or
        # they transitioned from false to true.
        for goal in goals_after.conditions:
            if goal.goal_id in self.credited_goal_ids:
                continue
            if not goal.is_satisfied(world_after):
                continue

            fact_key = (goal.subject, goal.field)
            before_goal = before_by_id.get(goal.goal_id)
            was_satisfied = (
                before_goal.is_satisfied(world_before)
                if before_goal is not None
                else False
            )
            had_event = (
                fact_key in changed_fact_keys
                or fact_key in successful_write_facts
                or not was_satisfied
            )
            if not had_event:
                continue

            # Ordered dependencies must have been credited BEFORE this
            # dispatch. Same-dispatch completion does not retroactively make
            # an earlier/parallel milestone ordered.
            missing = [
                dep
                for dep in goal.depends_on
                if dep not in credited_before
            ]
            if missing:
                out_of_order.append({
                    "goal_id": goal.goal_id,
                    "missing_credited_dependencies": missing,
                })
                continue

            self.credited_goal_ids.add(goal.goal_id)
            newly_credited.add(goal.goal_id)

        physically_satisfied_after = goals_after.satisfied_ids(
            world_after
        )
        current_goal_ids = {
            g.goal_id for g in goals_after.conditions
        }
        uncredited_satisfied = (
            physically_satisfied_after - self.credited_goal_ids
        )

        task_complete = self.is_complete(
            goals_after,
            world_after,
        )

        verified_success_count = sum(
            1 for result in report.results if result.verified_success
        )
        failed_action_count = len(report.results) - verified_success_count

        if verified_success_count == 0 and failed_action_count > 0 and not changed_facts:
            progress_class = "execution_failure"
        elif regressed:
            progress_class = "regression"
        elif out_of_order:
            progress_class = "out_of_order"
        elif newly_credited:
            progress_class = "positive"
        elif changed_facts:
            progress_class = "state_changed_no_direct_goal_credit"
        else:
            progress_class = "no_world_change"

        feedback = ProgressFeedback(
            world_changed=bool(changed_facts),
            changed_facts=changed_facts[:40],
            newly_credited_goals=sorted(newly_credited),
            regressed_goals=sorted(regressed),
            out_of_order_goal_events=out_of_order,
            credited_goals=sorted(
                self.credited_goal_ids & current_goal_ids
            ),
            uncredited_physically_satisfied_goals=sorted(
                uncredited_satisfied
            ),
            task_complete=task_complete,
            progress_class=progress_class,
            verified_success_count=verified_success_count,
            failed_action_count=failed_action_count,
        )
        self.last_feedback = feedback
        return feedback
