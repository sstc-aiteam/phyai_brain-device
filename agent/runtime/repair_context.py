from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .dispatch_schema import DispatchDecision, DispatchExecutionReport
from .world_state import WorldState


def _action_signature(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    return json.dumps(
        {
            "function_name": action.get("function_name"),
            "arguments": action.get("arguments", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _difference_status(
    difference: dict[str, Any],
    world: WorldState,
) -> dict[str, Any]:
    row = dict(difference)
    subject = row.get("subject")
    field = row.get("field")
    operator = row.get("operator")
    required = row.get("required")

    if not isinstance(subject, str) or not isinstance(field, str):
        row["now_satisfied"] = False
        row["current_now"] = None
        row["current_known_now"] = False
        return row

    known = world.has_field(subject, field)
    current = world.get(subject, field)

    if operator == "ne":
        satisfied = current != required
    elif operator == "if_present_eq":
        satisfied = (not known) or current == required
    elif operator == "if_known_eq":
        satisfied = (
            (not known)
            or required is None
            or current == required
        )
    else:
        # eq and unknown future equality-like operators conservatively use eq.
        satisfied = current == required

    row["current_now"] = current
    row["current_known_now"] = known
    row["now_satisfied"] = bool(satisfied)
    return row


@dataclass
class RepairFrame:
    blocked_action: dict[str, Any]
    state_differences: list[dict[str, Any]]
    created_iteration: int
    last_updated_iteration: int
    rejection_type: str = "PHYSICAL_PRECONDITION_FAILED"
    rejection_count: int = 1

    @property
    def signature(self) -> str:
        return _action_signature(self.blocked_action)


@dataclass
class RepairContextStats:
    frames_created: int = 0
    frames_completed: int = 0
    frames_expired: int = 0
    frames_abandoned: int = 0
    frames_superseded_by_progress: int = 0
    ready_resume_decisions: int = 0
    protocol_retries: int = 0



class PendingIntentProtocolError(RuntimeError):
    """Executive-memory consistency retry, NOT a physical hard reject."""

    def __init__(
        self,
        message: str,
        *,
        feedback: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.feedback = feedback or {}


def _decision_contains_action(
    decision: DispatchDecision,
    action: dict[str, Any],
) -> bool:
    target = _action_signature(action)
    return any(
        _action_signature(candidate.as_dict()) == target
        for candidate in decision.actions
    )


class RepairContextTracker:
    """Persistent memory for the brain's OWN rejected action intentions.

    This is not a planner and does not generate candidate actions.

    When the brain proposes an action that is physically infeasible, the
    runtime stores:
      - the action the brain itself proposed;
      - the CURRENT-vs-REQUIRED state differences returned by validation.

    That context survives successful prerequisite actions and therefore crosses
    World iterations.  The brain must still inspect available_tools and decide
    how to repair the missing state.

    Frames form a stack because repairing one blocked action may itself require
    another action that is temporarily blocked.
    """

    def __init__(
        self,
        *,
        max_depth: int = 6,
        max_age_iterations: int = 12,
    ):
        self.max_depth = max_depth
        self.max_age_iterations = max_age_iterations
        self._stack: list[RepairFrame] = []
        self.stats = RepairContextStats()

    def _expire(self, iteration: int) -> None:
        kept: list[RepairFrame] = []
        for frame in self._stack:
            age = iteration - frame.last_updated_iteration
            if age > self.max_age_iterations:
                self.stats.frames_expired += 1
                continue
            kept.append(frame)
        self._stack = kept[-self.max_depth:]

    def record_rejection(
        self,
        structured_feedback: dict[str, Any] | None,
        *,
        iteration: int,
    ) -> None:
        if not isinstance(structured_feedback, dict):
            return
        if (
            structured_feedback.get("rejection_type")
            != "PHYSICAL_PRECONDITION_FAILED"
        ):
            return

        action = structured_feedback.get("attempted_action")
        differences = structured_feedback.get("state_differences")
        if not isinstance(action, dict):
            return
        if not isinstance(differences, list) or not differences:
            return

        clean_differences = [
            dict(row)
            for row in differences
            if isinstance(row, dict)
        ]
        if not clean_differences:
            return

        self._expire(iteration)
        signature = _action_signature(action)

        if self._stack and self._stack[-1].signature == signature:
            frame = self._stack[-1]
            frame.state_differences = clean_differences
            frame.last_updated_iteration = iteration
            frame.rejection_count += 1
            return

        # If the same blocked intent already exists deeper in the stack, move
        # it to the top rather than duplicating it.
        existing_index = None
        for index, frame in enumerate(self._stack):
            if frame.signature == signature:
                existing_index = index
                break

        if existing_index is not None:
            frame = self._stack.pop(existing_index)
            frame.state_differences = clean_differences
            frame.last_updated_iteration = iteration
            frame.rejection_count += 1
            self._stack.append(frame)
            return

        self._stack.append(
            RepairFrame(
                blocked_action=dict(action),
                state_differences=clean_differences,
                created_iteration=iteration,
                last_updated_iteration=iteration,
                rejection_type=str(
                    structured_feedback.get(
                        "rejection_type",
                        "PHYSICAL_PRECONDITION_FAILED",
                    )
                ),
            )
        )
        self.stats.frames_created += 1
        if len(self._stack) > self.max_depth:
            self._stack = self._stack[-self.max_depth:]

    def _active_view(
        self,
        world: WorldState,
        *,
        iteration: int,
    ) -> dict[str, Any] | None:
        view = self.brain_view(
            world,
            iteration=iteration,
        )
        active = view.get("active_frame")
        return active if isinstance(active, dict) else None

    def validate_decision_continuity(
        self,
        decision: DispatchDecision,
        *,
        world: WorldState,
        iteration: int,
    ) -> dict[str, Any]:
        """Validate executive continuity without judging task correctness.

        Important:
        - This is NOT a hard physical validator.
        - The blocked action was originally selected by the brain itself.
        - When its reported preconditions become READY, the brain must either:
            1) continue that same intent, or
            2) explicitly abandon it with a reason.
        - The runtime never generates a replacement action.
        """
        active = self._active_view(
            world,
            iteration=iteration,
        )
        if active is None:
            return {
                "status": "NO_PENDING_INTENT",
            }

        resolution = decision.pending_intent_resolution

        if resolution == "abandon":
            reason = str(
                decision.pending_intent_abandon_reason or ""
            ).strip()
            if not reason:
                self.stats.protocol_retries += 1
                raise PendingIntentProtocolError(
                    "pending intent abandonment requires a reason",
                    feedback={
                        "rejection_type": (
                            "PENDING_INTENT_PROTOCOL"
                        ),
                        "pending_intent": active,
                        "required_resolution": (
                            "continue_or_explicit_abandon"
                        ),
                    },
                )

            abandoned = self._stack.pop()
            self.stats.frames_abandoned += 1
            return {
                "status": "PENDING_INTENT_ABANDONED",
                "abandoned_action": dict(
                    abandoned.blocked_action
                ),
                "reason": reason,
            }

        if active.get("status") != "READY":
            # While BLOCKED, the brain is still free to choose any legal
            # prerequisite/recovery action.
            return {
                "status": "PENDING_INTENT_BLOCKED",
                "pending_intent": active,
            }

        blocked_action = active.get("blocked_action")
        if (
            isinstance(blocked_action, dict)
            and _decision_contains_action(
                decision,
                blocked_action,
            )
        ):
            self.stats.ready_resume_decisions += 1
            return {
                "status": "PENDING_INTENT_RESUMED",
                "pending_intent": active,
            }

        # This is the key V5.4 behavior.  A READY intent cannot silently be
        # forgotten.  The brain can still change its mind, but must say so
        # explicitly with pending_intent_resolution='abandon'.
        self.stats.protocol_retries += 1
        raise PendingIntentProtocolError(
            "READY pending intent was ignored; continue the brain's own "
            "blocked action or explicitly abandon it",
            feedback={
                "rejection_type": (
                    "PENDING_INTENT_CONTINUITY_REQUIRED"
                ),
                "pending_intent": active,
                "required_resolution": {
                    "continue": (
                        "include the pending blocked_action in this "
                        "dispatch"
                    ),
                    "abandon": (
                        "set pending_intent_resolution='abandon' "
                        "and provide pending_intent_abandon_reason"
                    ),
                },
            },
        )

    def observe_execution(
        self,
        report: DispatchExecutionReport,
        *,
        world: WorldState,
        iteration: int,
        credited_goal_facts: list[dict[str, Any]] | None = None,
    ) -> None:
        self._expire(iteration)

        successful_signatures = {
            _action_signature(
                result.bound_action.action.as_dict()
            )
            for result in report.results
            if result.verified_success
        }

        # Only the top frame can complete directly.  If it completes, reveal
        # the parent frame underneath it for the next World iteration.
        while (
            self._stack
            and self._stack[-1].signature in successful_signatures
        ):
            self._stack.pop()
            self.stats.frames_completed += 1

        # Successful prerequisite actions intentionally do NOT clear the frame.
        # That persistence is the main purpose of this tracker.
        #
        # But do NOT over-remember stale errors. If a later successful action
        # earned task credit on an entity referenced by the blocked action,
        # that old intent may have been superseded by real task progress.
        #
        # Example:
        #   old rejected intent: pick(package_1)
        #   later progress:      package_1.location -> target_bin
        # Re-picking package_1 would now undo progress, so the old frame is
        # retired instead of being forced by continuity.
        credited_goal_facts = credited_goal_facts or []
        while self._stack and credited_goal_facts:
            frame = self._stack[-1]
            arguments = frame.blocked_action.get(
                "arguments",
                {},
            )
            referenced_entities = {
                value
                for value in (
                    arguments.values()
                    if isinstance(arguments, dict)
                    else []
                )
                if isinstance(value, str)
            }

            superseded = any(
                isinstance(row, dict)
                and row.get("subject") in referenced_entities
                for row in credited_goal_facts
            )
            if not superseded:
                break

            self._stack.pop()
            self.stats.frames_superseded_by_progress += 1

        self._expire(iteration)

    def clear(self) -> None:
        self._stack.clear()

    def brain_view(
        self,
        world: WorldState,
        *,
        iteration: int,
    ) -> dict[str, Any]:
        self._expire(iteration)

        frames: list[dict[str, Any]] = []
        for depth, frame in enumerate(self._stack):
            evaluated = [
                _difference_status(row, world)
                for row in frame.state_differences
            ]
            remaining = [
                row
                for row in evaluated
                if not row.get("now_satisfied", False)
            ]
            satisfied = [
                row
                for row in evaluated
                if row.get("now_satisfied", False)
            ]

            status = (
                "READY"
                if bool(evaluated) and not remaining
                else "BLOCKED"
            )

            frames.append({
                "depth": depth,
                "status": status,
                "blocked_action": dict(frame.blocked_action),
                "rejection_type": frame.rejection_type,
                "created_iteration": frame.created_iteration,
                "last_updated_iteration": frame.last_updated_iteration,
                "age_iterations": (
                    iteration - frame.last_updated_iteration
                ),
                "rejection_count": frame.rejection_count,
                "remaining_state_differences": remaining,
                "satisfied_state_differences": satisfied,
                "all_reported_preconditions_now_satisfied": (
                    bool(evaluated) and not remaining
                ),
            })

        return {
            "active": bool(frames),
            "stack_depth": len(frames),
            "frames": frames,
            "active_frame": (
                frames[-1] if frames else None
            ),
            "note": (
                "These are the brain's own prior rejected intentions. "
                "The runtime does not provide repair actions. A READY "
                "active intent must be continued or explicitly abandoned."
            ),
        }
