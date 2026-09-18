"""
Persistent repair-intent memory for the closed-loop runtime.

RepairContextTracker remembers actions that the Brain itself selected but that
were rejected because CURRENT physical preconditions were not satisfied.

It does NOT:
- choose a repair action
- invent candidate actions
- execute anything
- mutate WorldState
- silently convert predicted effects into truth

The tracker only preserves executive continuity across observation cycles:

    rejected intent
        ↓
    remember CURRENT-vs-REQUIRED facts
        ↓
    Brain repairs one prerequisite at a time
        ↓
    fresh observation
        ↓
    frame becomes READY
        ↓
    Brain must explicitly CONTINUE or ABANDON
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .dispatch import (
    DispatchDecision,
    DispatchExecutionReport,
)
from .world_state import WorldState


def _action_signature(
    action: dict[str, Any] | None,
) -> str:
    """
    Stable semantic action signature.

    Device IDs are intentionally not part of the signature because repair
    memory belongs to the Brain's semantic intent, before device binding.
    """

    if not isinstance(
        action,
        dict,
    ):
        return ""

    return json.dumps(
        {
            "function_name":
                action.get(
                    "function_name"
                ),
            "arguments":
                action.get(
                    "arguments",
                    {},
                ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _difference_status(
    difference: dict[str, Any],
    world: WorldState,
) -> dict[str, Any]:
    """
    Re-evaluate one previously reported CURRENT-vs-REQUIRED difference against
    the latest observed WorldState.
    """

    row = dict(
        difference
    )

    subject = row.get(
        "subject"
    )
    field = row.get(
        "field"
    )
    operator = row.get(
        "operator"
    )
    required = row.get(
        "required"
    )

    if (
        not isinstance(
            subject,
            str,
        )
        or not isinstance(
            field,
            str,
        )
    ):
        row[
            "current_now"
        ] = None

        row[
            "current_known_now"
        ] = False

        row[
            "now_satisfied"
        ] = False

        return row

    known = world.has_field(
        subject,
        field,
    )

    current = world.get(
        subject,
        field,
    )

    if operator == "ne":
        satisfied = (
            current
            != required
        )

    elif operator == "if_present_eq":
        satisfied = (
            (not known)
            or current == required
        )

    elif operator == "if_known_eq":
        satisfied = (
            (not known)
            or required is None
            or current == required
        )

    else:
        # "eq" and unknown equality-like operators conservatively use eq.
        satisfied = (
            current
            == required
        )

    row[
        "current_now"
    ] = current

    row[
        "current_known_now"
    ] = known

    row[
        "now_satisfied"
    ] = bool(
        satisfied
    )

    return row


@dataclass
class RepairFrame:
    blocked_action: dict[
        str,
        Any,
    ]

    state_differences: list[
        dict[str, Any]
    ]

    created_iteration: int

    last_updated_iteration: int

    rejection_type: str = (
        "PHYSICAL_PRECONDITION_FAILED"
    )

    rejection_count: int = 1

    @property
    def signature(
        self,
    ) -> str:
        return _action_signature(
            self.blocked_action
        )


@dataclass
class RepairContextStats:
    frames_created: int = 0
    frames_completed: int = 0
    frames_expired: int = 0
    frames_abandoned: int = 0
    ready_resume_decisions: int = 0
    protocol_retries: int = 0


class PendingIntentProtocolError(
    RuntimeError
):
    """
    Executive-memory protocol error.

    This is NOT a physical hard rejection.  The Brain's output violated the
    continuity protocol for a remembered pending intent.
    """

    def __init__(
        self,
        message: str,
        *,
        feedback: dict[
            str,
            Any,
        ] | None = None,
    ):
        super().__init__(
            message
        )

        self.feedback = (
            feedback
            or {}
        )


def _decision_contains_action(
    decision: DispatchDecision,
    action: dict[
        str,
        Any,
    ],
) -> bool:
    target = _action_signature(
        action
    )

    return any(
        _action_signature(
            candidate.as_dict()
        )
        == target
        for candidate
        in decision.actions
    )


class RepairContextTracker:
    """
    Persistent stack of the Brain's own physically blocked action intents.

    Frames form a stack because a repair action can itself become blocked.

    Example:

        place_object(...)
            blocked because object not held
                ↓
        pick_object(...)
            blocked because another object is held
                ↓
        repair top frame first
                ↓
        then reveal parent frame again

    The runtime stores facts, not repair plans.
    """

    def __init__(
        self,
        *,
        max_depth: int = 6,
        max_age_iterations: int = 12,
    ):
        if max_depth < 1:
            raise ValueError(
                "max_depth must be >= 1"
            )

        if max_age_iterations < 1:
            raise ValueError(
                "max_age_iterations must be >= 1"
            )

        self.max_depth = int(
            max_depth
        )

        self.max_age_iterations = int(
            max_age_iterations
        )

        self._stack: list[
            RepairFrame
        ] = []

        self.stats = (
            RepairContextStats()
        )

    def _expire(
        self,
        iteration: int,
    ) -> None:
        kept: list[
            RepairFrame
        ] = []

        for frame in self._stack:
            age = (
                iteration
                - frame.last_updated_iteration
            )

            if (
                age
                > self.max_age_iterations
            ):
                self.stats.frames_expired += 1
                continue

            kept.append(
                frame
            )

        self._stack = kept[
            -self.max_depth:
        ]

    def record_rejection(
        self,
        structured_feedback: dict[
            str,
            Any,
        ] | None,
        *,
        iteration: int,
    ) -> None:
        """
        Record one physical-precondition rejection.

        Other rejection classes are intentionally ignored.  Structural,
        resource, safety, and protocol failures are not semantic repair frames.
        """

        if not isinstance(
            structured_feedback,
            dict,
        ):
            return

        if (
            structured_feedback.get(
                "rejection_type"
            )
            != "PHYSICAL_PRECONDITION_FAILED"
        ):
            return

        action = (
            structured_feedback.get(
                "attempted_action"
            )
        )

        differences = (
            structured_feedback.get(
                "state_differences"
            )
        )

        if not isinstance(
            action,
            dict,
        ):
            return

        if (
            not isinstance(
                differences,
                list,
            )
            or not differences
        ):
            return

        clean_differences = [
            dict(
                row
            )
            for row
            in differences
            if isinstance(
                row,
                dict,
            )
        ]

        if not clean_differences:
            return

        self._expire(
            iteration
        )

        signature = (
            _action_signature(
                action
            )
        )

        # Same active intent rejected again: refresh the known differences.
        if (
            self._stack
            and self._stack[
                -1
            ].signature
            == signature
        ):
            frame = (
                self._stack[
                    -1
                ]
            )

            frame.state_differences = (
                clean_differences
            )

            frame.last_updated_iteration = (
                iteration
            )

            frame.rejection_count += 1

            return

        # The same intent may already exist deeper in a nested repair stack.
        # Move it to the top instead of duplicating it.
        existing_index: (
            int
            | None
        ) = None

        for (
            index,
            frame,
        ) in enumerate(
            self._stack
        ):
            if (
                frame.signature
                == signature
            ):
                existing_index = (
                    index
                )
                break

        if (
            existing_index
            is not None
        ):
            frame = (
                self._stack.pop(
                    existing_index
                )
            )

            frame.state_differences = (
                clean_differences
            )

            frame.last_updated_iteration = (
                iteration
            )

            frame.rejection_count += 1

            self._stack.append(
                frame
            )

            return

        self._stack.append(
            RepairFrame(
                blocked_action=
                    dict(
                        action
                    ),
                state_differences=
                    clean_differences,
                created_iteration=
                    iteration,
                last_updated_iteration=
                    iteration,
                rejection_type=
                    str(
                        structured_feedback.get(
                            "rejection_type",
                            "PHYSICAL_PRECONDITION_FAILED",
                        )
                    ),
            )
        )

        self.stats.frames_created += 1

        if (
            len(
                self._stack
            )
            > self.max_depth
        ):
            overflow = (
                len(
                    self._stack
                )
                - self.max_depth
            )

            self._stack = (
                self._stack[
                    -self.max_depth:
                ]
            )

            self.stats.frames_expired += (
                overflow
            )

    def _active_view(
        self,
        world: WorldState,
        *,
        iteration: int,
    ) -> dict[
        str,
        Any,
    ] | None:
        view = self.brain_view(
            world,
            iteration=iteration,
        )

        active = view.get(
            "active_frame"
        )

        return (
            active
            if isinstance(
                active,
                dict,
            )
            else None
        )

    def validate_decision_continuity(
        self,
        decision: DispatchDecision,
        *,
        world: WorldState,
        iteration: int,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Enforce pending-intent continuity.

        BLOCKED:
            Brain may choose one legal prerequisite/recovery action, or
            explicitly abandon the pending intent.

        READY:
            Brain must explicitly:
                pending_intent_resolution="continue"
            and include the exact blocked action,

            OR:

                pending_intent_resolution="abandon"
            with a non-empty reason.

        This function never chooses a replacement action.
        """

        active = self._active_view(
            world,
            iteration=iteration,
        )

        resolution = (
            decision
            .pending_intent_resolution
        )

        if active is None:
            # The pending-intent protocol only has meaning when a repair frame
            # actually exists.  Some structured-output models may still emit a
            # syntactically valid resolution field even when no frame is
            # active; treat it as irrelevant instead of burning a retry.
            return {
                "status":
                    "NO_PENDING_INTENT",
                "ignored_resolution":
                    resolution,
            }

        # ----------------------------------------------------
        # Explicit abandon is legal for BLOCKED or READY frames
        # ----------------------------------------------------

        if resolution == "abandon":
            reason = str(
                decision
                .pending_intent_abandon_reason
                or ""
            ).strip()

            if not reason:
                self.stats.protocol_retries += 1

                raise PendingIntentProtocolError(
                    "pending intent abandonment requires a reason",
                    feedback={
                        "rejection_type":
                            "PENDING_INTENT_PROTOCOL",
                        "pending_intent":
                            active,
                        "required_resolution":
                            "continue_or_explicit_abandon",
                    },
                )

            abandoned = (
                self._stack.pop()
            )

            self.stats.frames_abandoned += 1

            return {
                "status":
                    "PENDING_INTENT_ABANDONED",
                "abandoned_action":
                    dict(
                        abandoned.blocked_action
                    ),
                "reason":
                    reason,
            }

        # ----------------------------------------------------
        # Still BLOCKED
        # ----------------------------------------------------

        if (
            active.get(
                "status"
            )
            != "READY"
        ):
            if resolution == "continue":
                self.stats.protocol_retries += 1

                raise PendingIntentProtocolError(
                    "pending intent is still BLOCKED; "
                    "repair its remaining state differences before continuing",
                    feedback={
                        "rejection_type":
                            "PENDING_INTENT_NOT_READY",
                        "pending_intent":
                            active,
                        "required_resolution":
                            "repair_or_explicit_abandon",
                    },
                )

            return {
                "status":
                    "PENDING_INTENT_BLOCKED",
                "pending_intent":
                    active,
            }

        # ----------------------------------------------------
        # READY: continuation must be explicit
        # ----------------------------------------------------

        blocked_action = active.get(
            "blocked_action"
        )

        contains_blocked_action = (
            isinstance(
                blocked_action,
                dict,
            )
            and _decision_contains_action(
                decision,
                blocked_action,
            )
        )

        if (
            resolution
            == "continue"
            and contains_blocked_action
        ):
            self.stats.ready_resume_decisions += 1

            return {
                "status":
                    "PENDING_INTENT_RESUMED",
                "pending_intent":
                    active,
            }

        self.stats.protocol_retries += 1

        raise PendingIntentProtocolError(
            "READY pending intent must be explicitly continued "
            "or abandoned",
            feedback={
                "rejection_type":
                    "PENDING_INTENT_CONTINUITY_REQUIRED",
                "pending_intent":
                    active,
                "required_resolution": {
                    "continue": (
                        "set pending_intent_resolution='continue' "
                        "and include the exact blocked_action"
                    ),
                    "abandon": (
                        "set pending_intent_resolution='abandon' "
                        "and provide pending_intent_abandon_reason"
                    ),
                },
                "decision_contains_blocked_action":
                    contains_blocked_action,
                "provided_resolution":
                    resolution,
            },
        )

    def observe_execution(
        self,
        report: DispatchExecutionReport,
        *,
        world: WorldState,
        iteration: int,
        credited_goal_facts: list[
            dict[str, Any]
        ] | None = None,
    ) -> None:
        """
        Update repair stack after fresh post-execution observation.

        A frame completes only when its exact blocked action was itself
        semantically verified successful.

        Successful prerequisite actions intentionally do NOT erase the parent
        repair frame.  The next `brain_view()` re-evaluates its state
        differences against the fresh WorldState and may turn it READY.

        `world` and `credited_goal_facts` remain in the signature for current
        coordinator compatibility; they are intentionally not used to silently
        discard pending intents.  Stale intents must be explicitly abandoned
        by the Brain.
        """

        _ = (
            world,
            credited_goal_facts,
        )

        self._expire(
            iteration
        )

        successful_signatures = {
            _action_signature(
                result
                .bound_action
                .action
                .as_dict()
            )
            for result
            in report.results
            if result.verified_success
        }

        # Nested repair frames complete top-down.
        while (
            self._stack
            and self._stack[
                -1
            ].signature
            in successful_signatures
        ):
            self._stack.pop()

            self.stats.frames_completed += 1

        self._expire(
            iteration
        )

    def clear(
        self,
    ) -> None:
        self._stack.clear()

    def brain_view(
        self,
        world: WorldState,
        *,
        iteration: int,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Return the current repair-memory view for RuntimeBrain.
        """

        self._expire(
            iteration
        )

        frames: list[
            dict[str, Any]
        ] = []

        for (
            depth,
            frame,
        ) in enumerate(
            self._stack
        ):
            evaluated = [
                _difference_status(
                    row,
                    world,
                )
                for row
                in frame.state_differences
            ]

            remaining = [
                row
                for row
                in evaluated
                if not row.get(
                    "now_satisfied",
                    False,
                )
            ]

            satisfied = [
                row
                for row
                in evaluated
                if row.get(
                    "now_satisfied",
                    False,
                )
            ]

            status = (
                "READY"
                if (
                    bool(
                        evaluated
                    )
                    and not remaining
                )
                else "BLOCKED"
            )

            frames.append({
                "depth":
                    depth,

                "status":
                    status,

                "blocked_action":
                    dict(
                        frame.blocked_action
                    ),

                "rejection_type":
                    frame.rejection_type,

                "created_iteration":
                    frame.created_iteration,

                "last_updated_iteration":
                    frame.last_updated_iteration,

                "age_iterations":
                    (
                        iteration
                        - frame.last_updated_iteration
                    ),

                "rejection_count":
                    frame.rejection_count,

                "remaining_state_differences":
                    remaining,

                "satisfied_state_differences":
                    satisfied,

                "all_reported_preconditions_now_satisfied":
                    (
                        bool(
                            evaluated
                        )
                        and not remaining
                    ),
            })

        return {
            "active":
                bool(
                    frames
                ),

            "stack_depth":
                len(
                    frames
                ),

            "frames":
                frames,

            "active_frame":
                (
                    frames[
                        -1
                    ]
                    if frames
                    else None
                ),

            "note": (
                "These are the Brain's own prior rejected intentions. "
                "The runtime does not provide repair actions. "
                "A READY active intent must be explicitly continued "
                "or explicitly abandoned."
            ),
        }


__all__ = [
    "RepairFrame",
    "RepairContextStats",
    "PendingIntentProtocolError",
    "RepairContextTracker",
]
