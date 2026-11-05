"""
Manual Runtime session state and UI-facing session helpers.

Session lifecycle state belongs to the Runtime interface layer rather
than services.plan_service.
"""

from __future__ import annotations

from threading import Event, RLock, Thread
from ..domain.world import WorldState
from dataclasses import dataclass, field
from datetime import datetime, timezone
from copy import deepcopy


class ManualRuntimeCancelled(RuntimeError):
    """Raised when the human cancels an active manual Runtime session."""


def _manual_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _manual_world_summary(
    world: WorldState | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compact semantic state for UI history/debugging."""

    if world is None:
        return None

    if isinstance(
        world,
        WorldState,
    ):
        snapshot = world.snapshot()
    elif isinstance(
        world,
        dict,
    ):
        snapshot = deepcopy(
            world
        )
    else:
        return None

    entities = snapshot.get(
        "entities",
        {},
    )

    result = {
        "entities": {},
        "relations": deepcopy(
            snapshot.get(
                "relations",
                [],
            )
        ),
    }

    if not isinstance(
        entities,
        dict,
    ):
        return result

    fields = (
        "type",
        "class_name",
        "name",
        "exists",
        "region_id",
        "open_state",
        "held_by",
        "location",
        "holding",
        "controlling",
        "pickable",
        "pickable_by",
        "reachable_by",
        "tags",
        "affordances",
        "source",
        "camera_source",
    )

    for entity_id, entity in entities.items():
        if not isinstance(
            entity,
            dict,
        ):
            continue

        row = {}

        for field_name in fields:
            if field_name in entity:
                row[
                    field_name
                ] = deepcopy(
                    entity[
                        field_name
                    ]
                )

        result[
            "entities"
        ][
            entity_id
        ] = row

    return result


def _manual_short_error(
    exc: Exception,
    *,
    limit: int = 600,
) -> str:
    text = str(
        exc
    ).strip()

    if not text:
        return type(
            exc
        ).__name__

    if len(
        text
    ) <= limit:
        return text

    return (
        text[
            :limit
        ]
        + " ..."
    )


@dataclass
class _ManualRuntimeSession:
    session_id: str
    user_text: str
    planning_options: dict[str, Any]
    max_iterations: int
    openai_api_key: str | None = field(
        default=None,
        repr=False,
    )
    force_device_id: str | None = None
    execution_memory_path: str | None = None

    status: str = "starting"
    stage: str = "starting"
    message: str | None = None

    instruction: str | None = None
    pending_action: dict[str, Any] | None = None
    pending_step: int = 0

    # UI-visible action proposal.  This is set as soon as Brain/Binder selects
    # an action, before PRE-COMMIT observation/validation.  It lets the human
    # distinguish "Runtime wanted to do X" from "X was actually approved".
    proposed_action: dict[str, Any] | None = None
    proposed_instruction: str | None = None
    proposed_step: int = 0
    proposed_state: str | None = None
    proposed_reason: str | None = None

    task_spec: dict[str, Any] | None = None
    latest_world: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    # Persistent debug history.  Do not overwrite earlier rows when the
    # Runtime advances to a later stage or fails.
    history: list[dict[str, Any]] = field(
        default_factory=list
    )
    steps: list[dict[str, Any]] = field(
        default_factory=list
    )
    history_seq: int = 0

    cancel_requested: bool = False

    continue_event: Event = field(
        default_factory=Event,
        repr=False,
    )

    lock: RLock = field(
        default_factory=RLock,
        repr=False,
    )

    thread: Thread | None = field(
        default=None,
        repr=False,
    )

    def update(
        self,
        **changes,
    ) -> None:
        with self.lock:
            for key, value in changes.items():
                setattr(
                    self,
                    key,
                    value,
                )

    def record(
        self,
        event: str,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self.lock:
            self.history_seq += 1

            row = {
                "seq":
                    self.history_seq,
                "at":
                    _manual_now(),
                "event":
                    event,
                "status":
                    self.status,
                "stage":
                    self.stage,
            }

            if message is not None:
                row[
                    "message"
                ] = message

            if isinstance(
                data,
                dict,
            ):
                row[
                    "data"
                ] = deepcopy(
                    data
                )

            self.history.append(
                row
            )

    def begin_step(
        self,
        *,
        action: dict[str, Any],
        world_before: WorldState,
    ) -> int:
        with self.lock:
            step_number = len(
                self.steps
            ) + 1

            self.steps.append({
                "step":
                    step_number,
                "started_at":
                    _manual_now(),
                "status":
                    "precommit_observation",
                "action":
                    deepcopy(
                        action
                    ),
                "instruction":
                    None,
                "proposal":
                    None,
                "gate_status":
                    "proposed",
                "blocked_reason":
                    None,
                "world_before":
                    _manual_world_summary(
                        world_before
                    ),
                "precommit_observation_raw":
                    None,
                "precommit_world":
                    None,
                "precommit_reconcile_changes":
                    [],
                "semantic_world_raw":
                    None,
                "semantic_world":
                    None,
                "post_observation_raw":
                    None,
                "world_after":
                    None,
                "reconcile_changes":
                    [],
                "error":
                    None,
            })

            return step_number

    def update_step(
        self,
        step_number: int,
        **changes,
    ) -> None:
        with self.lock:
            index = (
                step_number
                - 1
            )

            if (
                index < 0
                or index
                >= len(
                    self.steps
                )
            ):
                raise KeyError(
                    f"manual step not found: {step_number}"
                )

            row = self.steps[
                index
            ]

            for key, value in changes.items():
                row[
                    key
                ] = deepcopy(
                    value
                )

    def snapshot(
        self,
    ) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id":
                    self.session_id,
                "user_text":
                    self.user_text,
                "model_config":
                    deepcopy(
                        self.planning_options
                    ),
                "force_device_id":
                    self.force_device_id,
                "execution_memory_path":
                    self.execution_memory_path,
                "status":
                    self.status,
                "stage":
                    self.stage,
                "message":
                    self.message,
                "awaiting_confirmation":
                    self.status
                    == "awaiting_confirmation",
                "instruction":
                    self.instruction,
                "pending_action":
                    deepcopy(
                        self.pending_action
                    ),
                "pending_step":
                    self.pending_step,
                "proposed_action":
                    deepcopy(
                        self.proposed_action
                    ),
                "proposed_instruction":
                    self.proposed_instruction,
                "proposed_step":
                    self.proposed_step,
                "proposed_state":
                    self.proposed_state,
                "proposed_reason":
                    self.proposed_reason,
                "task_spec":
                    deepcopy(
                        self.task_spec
                    ),
                "latest_world":
                    deepcopy(
                        self.latest_world
                    ),
                "history":
                    deepcopy(
                        self.history
                    ),
                "steps":
                    deepcopy(
                        self.steps
                    ),
                "result":
                    deepcopy(
                        self.result
                    ),
                "error":
                    deepcopy(
                        self.error
                    ),
            }


__all__ = [
    "ManualRuntimeCancelled",
    "_ManualRuntimeSession",
    "_manual_now",
    "_manual_world_summary",
    "_manual_short_error",
]
