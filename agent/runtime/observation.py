"""
Runtime observation adapter.

This module is the boundary between runtime execution and the current
world-model service.

Current service name:
    services.vlm_narrator_service

That service is expected to be renamed later.  Keep the import isolated here
so the runtime does not depend on the temporary service name everywhere.

Observation flow:

    services / perception
            |
            v
    world-model service
            |
            v
    runtime.observation
            |
            v
        WorldState

This module does NOT run YOLO, VLM, material inference, camera access, or arm
drivers directly.  Those remain service responsibilities.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .world_state import WorldState

if TYPE_CHECKING:
    from .dispatch import DispatchExecutionReport


@dataclass(frozen=True)
class ObservationConfig:
    """
    Controls what one fresh runtime observation requests.

    Defaults intentionally match the current build_world_state() service
    behavior.
    """

    include_robot: bool = True
    include_cameras: bool = True
    include_detections: bool = True
    include_materials: bool = False
    include_relations: bool = True
    camera_names: tuple[str, ...] = (
        "left",
        "right",
    )


def _get_world_state_service():
    """
    Resolve the current world-model service lazily.

    Lazy import avoids coupling package import order to the service layer and
    makes the future rename from vlm_narrator_service to world_state_service a
    one-line change here.
    """

    from services import (
        vlm_narrator_service as world_state_service,
    )

    return world_state_service


def _normalize_context(
    context: WorldState | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if context is None:
        return None

    if isinstance(
        context,
        WorldState,
    ):
        return context.snapshot()

    if isinstance(
        context,
        dict,
    ):
        return deepcopy(
            context
        )

    raise TypeError(
        "context must be WorldState, dict, or None"
    )


def capture_world(
    *,
    config: ObservationConfig | None = None,
    context: WorldState | dict[str, Any] | None = None,
) -> WorldState:
    """
    Request one fresh canonical WorldState from the world-model service.

    `context` is optional carry-in semantic context.  The service decides how
    observed fields override that context.
    """

    if config is None:
        config = ObservationConfig()

    if not isinstance(
        config,
        ObservationConfig,
    ):
        raise TypeError(
            "config must be ObservationConfig"
        )

    service = (
        _get_world_state_service()
    )

    result = service.build_world_state(
        context=_normalize_context(
            context
        ),
        include_robot=
            config.include_robot,
        include_cameras=
            config.include_cameras,
        include_detections=
            config.include_detections,
        include_materials=
            config.include_materials,
        include_relations=
            config.include_relations,
        camera_names=
            config.camera_names,
    )

    if isinstance(
        result,
        WorldState,
    ):
        return result

    # Compatibility for a future service implementation that may return the
    # canonical state as a plain dict instead of constructing WorldState.
    if isinstance(
        result,
        dict,
    ):
        return WorldState(
            result
        )

    raise RuntimeError(
        "world-state service returned unsupported type: "
        f"{type(result).__name__}"
    )


def observe_world(
    previous_world: WorldState | None = None,
    report: DispatchExecutionReport | None = None,
    *,
    config: ObservationConfig | None = None,
    context: WorldState | dict[str, Any] | None = None,
) -> WorldState:
    """
    Runtime callback-compatible fresh observation.

    The `(previous_world, report)` positional shape is kept so this function
    can be passed to the executor/coordinator observation callback.

    If explicit `context` is not provided, `previous_world` is used as carry-in
    context.  Fresh observations produced by the world-model service still
    override fields according to that service's merge policy.

    `report` is intentionally not treated as world truth.  Command success
    alone must not mutate semantic state.
    """

    if (
        previous_world is not None
        and not isinstance(
            previous_world,
            WorldState,
        )
    ):
        raise TypeError(
            "previous_world must be WorldState or None"
        )

    # Kept for callback compatibility and future verification-aware adapters.
    _ = report

    effective_context = (
        context
        if context is not None
        else previous_world
    )

    return capture_world(
        config=config,
        context=effective_context,
    )


def make_observer(
    *,
    config: ObservationConfig | None = None,
    context: WorldState | dict[str, Any] | None = None,
):
    """
    Build a callback suitable for runtime coordinator/executor wiring.
    """

    def _observer(
        previous_world: WorldState | None = None,
        report: DispatchExecutionReport | None = None,
    ) -> WorldState:
        return observe_world(
            previous_world,
            report,
            config=config,
            context=context,
        )

    return _observer


__all__ = [
    "ObservationConfig",
    "capture_world",
    "observe_world",
    "make_observer",
]
