"""
Runtime observation adapter.
This module owns fresh runtime observation and generic action-coupled
observation helpers.
It intentionally does NOT contain task-specific concepts such as
"trash can", "chair", or fixed joint poses.  Concrete region movement
and camera behavior are injected through callbacks.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING
import requests
from .world_state import WorldState
if TYPE_CHECKING:
    from .dispatch import (
        BoundAction,
        BoundDispatch,
        DispatchExecutionReport,
    )
_WORLD_STATE_URL = (
    "http://127.0.0.1:5001"
    "/api/worldmodel/world_state"
)
_WORLD_STATE_TIMEOUT_S = 60.0
# Conventional semantic argument names used only as a fallback resolver.
# Callers can inject their own target resolver for new tool schemas.
_DEFAULT_TARGET_ARGUMENT_PRIORITY = (
    "destination_id",
    "container_id",
    "object_id",
    "target_id",
    "entity_id",
)
@dataclass(frozen=True)
class ObservationConfig:
    """
    Controls what one fresh runtime observation requests from the
    Flask World Model API.
    """
    include_robot: bool = True
    include_cameras: bool = True
    include_detections: bool = True
    include_materials: bool = False
    include_relations: bool = True
    include_grasp: bool = True
    camera_names: tuple[str, ...] = (
        "left",
        "right",
    )
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
def _request_world_state(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Call the Flask World Model endpoint and return the canonical snapshot.
    """
    try:
        response = requests.post(
            _WORLD_STATE_URL,
            json=payload,
            timeout=_WORLD_STATE_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "world-state API request failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "world-state API returned non-JSON response: "
            f"HTTP {response.status_code}"
        ) from exc
    if not isinstance(
        body,
        dict,
    ):
        raise RuntimeError(
            "world-state API returned invalid JSON payload"
        )
    if (
        response.status_code != 200
        or body.get(
            "result"
        )
        is not True
    ):
        message = body.get(
            "message"
        )
        raise RuntimeError(
            "world-state API failed: "
            f"HTTP {response.status_code}, "
            f"message={message!r}"
        )
    data = body.get(
        "data"
    )
    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "world-state API response data must be an object"
        )
    return data
def capture_world(
    *,
    config: ObservationConfig | None = None,
    context: WorldState | dict[str, Any] | None = None,
    active_region_id: str | None = None,
) -> WorldState:
    """
    Request one fresh canonical WorldState through the Flask API.
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
    if active_region_id is not None:
        if (
            not isinstance(
                active_region_id,
                str,
            )
            or not active_region_id.strip()
        ):
            raise ValueError(
                "active_region_id must be a non-empty string or None"
            )
        active_region_id = (
            active_region_id.strip()
        )
    payload = {
        "active_region_id":
            active_region_id,
        "context":
            _normalize_context(
                context
            ),
        "include_robot":
            config.include_robot,
        "include_cameras":
            config.include_cameras,
        "include_detections":
            config.include_detections,
        "include_materials":
            config.include_materials,
        "include_relations":
            config.include_relations,
        "include_grasp":
            config.include_grasp,
        "camera_names":
            list(
                config.camera_names
            ),
    }
    snapshot = (
        _request_world_state(
            payload
        )
    )
    return WorldState(
        snapshot
    )
def observe_world(
    previous_world: WorldState | None = None,
    report: DispatchExecutionReport | None = None,
    *,
    config: ObservationConfig | None = None,
    context: WorldState | dict[str, Any] | None = None,
    active_region_id: str | None = None,
) -> WorldState:
    """
    Runtime callback-compatible fresh observation.
    `report` is intentionally not treated as world truth.
    If explicit context is not provided, the previous WorldState is sent as
    carry-in semantic context. Fresh observations remain authoritative at the
    World Model boundary.
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
    _ = report
    effective_context = (
        context
        if context is not None
        else previous_world
    )
    return capture_world(
        config=config,
        context=effective_context,
        active_region_id=
            active_region_id,
    )
def make_observer(
    *,
    config: ObservationConfig | None = None,
    context: WorldState | dict[str, Any] | None = None,
    active_region_id: str | None = None,
):
    """
    Build a callback suitable for CallbackExecutor / RuntimeCoordinator.
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
            active_region_id=
                active_region_id,
        )
    return _observer
# ============================================================
# Generic action-coupled observation helpers
# ============================================================
def _bound_action_arguments(
    bound: BoundAction,
) -> dict[str, Any]:
    action = getattr(
        bound,
        "action",
        None,
    )
    arguments = getattr(
        action,
        "arguments",
        None,
    )
    if not isinstance(
        arguments,
        dict,
    ):
        return {}
    return arguments
def default_action_target_entity(
    bound: BoundAction,
) -> str | None:
    """
    Resolve the semantic entity that should normally be observed before an
    action commits.
    This is only a conventional fallback based on generic runtime argument
    names. New/custom tools can inject a target resolver instead of modifying
    this function.
    """
    arguments = (
        _bound_action_arguments(
            bound
        )
    )
    for field in (
        _DEFAULT_TARGET_ARGUMENT_PRIORITY
    ):
        value = arguments.get(
            field
        )
        if (
            isinstance(
                value,
                str,
            )
            and value
        ):
            return value
    return None
def resolve_action_region(
    bound: BoundAction,
    world: WorldState,
    *,
    target_resolver: Callable[
        [BoundAction],
        str | None,
    ] | None = None,
    region_field: str = "region_id",
) -> str | None:
    """
    Resolve one action to a semantic observation region.
    Resolution order:
      1. explicit action argument `region_id`
      2. target entity -> WorldState[target][region_field]
    No physical pose/camera mapping lives here.
    """
    if not isinstance(
        world,
        WorldState,
    ):
        raise TypeError(
            "world must be WorldState"
        )
    arguments = (
        _bound_action_arguments(
            bound
        )
    )
    direct_region = arguments.get(
        "region_id"
    )
    if (
        isinstance(
            direct_region,
            str,
        )
        and direct_region
    ):
        return direct_region
    resolver = (
        target_resolver
        or default_action_target_entity
    )
    entity_id = resolver(
        bound
    )
    if (
        not isinstance(
            entity_id,
            str,
        )
        or not entity_id
        or not world.has_entity(
            entity_id
        )
    ):
        return None
    region_id = world.get(
        entity_id,
        region_field,
    )
    if (
        isinstance(
            region_id,
            str,
        )
        and region_id
    ):
        return region_id
    return None
def resolve_dispatch_regions(
    dispatch: BoundDispatch,
    world: WorldState,
    *,
    target_resolver: Callable[
        [BoundAction],
        str | None,
    ] | None = None,
    region_field: str = "region_id",
) -> tuple[str, ...]:
    """
    Resolve all unique semantic regions required by a dispatch.
    Order follows dispatch action order.
    """
    actions = getattr(
        dispatch,
        "actions",
        None,
    )
    if not isinstance(
        actions,
        list,
    ):
        try:
            actions = list(
                actions
            )
        except (
            TypeError,
            AttributeError,
        ):
            raise TypeError(
                "dispatch must expose iterable actions"
            )
    regions: list[str] = []
    for bound in actions:
        region_id = (
            resolve_action_region(
                bound,
                world,
                target_resolver=
                    target_resolver,
                region_field=
                    region_field,
            )
        )
        if (
            region_id is not None
            and region_id
            not in regions
        ):
            regions.append(
                region_id
            )
    return tuple(
        regions
    )
def make_dispatch_observer(
    observe_region: Callable[
        [
            str,
            WorldState,
            DispatchExecutionReport | None,
        ],
        WorldState,
    ],
    *,
    fallback_observer: Callable[
        [
            WorldState | None,
            DispatchExecutionReport | None,
        ],
        WorldState,
    ] | None = None,
    target_resolver: Callable[
        [BoundAction],
        str | None,
    ] | None = None,
    region_field: str = "region_id",
):
    """
    Build a generic action-coupled observer.
    `observe_region` is environment/site-specific. It may:
      - move a wrist camera / robot to the configured observation pose
      - choose the relevant camera
      - call the World Model
      - return a canonical WorldState
    This runtime module only resolves semantic region IDs. It does not know
    any joint values, camera poses, trash-can IDs, chair IDs, etc.
    The returned callback has signature:
        observer(dispatch, previous_world, report=None) -> WorldState
    For a multi-action dispatch, unique regions are observed in action order.
    Each fresh WorldState is carried into the next region observation.
    """
    if not callable(
        observe_region
    ):
        raise TypeError(
            "observe_region must be callable"
        )
    def _observer(
        dispatch: BoundDispatch,
        previous_world: WorldState,
        report: DispatchExecutionReport | None = None,
    ) -> WorldState:
        if not isinstance(
            previous_world,
            WorldState,
        ):
            raise TypeError(
                "previous_world must be WorldState"
            )
        regions = (
            resolve_dispatch_regions(
                dispatch,
                previous_world,
                target_resolver=
                    target_resolver,
                region_field=
                    region_field,
            )
        )
        if not regions:
            if fallback_observer is not None:
                observed = fallback_observer(
                    previous_world.clone(),
                    report,
                )
            else:
                observed = observe_world(
                    previous_world.clone(),
                    report,
                )
            if not isinstance(
                observed,
                WorldState,
            ):
                raise TypeError(
                    "fallback observer must return WorldState"
                )
            return observed
        current = (
            previous_world.clone()
        )
        for region_id in regions:
            observed = observe_region(
                region_id,
                current.clone(),
                report,
            )
            if not isinstance(
                observed,
                WorldState,
            ):
                raise TypeError(
                    "observe_region must return WorldState"
                )
            current = observed
        return current
    return _observer
__all__ = [
    "ObservationConfig",
    "capture_world",
    "observe_world",
    "make_observer",
    "default_action_target_entity",
    "resolve_action_region",
    "resolve_dispatch_regions",
    "make_dispatch_observer",
]
