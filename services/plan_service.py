from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock, Thread
from time import sleep
from typing import Any
from uuid import uuid4
from agent.config.scene_config import (
    bootstrap_scene_region_ids,
    get_scene_region,
)
from agent.runtime.observation import (
    ObservationConfig,
    capture_world,
    make_dispatch_observer,
)
from agent.runtime.brain import RuntimeBrain
from agent.runtime.coordinator import RuntimeCoordinator
from agent.runtime.execution_memory import ExecutionMemory
from agent.runtime.device_binding import StaticWorldDeviceBinder
from agent.runtime.coordination import (
    CoordinationSpec,
    DeviceRule,
)
from agent.runtime.dispatch import (
    ActionExecutionResult,
    DispatchExecutionReport,
)
from agent.runtime.executor import FakeDispatchExecutor
from agent.runtime.tool_catalogs import runtime_tools
from agent.runtime.validator import validate_dispatch
from agent.runtime.task_interpreter import (
    TaskSpec,
    interpret_task,
)
from agent.runtime.world_state import WorldState
from services import arm_service, worldmodel_service
from utils import response
MODULE = "plan"
_MANUAL_RUNTIME_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "runtime_data"
    / "manual"
)
_RUNTIME_ALLOWED_PROVIDERS = {
    "local",
    "remote",
    "openai",
}
# ============================================================
# Planning options
# ============================================================
def _normalize_stage_options(
    value: Any,
    *,
    default_provider: str,
    default_model: str | None,
    field_name: str,
    allowed_providers: set[str],
) -> dict[str, Any]:
    if value is None:
        return {
            "provider":
                default_provider,
            "model":
                default_model,
        }
    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{field_name} 必須是 object"
        )
    unknown = set(
        value
    ) - {
        "provider",
        "model",
    }
    if unknown:
        raise ValueError(
            f"{field_name} 包含未知欄位："
            f"{sorted(unknown)}"
        )
    provider = value.get(
        "provider",
        default_provider,
    )
    model = value.get(
        "model",
        default_model,
    )
    if provider not in allowed_providers:
        raise ValueError(
            f"{field_name}.provider 只支援 "
            f"{'/'.join(sorted(allowed_providers))}"
        )
    if model is not None:
        if (
            not isinstance(
                model,
                str,
            )
            or not model.strip()
        ):
            raise ValueError(
                f"{field_name}.model 必須是非空字串"
            )
        model = (
            model.strip()
        )
    return {
        "provider":
            provider,
        "model":
            model,
    }
def _normalize_planning_options(
    options: Any,
    *,
    allowed_providers: set[str],
    allow_openai_api_key: bool,
) -> dict[str, Any]:
    """
    Normalize provider/model options for a two-stage semantic pipeline.
    """
    if options is None:
        return {
            "stage1": {
                "provider": "local",
                "model": None,
            },
            "stage2": {
                "provider": "local",
                "model": None,
            },
        }
    if not isinstance(
        options,
        dict,
    ):
        raise ValueError(
            "planning_options 必須是 object"
        )
    allowed_fields = {
        "provider",
        "model",
        "stage1",
        "stage2",
    }
    if allow_openai_api_key:
        allowed_fields.add(
            "openai_api_key"
        )
    unknown = (
        set(options)
        - allowed_fields
    )
    if unknown:
        raise ValueError(
            "planning_options 包含未知欄位："
            f"{sorted(unknown)}"
        )
    has_simple = (
        "provider" in options
        or "model" in options
    )
    has_stages = (
        "stage1" in options
        or "stage2" in options
    )
    if (
        has_simple
        and has_stages
    ):
        raise ValueError(
            "planning_options 不可同時使用 "
            "provider/model 與 stage1/stage2"
        )
    if has_simple:
        provider = options.get(
            "provider",
            "local",
        )
        model = options.get(
            "model"
        )
        if (
            provider
            not in allowed_providers
        ):
            raise ValueError(
                "planning_options.provider 只支援 "
                f"{'/'.join(sorted(allowed_providers))}"
            )
        if model is not None:
            if (
                not isinstance(
                    model,
                    str,
                )
                or not model.strip()
            ):
                raise ValueError(
                    "planning_options.model "
                    "必須是非空字串"
                )
            model = model.strip()
        stage1 = {
            "provider": provider,
            "model": model,
        }
        stage2 = deepcopy(
            stage1
        )
    else:
        stage1 = (
            _normalize_stage_options(
                options.get(
                    "stage1"
                ),
                default_provider=
                    "local",
                default_model=
                    None,
                field_name=
                    "planning_options.stage1",
                allowed_providers=
                    allowed_providers,
            )
        )
        stage2 = (
            _normalize_stage_options(
                options.get(
                    "stage2"
                ),
                default_provider=
                    "local",
                default_model=
                    None,
                field_name=
                    "planning_options.stage2",
                allowed_providers=
                    allowed_providers,
            )
        )
    normalized: dict[str, Any] = {
        "stage1": stage1,
        "stage2": stage2,
    }
    if allow_openai_api_key:
        key = options.get(
            "openai_api_key"
        )
        if key is not None:
            if (
                not isinstance(
                    key,
                    str,
                )
                or not key.strip()
            ):
                raise ValueError(
                    "planning_options.openai_api_key "
                    "必須是非空字串"
                )
            normalized[
                "openai_api_key"
            ] = key.strip()
    return normalized
def normalize_runtime_planning_options(
    options: Any,
) -> dict[str, Any]:
    """
    Normalize model routing for closed-loop Runtime.

    Canonical fields:
        goal_decomposition
        coordination_extraction
        step_planning

    Backward compatibility:
        stage1 -> goal_decomposition
        stage2 -> coordination_extraction + step_planning

    Simple provider/model applies to all three phases.
    """
    if options is None:
        return {
            "goal_decomposition": {
                "provider": "local",
                "model": None,
            },
            "coordination_extraction": {
                "provider": "local",
                "model": None,
            },
            "step_planning": {
                "provider": "local",
                "model": None,
            },
        }

    if not isinstance(options, dict):
        raise ValueError(
            "planning_options 必須是 object"
        )

    allowed_fields = {
        "provider",
        "model",
        "stage1",
        "stage2",
        "goal_decomposition",
        "coordination_extraction",
        "step_planning",
    }

    unknown = set(options) - allowed_fields
    if unknown:
        raise ValueError(
            "planning_options 包含未知欄位："
            f"{sorted(unknown)}"
        )

    has_simple = (
        "provider" in options
        or "model" in options
    )
    has_legacy = (
        "stage1" in options
        or "stage2" in options
    )
    has_named = any(
        key in options
        for key in (
            "goal_decomposition",
            "coordination_extraction",
            "step_planning",
        )
    )

    if sum(bool(v) for v in (
        has_simple,
        has_legacy,
        has_named,
    )) > 1:
        raise ValueError(
            "planning_options 不可混用 provider/model、"
            "stage1/stage2 與具名 phase 欄位"
        )

    if has_simple:
        row = _normalize_stage_options(
            {
                "provider": options.get(
                    "provider",
                    "local",
                ),
                "model": options.get(
                    "model"
                ),
            },
            default_provider="local",
            default_model=None,
            field_name="planning_options",
            allowed_providers=
                _RUNTIME_ALLOWED_PROVIDERS,
        )
        return {
            "goal_decomposition": deepcopy(row),
            "coordination_extraction": deepcopy(row),
            "step_planning": deepcopy(row),
        }

    if has_legacy:
        stage1 = _normalize_stage_options(
            options.get("stage1"),
            default_provider="local",
            default_model=None,
            field_name="planning_options.stage1",
            allowed_providers=
                _RUNTIME_ALLOWED_PROVIDERS,
        )
        stage2 = _normalize_stage_options(
            options.get("stage2"),
            default_provider="local",
            default_model=None,
            field_name="planning_options.stage2",
            allowed_providers=
                _RUNTIME_ALLOWED_PROVIDERS,
        )
        return {
            "goal_decomposition": stage1,
            "coordination_extraction": stage2,
            "step_planning": deepcopy(stage2),
        }

    return {
        "goal_decomposition":
            _normalize_stage_options(
                options.get(
                    "goal_decomposition"
                ),
                default_provider="local",
                default_model=None,
                field_name=(
                    "planning_options."
                    "goal_decomposition"
                ),
                allowed_providers=
                    _RUNTIME_ALLOWED_PROVIDERS,
            ),
        "coordination_extraction":
            _normalize_stage_options(
                options.get(
                    "coordination_extraction"
                ),
                default_provider="local",
                default_model=None,
                field_name=(
                    "planning_options."
                    "coordination_extraction"
                ),
                allowed_providers=
                    _RUNTIME_ALLOWED_PROVIDERS,
            ),
        "step_planning":
            _normalize_stage_options(
                options.get(
                    "step_planning"
                ),
                default_provider="local",
                default_model=None,
                field_name=(
                    "planning_options.step_planning"
                ),
                allowed_providers=
                    _RUNTIME_ALLOWED_PROVIDERS,
            ),
    }

# ============================================================
# Runtime region observation
# ============================================================

def _runtime_arm_name(
    device_id: str,
) -> str:
    if (
        not isinstance(
            device_id,
            str,
        )
        or not device_id.strip()
    ):
        raise ValueError(
            "scene region device_id must be a non-empty string"
        )

    value = device_id.strip()

    if value.endswith(
        "_arm"
    ):
        value = value[
            :-len("_arm")
        ]

    if not value:
        raise ValueError(
            "scene region device_id does not contain a valid arm name"
        )

    return value


def _move_runtime_observer_to_region(
    region_id: str,
) -> dict[str, Any]:
    """
    Resolve a semantic region through agent.config.scene_config and move the
    configured observation arm to that region's observation pose.
    """

    region = get_scene_region(
        region_id
    )

    device_id = region.get(
        "device_id"
    )

    joints = region.get(
        "joints"
    )

    if not isinstance(
        joints,
        list,
    ) or not joints:
        raise RuntimeError(
            f"scene region {region_id!r} has no valid joints"
        )

    arm_name = _runtime_arm_name(
        device_id
    )

    result = (
        arm_service.move_arm_joints(
            arm_name=
                arm_name,

            joints=
                joints,

            speed=
                region.get(
                    "speed"
                ),

            acceleration=
                region.get(
                    "acceleration"
                ),

            wait=True,
        )
    )

    if (
        not isinstance(
            result,
            dict,
        )
        or result.get(
            "result"
        )
        is not True
    ):
        raise RuntimeError(
            "failed to move runtime observer: "
            f"region={region_id!r}, "
            f"response={result!r}"
        )

    settle_seconds = region.get(
        "settle_seconds",
        0.6,
    )

    if not isinstance(
        settle_seconds,
        (
            int,
            float,
        ),
    ):
        raise ValueError(
            f"scene region {region_id!r} "
            "settle_seconds must be numeric"
        )

    if settle_seconds < 0:
        raise ValueError(
            f"scene region {region_id!r} "
            "settle_seconds must be >= 0"
        )

    if settle_seconds > 0:
        sleep(
            float(
                settle_seconds
            )
        )

    return region


def _runtime_observation_config(
    region: dict[str, Any],
) -> ObservationConfig:
    camera_names = region.get(
        "camera_names"
    )

    if (
        not isinstance(
            camera_names,
            (
                list,
                tuple,
            ),
        )
        or not camera_names
        or not all(
            isinstance(
                item,
                str,
            )
            and item
            for item
            in camera_names
        )
    ):
        raise RuntimeError(
            "scene region has no valid camera_names"
        )

    return ObservationConfig(
        include_robot=True,
        include_cameras=False,
        include_detections=True,
        include_materials=False,
        include_relations=True,
        include_grasp=True,
        camera_names=tuple(
            camera_names
        ),
    )


def _scene_from_describe_response(
    result: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "describe_scene returned invalid response"
        )

    if (
        "result" in result
        and result.get(
            "result"
        )
        is not True
    ):
        raise RuntimeError(
            "describe_scene failed: "
            f"{result!r}"
        )

    data = result.get(
        "data"
    )

    if isinstance(
        data,
        dict,
    ):
        scene = data.get(
            "scene"
        )

        if isinstance(
            scene,
            dict,
        ):
            return scene

    scene = result.get(
        "scene"
    )

    if isinstance(
        scene,
        dict,
    ):
        return scene

    raise RuntimeError(
        "describe_scene returned no scene object"
    )


def _openable_region_entities(
    world: WorldState,
    *,
    region_id: str,
) -> list[
    tuple[
        str,
        dict[str, Any],
    ]
]:
    candidates = []

    snapshot = world.snapshot()

    entities = snapshot.get(
        "entities",
        {},
    )

    if not isinstance(
        entities,
        dict,
    ):
        return candidates

    for (
        entity_id,
        entity,
    ) in entities.items():
        if not isinstance(
            entity,
            dict,
        ):
            continue

        if entity.get(
            "region_id"
        ) != region_id:
            continue

        tags = entity.get(
            "tags"
        )

        is_openable = (
            entity.get(
                "open_state"
            )
            in {
                "open",
                "closed",
            }
            or (
                isinstance(
                    tags,
                    list,
                )
                and "openable"
                in tags
            )
        )

        if not is_openable:
            continue

        candidates.append(
            (
                entity_id,
                entity,
            )
        )

    return candidates


def _openable_scene_elements(
    scene: dict[str, Any],
) -> list[dict[str, Any]]:
    elements = scene.get(
        "scene_elements",
        [],
    )

    if not isinstance(
        elements,
        list,
    ):
        return []

    return [
        deepcopy(
            element
        )
        for element
        in elements
        if (
            isinstance(
                element,
                dict,
            )
            and element.get(
                "open_state"
            )
            in {
                "open",
                "closed",
            }
        )
    ]


def _select_scene_element_for_entity(
    *,
    entity: dict[str, Any],
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    if not elements:
        raise RuntimeError(
            "describe_scene found no openable scene element"
        )

    known_name = str(
        entity.get(
            "name"
        )
        or ""
    ).strip()

    if known_name:
        matching = []

        for element in elements:
            name = str(
                element.get(
                    "name"
                )
                or ""
            ).strip()

            if (
                name
                and (
                    known_name in name
                    or name in known_name
                )
            ):
                matching.append(
                    element
                )

        if len(
            matching
        ) == 1:
            return matching[0]

    if len(
        elements
    ) == 1:
        return elements[0]

    raise RuntimeError(
        "describe_scene openable element is ambiguous; "
        f"known_name={known_name!r}, "
        f"visible={[item.get('name') for item in elements]!r}"
    )


def _merge_describe_scene_region(
    previous_world: WorldState,
    *,
    region_id: str,
    camera_name: str,
    scene: dict[str, Any],
) -> WorldState:
    """
    Update an already-known openable entity in one semantic region.

    This is intentionally generic: no trash-can/chair/entity ID is hardcoded.
    Initial discovery/bootstrap of scene-only entities is handled separately.
    """

    candidates = _openable_region_entities(
        previous_world,
        region_id=
            region_id,
    )

    if not candidates:
        raise RuntimeError(
            "describe_scene region has no known openable entity "
            f"in WorldState: region={region_id!r}; "
            "initial runtime bootstrap must seed it first"
        )

    if len(
        candidates
    ) != 1:
        raise RuntimeError(
            "describe_scene region has multiple openable entities; "
            "a target-aware scene matcher is required: "
            f"region={region_id!r}, "
            f"entities={[item[0] for item in candidates]!r}"
        )

    entity_id, previous_entity = (
        candidates[0]
    )

    elements = (
        _openable_scene_elements(
            scene
        )
    )

    element = (
        _select_scene_element_for_entity(
            entity=
                previous_entity,

            elements=
                elements,
        )
    )

    snapshot = (
        previous_world.snapshot()
    )

    entities = snapshot.setdefault(
        "entities",
        {},
    )

    entity = deepcopy(
        previous_entity
    )

    entity[
        "open_state"
    ] = element.get(
        "open_state"
    )

    entity[
        "region_id"
    ] = region_id

    entity[
        "camera_source"
    ] = camera_name

    entity[
        "source"
    ] = "describe_scene"

    if element.get(
        "name"
    ) is not None:
        entity[
            "name"
        ] = element.get(
            "name"
        )

    for (
        source_field,
        target_field,
    ) in (
        (
            "attributes",
            "attributes",
        ),
        (
            "count",
            "count",
        ),
        (
            "location",
            "scene_location",
        ),
    ):
        if source_field in element:
            entity[
                target_field
            ] = deepcopy(
                element[
                    source_field
                ]
            )

    if "scene_summary" in scene:
        entity[
            "scene_summary"
        ] = deepcopy(
            scene.get(
                "scene_summary"
            )
        )

    entities[
        entity_id
    ] = entity

    return WorldState(
        snapshot
    )


def _observe_runtime_region(
    region_id: str,
    previous_world: WorldState,
    report=None,
) -> WorldState:
    """
    Production region observer used by make_dispatch_observer().

    Flow:
        region_id
        -> scene_config
        -> move observation arm
        -> configured camera/backend
        -> fresh WorldState
    """

    _ = report

    if not isinstance(
        previous_world,
        WorldState,
    ):
        raise TypeError(
            "previous_world must be WorldState"
        )

    region = (
        _move_runtime_observer_to_region(
            region_id
        )
    )

    observation_config = (
        _runtime_observation_config(
            region
        )
    )

    observer = str(
        region.get(
            "observer"
        )
        or ""
    ).strip()

    if observer in {
        "world_state",
        "capture_world",
    }:
        return capture_world(
            config=
                observation_config,

            context=
                previous_world,

            active_region_id=
                region_id,
        )

    if observer == "describe_scene":
        camera_name = (
            observation_config
            .camera_names[
                0
            ]
        )

        result = (
            worldmodel_service
            .describe_scene(
                camera_name=
                    camera_name,

                vlm_backend=
                    region.get(
                        "vlm_backend",
                        "remote",
                    ),
            )
        )

        scene = (
            _scene_from_describe_response(
                result
            )
        )

        return (
            _merge_describe_scene_region(
                previous_world,

                region_id=
                    region_id,

                camera_name=
                    camera_name,

                scene=
                    scene,
            )
        )

    raise RuntimeError(
        "unsupported scene region observer: "
        f"region={region_id!r}, "
        f"observer={observer!r}"
    )


def build_runtime_dispatch_observer():
    """
    Build the production action-coupled observer.

    The returned callback has the signature expected by
    agent.runtime.observation.make_dispatch_observer().
    """

    return make_dispatch_observer(
        observe_region=
            _observe_runtime_region,
    )



# ============================================================
# Runtime initial multi-region bootstrap
# ============================================================

def _merge_unique_strings(
    old_value,
    new_value,
) -> list[str]:
    result = []

    for values in (
        old_value,
        new_value,
    ):
        if not isinstance(
            values,
            (
                list,
                tuple,
                set,
            ),
        ):
            continue

        for value in values:
            if (
                isinstance(
                    value,
                    str,
                )
                and value
                and value not in result
            ):
                result.append(
                    value
                )

    return result


def _entity_matches_rule(
    entity: dict[str, Any],
    match: dict[str, Any],
) -> bool:
    if not match:
        return False

    for (
        field,
        expected,
    ) in match.items():
        if entity.get(
            field
        ) != expected:
            return False

    return True


def _apply_runtime_region_entity_rules(
    world: WorldState,
    *,
    region_id: str,
    region: dict[str, Any],
) -> WorldState:
    rules = region.get(
        "entity_rules"
    )

    if not isinstance(
        rules,
        list,
    ) or not rules:
        return world

    snapshot = world.snapshot()

    entities = snapshot.get(
        "entities",
        {},
    )

    if not isinstance(
        entities,
        dict,
    ):
        return world

    for entity in entities.values():
        if not isinstance(
            entity,
            dict,
        ):
            continue

        if entity.get(
            "region_id"
        ) != region_id:
            continue

        for rule in rules:
            if not isinstance(
                rule,
                dict,
            ):
                continue

            match = rule.get(
                "match"
            )

            if not isinstance(
                match,
                dict,
            ):
                continue

            if not _entity_matches_rule(
                entity,
                match,
            ):
                continue

            tags = _merge_unique_strings(
                entity.get(
                    "tags"
                ),
                rule.get(
                    "tags"
                ),
            )

            if tags:
                entity[
                    "tags"
                ] = tags

            affordances = (
                _merge_unique_strings(
                    entity.get(
                        "affordances"
                    ),
                    rule.get(
                        "affordances"
                    ),
                )
            )

            if affordances:
                entity[
                    "affordances"
                ] = affordances

    return WorldState(
        snapshot
    )


def _normalize_name_match(
    value,
) -> tuple[str, ...]:
    if isinstance(
        value,
        str,
    ):
        values = (
            value,
        )

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        values = value

    else:
        values = ()

    result = []

    for item in values:
        if not isinstance(
            item,
            str,
        ):
            continue

        item = item.strip()

        if (
            item
            and item not in result
        ):
            result.append(
                item
            )

    return tuple(
        result
    )


def _select_bootstrap_scene_element(
    scene: dict[str, Any],
    *,
    entity_config: dict[str, Any],
) -> dict[str, Any]:
    elements = (
        _openable_scene_elements(
            scene
        )
    )

    if not elements:
        raise RuntimeError(
            "bootstrap describe_scene found "
            "no openable scene element"
        )

    terms = (
        _normalize_name_match(
            entity_config.get(
                "name_match"
            )
        )
    )

    if terms:
        matches = []

        for element in elements:
            name = str(
                element.get(
                    "name"
                )
                or ""
            )

            if any(
                term in name
                for term in terms
            ):
                matches.append(
                    element
                )

        if len(
            matches
        ) == 1:
            return matches[0]

        if len(
            matches
        ) > 1:
            raise RuntimeError(
                "bootstrap scene element "
                "match is ambiguous: "
                f"terms={terms!r}, "
                f"visible={[item.get('name') for item in matches]!r}"
            )

    if len(
        elements
    ) == 1:
        return elements[0]

    raise RuntimeError(
        "bootstrap scene element not found: "
        f"terms={terms!r}, "
        f"visible={[item.get('name') for item in elements]!r}"
    )


def _bootstrap_describe_scene_region(
    previous_world: WorldState,
    *,
    region_id: str,
    region: dict[str, Any],
    camera_name: str,
    scene: dict[str, Any],
) -> WorldState:
    entity_configs = region.get(
        "bootstrap_entities"
    )

    if not isinstance(
        entity_configs,
        list,
    ) or not entity_configs:
        raise RuntimeError(
            "describe_scene bootstrap region "
            f"{region_id!r} has no bootstrap_entities"
        )

    snapshot = (
        previous_world.snapshot()
    )

    entities = snapshot.setdefault(
        "entities",
        {},
    )

    for entity_config in entity_configs:
        if not isinstance(
            entity_config,
            dict,
        ):
            continue

        entity_id = str(
            entity_config.get(
                "entity_id"
            )
            or ""
        ).strip()

        if not entity_id:
            raise RuntimeError(
                "bootstrap entity has no entity_id: "
                f"region={region_id!r}"
            )

        element = (
            _select_bootstrap_scene_element(
                scene,
                entity_config=
                    entity_config,
            )
        )

        entity = {
            "type":
                entity_config.get(
                    "type",
                    "object",
                ),

            "class_name":
                entity_config.get(
                    "class_name",
                    "object",
                ),

            "name":
                element.get(
                    "name"
                ),

            "exists":
                True,

            "open_state":
                element.get(
                    "open_state"
                ),

            "camera_source":
                camera_name,

            "source":
                "describe_scene",

            "region_id":
                region_id,

            "tags":
                _merge_unique_strings(
                    [],
                    entity_config.get(
                        "tags"
                    ),
                ),
        }

        affordances = (
            _merge_unique_strings(
                [],
                entity_config.get(
                    "affordances"
                ),
            )
        )

        if affordances:
            entity[
                "affordances"
            ] = affordances

        for (
            source_field,
            target_field,
        ) in (
            (
                "attributes",
                "attributes",
            ),
            (
                "count",
                "count",
            ),
            (
                "location",
                "scene_location",
            ),
        ):
            if source_field in element:
                entity[
                    target_field
                ] = deepcopy(
                    element[
                        source_field
                    ]
                )

        if "scene_summary" in scene:
            entity[
                "scene_summary"
            ] = deepcopy(
                scene.get(
                    "scene_summary"
                )
            )

        entities[
            entity_id
        ] = entity

    world = WorldState(
        snapshot
    )

    return (
        _apply_runtime_region_entity_rules(
            world,
            region_id=
                region_id,
            region=
                region,
        )
    )


def build_runtime_initial_world() -> WorldState:
    """
    Bootstrap all configured semantic observation regions exactly once.

    This is the production replacement for the test-local full-world bootstrap:
        describe_scene region(s)
        -> structured object region(s)
        -> one merged initial WorldState

    The bootstrap order comes from agent.config.scene_config.
    """

    world = WorldState({
        "entities": {},
        "relations": [],
        "meta": {
            "source":
                "plan_service.runtime_bootstrap",
        },
    })

    for region_id in (
        bootstrap_scene_region_ids()
    ):
        region = (
            _move_runtime_observer_to_region(
                region_id
            )
        )

        observation_config = (
            _runtime_observation_config(
                region
            )
        )

        observer = str(
            region.get(
                "observer"
            )
            or ""
        ).strip()

        if observer in {
            "world_state",
            "capture_world",
        }:
            world = capture_world(
                config=
                    observation_config,

                context=
                    world,

                active_region_id=
                    region_id,
            )

            world = (
                _apply_runtime_region_entity_rules(
                    world,
                    region_id=
                        region_id,
                    region=
                        region,
                )
            )

            continue

        if observer == "describe_scene":
            camera_name = (
                observation_config
                .camera_names[
                    0
                ]
            )

            result = (
                worldmodel_service
                .describe_scene(
                    camera_name=
                        camera_name,

                    vlm_backend=
                        region.get(
                            "vlm_backend",
                            "remote",
                        ),
                )
            )

            scene = (
                _scene_from_describe_response(
                    result
                )
            )

            world = (
                _bootstrap_describe_scene_region(
                    world,

                    region_id=
                        region_id,

                    region=
                        region,

                    camera_name=
                        camera_name,

                    scene=
                        scene,
                )
            )

            continue

        raise RuntimeError(
            "unsupported bootstrap scene observer: "
            f"region={region_id!r}, "
            f"observer={observer!r}"
        )

    return world


# ============================================================
# Runtime task interpretation
# ============================================================
def interpret_runtime_task(
    user_text: str,
    *,
    world: WorldState,
    grounded_targets: Any = None,
    planning_options: dict[str, Any] | None = None,
    openai_api_key: str | None = None,
) -> TaskSpec:
    """
    Human task -> Goals / CoordinationSpec.
    This is the semantic entry point for the new closed-loop runtime.
    It does not generate or execute a future action sequence.
    """
    options = (
        normalize_runtime_planning_options(
            planning_options
        )
    )
    return interpret_task(
        user_text,
        world=
            world,
        grounded_targets=
            grounded_targets,
        llm_options=
            options,
        openai_api_key=
            openai_api_key,
    )
def task_spec_to_dict(
    spec: TaskSpec,
) -> dict[str, Any]:
    """Serialize TaskSpec for service/API debugging."""
    goals = []
    if spec.goals is not None:
        for goal in spec.goals.conditions:
            goals.append({
                "goal_id":
                    goal.goal_id,
                "subject":
                    goal.subject,
                "field":
                    goal.field,
                "operator":
                    goal.operator,
                "value":
                    deepcopy(
                        goal.value
                    ),
                "depends_on":
                    list(
                        goal.depends_on
                    ),
            })
    coordination = {
        "maintain_until": [
            {
                "subject":
                    item.subject,
                "field":
                    item.field,
                "value":
                    deepcopy(
                        item.value
                    ),
                "until_goal":
                    item.until_goal,
            }
            for item
            in spec.coordination.maintain_until
        ],
        "device_rules": [
            {
                "function_name":
                    item.function_name,
                "subject_arg":
                    item.subject_arg,
                "subject_id":
                    item.subject_id,
                "device_id":
                    item.device_id,
                "mode":
                    item.mode,
            }
            for item
            in spec.coordination.device_rules
        ],
    }
    return {
        "summary":
            spec.summary,
        "goals":
            goals,
        "coordination":
            coordination,
        "motion_request":
            deepcopy(
                spec.motion_request
            ),
        "raw":
            deepcopy(
                spec.raw
            ),
    }

# ============================================================
# Manual UI Runtime
# ============================================================

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


_MANUAL_RUNTIME_LOCK = RLock()
_MANUAL_RUNTIME_SESSION: _ManualRuntimeSession | None = None


def _manual_action_summary(
    bound_dict: dict[str, Any],
) -> str:
    """
    Human-readable proposed action shown BEFORE PRE-COMMIT validation.

    This wording is intentionally neutral: it describes what Runtime wants to
    do, but does not instruct the human to execute it yet.
    """
    actions = bound_dict.get(
        "actions",
        [],
    )

    if not actions:
        return "Runtime 想做：未知動作"

    action = actions[0]
    function_name = action.get(
        "function_name"
    )
    arguments = action.get(
        "arguments",
        {},
    )
    device_id = action.get(
        "device_id"
    )

    if function_name == "pick_object":
        text = (
            "Runtime 想做：拿起 "
            f"{arguments.get('object_id')}"
        )
    elif function_name == "place_object":
        text = (
            "Runtime 想做：把 "
            f"{arguments.get('object_id')} "
            "放到／放進 "
            f"{arguments.get('destination_id')}"
        )
    elif function_name in {
        "open_container",
        "open_trash_can",
    }:
        target = (
            arguments.get(
                "container_id"
            )
            or arguments.get(
                "object_id"
            )
        )
        text = (
            f"Runtime 想做：打開 {target}"
        )
    elif function_name in {
        "close_container",
        "close_trash_can",
    }:
        target = (
            arguments.get(
                "container_id"
            )
            or arguments.get(
                "object_id"
            )
        )
        text = (
            f"Runtime 想做：關閉 {target}"
        )
    else:
        text = (
            "Runtime 想做："
            f"{function_name} {arguments}"
        )

    if (
        isinstance(
            device_id,
            str,
        )
        and device_id
    ):
        text += (
            f"\n綁定裝置：{device_id}"
        )

    return text


def _manual_instruction(
    bound_dict: dict[str, Any],
) -> str:
    actions = bound_dict.get(
        "actions",
        [],
    )

    if not actions:
        return "請手動完成 Runtime 指定的動作。"

    action = actions[0]
    function_name = action.get(
        "function_name"
    )
    arguments = action.get(
        "arguments",
        {},
    )
    device_id = action.get(
        "device_id"
    )

    if function_name == "pick_object":
        base = (
            f"請手動拿起 {arguments.get('object_id')}，"
            "並移離原本位置，模擬機械手已經夾住它。"
        )
    elif function_name == "place_object":
        base = (
            f"請手動把 {arguments.get('object_id')} "
            f"放到／放進 {arguments.get('destination_id')}。"
        )
    elif function_name in {
        "open_container",
        "open_trash_can",
    }:
        target = (
            arguments.get(
                "container_id"
            )
            or arguments.get(
                "object_id"
            )
        )
        base = f"請手動打開 {target}。"
    elif function_name in {
        "close_container",
        "close_trash_can",
    }:
        target = (
            arguments.get(
                "container_id"
            )
            or arguments.get(
                "object_id"
            )
        )
        base = f"請手動關閉 {target}。"
    else:
        base = (
            "請手動完成："
            f"{function_name} {arguments}"
        )

    if (
        isinstance(
            device_id,
            str,
        )
        and device_id
    ):
        base += (
            f"\nRuntime 綁定裝置：{device_id}"
        )

    return base


def _manual_reconcile_holding_state(
    world: WorldState,
) -> tuple[
    WorldState,
    list[dict[str, Any]],
]:
    """
    Keep the two sides of the manual-mode holding relation consistent:

        object.held_by == arm_id
        arm.holding == object_id

    This is only a manual-executor semantic repair.  It does not infer physical
    grasp success and is not used by the production physical executor.
    """

    snapshot = world.snapshot()
    entities = snapshot.get(
        "entities",
        {},
    )

    if not isinstance(
        entities,
        dict,
    ):
        return world, []

    changes: list[
        dict[str, Any]
    ] = []

    held_objects_by_arm: dict[
        str,
        list[str],
    ] = {}

    for entity_id, entity in entities.items():
        if not isinstance(
            entity,
            dict,
        ):
            continue

        held_by = entity.get(
            "held_by"
        )

        if not isinstance(
            held_by,
            str,
        ) or not held_by:
            continue

        arm = entities.get(
            held_by
        )

        if not isinstance(
            arm,
            dict,
        ) or arm.get(
            "type"
        ) != "robot_arm":
            continue

        held_objects_by_arm.setdefault(
            held_by,
            [],
        ).append(
            entity_id
        )

    for arm_id, object_ids in held_objects_by_arm.items():
        if len(
            object_ids
        ) > 1:
            raise RuntimeError(
                "manual semantic state inconsistent: "
                f"{arm_id} has multiple objects with held_by={arm_id}: "
                f"{object_ids!r}"
            )

        object_id = object_ids[
            0
        ]
        arm = entities[
            arm_id
        ]
        current = arm.get(
            "holding"
        )

        if current in {
            None,
            "",
        }:
            arm[
                "holding"
            ] = object_id
            changes.append({
                "subject":
                    arm_id,
                "field":
                    "holding",
                "old":
                    current,
                "new":
                    object_id,
                "reason":
                    "reconcile_from_object.held_by",
            })
        elif current != object_id:
            raise RuntimeError(
                "manual semantic state inconsistent: "
                f"{arm_id}.holding={current!r} but "
                f"{object_id}.held_by={arm_id!r}"
            )

    for arm_id, arm in entities.items():
        if (
            not isinstance(
                arm,
                dict,
            )
            or arm.get(
                "type"
            ) != "robot_arm"
        ):
            continue

        object_id = arm.get(
            "holding"
        )

        if not isinstance(
            object_id,
            str,
        ) or not object_id:
            continue

        obj = entities.get(
            object_id
        )

        if not isinstance(
            obj,
            dict,
        ):
            continue

        held_by = obj.get(
            "held_by"
        )

        if held_by in {
            None,
            "",
        }:
            obj[
                "held_by"
            ] = arm_id
            changes.append({
                "subject":
                    object_id,
                "field":
                    "held_by",
                "old":
                    held_by,
                "new":
                    arm_id,
                "reason":
                    "reconcile_from_arm.holding",
            })
        elif held_by != arm_id:
            raise RuntimeError(
                "manual semantic state inconsistent: "
                f"{arm_id}.holding={object_id!r} but "
                f"{object_id}.held_by={held_by!r}"
            )

    return WorldState(
        snapshot
    ), changes


def _manual_carry_semantic_state(
    *,
    semantic_world: WorldState,
    observed_world: WorldState,
) -> WorldState:
    """
    Preserve only manual semantic facts that perception cannot yet verify.
    Fresh visual facts remain authoritative.
    """

    semantic = semantic_world.snapshot()
    observed = observed_world.snapshot()

    old_entities = semantic.get(
        "entities",
        {},
    )
    new_entities = observed.get(
        "entities",
        {},
    )

    if (
        not isinstance(
            old_entities,
            dict,
        )
        or not isinstance(
            new_entities,
            dict,
        )
    ):
        return observed_world

    semantic_fields = (
        "held_by",
        "location",
        "holding",
        "controlling",
    )

    for entity_id, current in new_entities.items():
        if not isinstance(
            current,
            dict,
        ):
            continue

        previous = old_entities.get(
            entity_id
        )

        if not isinstance(
            previous,
            dict,
        ):
            continue

        for field_name in semantic_fields:
            if field_name in previous:
                current[
                    field_name
                ] = deepcopy(
                    previous[
                        field_name
                    ]
                )

    for entity_id, previous in old_entities.items():
        if (
            entity_id in new_entities
            or not isinstance(
                previous,
                dict,
            )
        ):
            continue

        held_by = previous.get(
            "held_by"
        )
        location = previous.get(
            "location"
        )
        holding = previous.get(
            "holding"
        )
        controlling = previous.get(
            "controlling"
        )
        tags = previous.get(
            "tags"
        )
        affordances = previous.get(
            "affordances"
        )

        is_temporary_placement = (
            isinstance(
                tags,
                list,
            )
            and "temporary_placement"
            in tags
        ) or (
            isinstance(
                affordances,
                list,
            )
            and "temporary_placement"
            in affordances
        )

        keep = (
            isinstance(
                held_by,
                str,
            )
            and bool(
                held_by
            )
        ) or (
            isinstance(
                location,
                str,
            )
            and bool(
                location
            )
        ) or (
            isinstance(
                holding,
                str,
            )
            and bool(
                holding
            )
        ) or (
            isinstance(
                controlling,
                str,
            )
            and bool(
                controlling
            )
        ) or is_temporary_placement

        if not keep:
            continue

        carried = deepcopy(
            previous
        )

        for field_name in (
            "bbox",
            "position",
            "yaw_deg",
            "confidence",
            "grasp",
            "observed_at",
            "camera_source",
        ):
            carried.pop(
                field_name,
                None,
            )

        carried[
            "source"
        ] = "manual_confirmed_overlay"

        new_entities[
            entity_id
        ] = carried

    return WorldState(
        observed
    )


def _manual_blocked_report(
    bound,
    *,
    exc: Exception,
) -> DispatchExecutionReport:
    bound_actions = getattr(
        bound,
        "actions",
        None,
    )

    if not bound_actions:
        raise RuntimeError(
            "BoundDispatch has no actions"
        )

    feedback = getattr(
        exc,
        "feedback",
        None,
    )

    raw_result = {
        "status":
            "blocked",
        "reason":
            "PRECONDITION_CHANGED",
        "error_type":
            type(
                exc
            ).__name__,
        "message":
            str(
                exc
            ),
    }

    if isinstance(
        feedback,
        dict,
    ):
        raw_result[
            "feedback"
        ] = deepcopy(
            feedback
        )

    result = ActionExecutionResult(
        bound_action=
            bound_actions[
                0
            ],
        command_success=
            False,
        verified_success=
            False,
        error=
            "PRECOMMIT_VALIDATION_FAILED",
        raw_result=
            raw_result,
    )

    return DispatchExecutionReport(
        results=[
            result,
        ],
    )


class ManualUIRuntimeExecutor:
    """
    UI-driven manual manipulation executor.

    Observation-pose movement and sensing are real. Manipulation is performed
    by the human. Confirmation is the manual commit point; ToolSpec effects are
    then applied by FakeDispatchExecutor solely as a semantic overlay.
    """

    def __init__(
        self,
        *,
        session: _ManualRuntimeSession,
    ):
        self.session = session
        self.fake = FakeDispatchExecutor(
            failure_probability=0.0
        )
        self.observe_dispatch = (
            build_runtime_dispatch_observer()
        )

    def _check_cancelled(
        self,
    ) -> None:
        with self.session.lock:
            cancelled = (
                self.session
                .cancel_requested
            )

        if cancelled:
            raise ManualRuntimeCancelled(
                "manual Runtime cancelled by user"
            )

    def _wait_for_human(
        self,
        *,
        step_number: int,
        bound_dict: dict[str, Any],
    ) -> None:
        instruction = _manual_instruction(
            bound_dict
        )

        self.session.continue_event.clear()
        self.session.update_step(
            step_number,
            status=
                "awaiting_confirmation",
            gate_status=
                "awaiting_human",
            instruction=
                instruction,
        )
        self.session.update(
            status=
                "awaiting_confirmation",
            stage=
                "manual_action",
            instruction=
                instruction,
            pending_action=
                deepcopy(
                    bound_dict
                ),
            pending_step=
                step_number,
            proposed_state=
                "awaiting_human",
            proposed_reason=
                None,
            message=
                "PRE-COMMIT 已通過。請依照指示手動完成動作，完成後按確認。",
        )
        self.session.record(
            "manual_action_requested",
            message=
                instruction,
            data={
                "step":
                    step_number,
                "action":
                    bound_dict,
            },
        )

        self.session.continue_event.wait()
        self._check_cancelled()

        self.session.update_step(
            step_number,
            status=
                "human_confirmed",
            gate_status=
                "human_confirmed",
            confirmed_at=
                _manual_now(),
        )
        self.session.update(
            status=
                "running",
            stage=
                "semantic_commit",
            instruction=
                None,
            pending_action=
                None,
            proposed_state=
                "human_confirmed",
            message=
                "已收到人工確認，正在套用 manual semantic effect。",
        )
        self.session.record(
            "manual_action_confirmed",
            data={
                "step":
                    step_number,
            },
        )

    def execute(
        self,
        bound,
        *,
        world: WorldState,
        tools,
        goals=None,
        coordination=None,
    ):
        self._check_cancelled()

        bound_dict = bound.as_dict()
        step_number = self.session.begin_step(
            action=
                bound_dict,
            world_before=
                world,
        )

        proposal = _manual_action_summary(
            bound_dict
        )

        self.session.update_step(
            step_number,
            proposal=
                proposal,
            gate_status=
                "checking",
        )
        self.session.update(
            status=
                "running",
            stage=
                "precommit_observation",
            pending_step=
                step_number,
            proposed_action=
                deepcopy(
                    bound_dict
                ),
            proposed_instruction=
                proposal,
            proposed_step=
                step_number,
            proposed_state=
                "checking",
            proposed_reason=
                None,
            message=(
                f"Step {step_number}: Runtime 已選擇下一個動作；"
                "正在執行 PRE-COMMIT fresh observation / validation。"
            ),
        )
        self.session.record(
            "action_proposed",
            message=
                proposal,
            data={
                "step":
                    step_number,
                "action":
                    bound_dict,
            },
        )
        self.session.record(
            "step_started",
            data={
                "step":
                    step_number,
                "action":
                    bound_dict,
                "world_before":
                    _manual_world_summary(
                        world
                    ),
            },
        )

        raw_fresh_world = self.observe_dispatch(
            bound,
            world,
        )

        # PRE-COMMIT observation is visually fresh, but fields such as
        # object.held_by / arm.holding are manual semantic state and are not
        # produced by the physical perception stack.  Preserve those facts
        # before validating the selected action; otherwise a fresh arm_service
        # entity can erase arm.holding while object.held_by is still carried,
        # causing a false PHYSICAL_PRECONDITION_FAILED for place_object.
        fresh_world = _manual_carry_semantic_state(
            semantic_world=
                world,
            observed_world=
                raw_fresh_world,
        )
        fresh_world, precommit_reconcile_changes = (
            _manual_reconcile_holding_state(
                fresh_world
            )
        )

        self.session.update_step(
            step_number,
            status=
                "precommit_validation",
            precommit_observation_raw=
                _manual_world_summary(
                    raw_fresh_world
                ),
            precommit_world=
                _manual_world_summary(
                    fresh_world
                ),
            precommit_reconcile_changes=
                precommit_reconcile_changes,
        )
        self.session.update(
            latest_world=
                fresh_world.snapshot(),
            stage=
                "precommit_validation",
            message=(
                f"Step {step_number}: "
                "fresh observation 完成，正在重新驗證 preconditions。"
            ),
        )
        self.session.record(
            "precommit_observed",
            data={
                "step":
                    step_number,
                "raw_world":
                    _manual_world_summary(
                        raw_fresh_world
                    ),
                "world":
                    _manual_world_summary(
                        fresh_world
                    ),
                "reconcile_changes":
                    precommit_reconcile_changes,
            },
        )

        if goals is not None:
            try:
                validate_dispatch(
                    bound,
                    world=
                        fresh_world,
                    goals=
                        goals,
                    tools=
                        tools,
                    coordination=
                        coordination,
                )
            except Exception as exc:
                error_row = {
                    "type":
                        type(
                            exc
                        ).__name__,
                    "message":
                        str(
                            exc
                        ),
                    "feedback":
                        deepcopy(
                            getattr(
                                exc,
                                "feedback",
                                None,
                            )
                        ),
                }
                blocked_reason = _manual_short_error(
                    exc
                )

                self.session.update_step(
                    step_number,
                    status=
                        "blocked",
                    gate_status=
                        "blocked",
                    blocked_reason=
                        blocked_reason,
                    completed_at=
                        _manual_now(),
                    error=
                        error_row,
                    world_after=
                        _manual_world_summary(
                            fresh_world
                        ),
                )
                self.session.update(
                    status=
                        "running",
                    stage=
                        "precommit_blocked",
                    proposed_state=
                        "blocked",
                    proposed_reason=
                        blocked_reason,
                    message=(
                        f"Step {step_number}: PRE-COMMIT 驗證失敗。"
                        "上方仍會顯示 Runtime 原本想做的動作；"
                        "此動作沒有要求人類執行，Runtime 將以 fresh state 重規劃。"
                    ),
                )
                self.session.record(
                    "precommit_blocked",
                    message=
                        _manual_short_error(
                            exc
                        ),
                    data={
                        "step":
                            step_number,
                        "error":
                            error_row,
                    },
                )
                return (
                    fresh_world,
                    _manual_blocked_report(
                        bound,
                        exc=
                            exc,
                    ),
                )

        self.session.update_step(
            step_number,
            gate_status=
                "validated",
        )
        self.session.update(
            proposed_state=
                "validated",
            proposed_reason=
                None,
        )
        self.session.record(
            "precommit_validation_passed",
            data={
                "step":
                    step_number,
            },
        )

        self._wait_for_human(
            step_number=
                step_number,
            bound_dict=
                bound_dict,
        )

        self._check_cancelled()

        semantic_world_raw, report = self.fake.execute(
            bound,
            world=
                fresh_world,
            tools=
                tools,
        )

        semantic_world, reconcile_changes = (
            _manual_reconcile_holding_state(
                semantic_world_raw
            )
        )

        self.session.update_step(
            step_number,
            status=
                "postcommit_observation",
            semantic_world_raw=
                _manual_world_summary(
                    semantic_world_raw
                ),
            semantic_world=
                _manual_world_summary(
                    semantic_world
                ),
            reconcile_changes=
                reconcile_changes,
        )
        self.session.update(
            stage=
                "postcommit_observation",
            latest_world=
                semantic_world.snapshot(),
            message=(
                f"Step {step_number}: human commit 完成，"
                "正在執行 fresh POST-COMMIT observation。"
            ),
        )
        self.session.record(
            "semantic_effect_applied",
            data={
                "step":
                    step_number,
                "raw_world":
                    _manual_world_summary(
                        semantic_world_raw
                    ),
                "reconciled_world":
                    _manual_world_summary(
                        semantic_world
                    ),
                "reconcile_changes":
                    reconcile_changes,
            },
        )

        observed_world = self.observe_dispatch(
            bound,
            semantic_world,
            report,
        )

        updated_world = _manual_carry_semantic_state(
            semantic_world=
                semantic_world,
            observed_world=
                observed_world,
        )

        updated_world, post_reconcile_changes = (
            _manual_reconcile_holding_state(
                updated_world
            )
        )

        all_reconcile_changes = (
            list(
                reconcile_changes
            )
            + list(
                post_reconcile_changes
            )
        )

        self.session.update_step(
            step_number,
            status=
                "completed",
            gate_status=
                "completed",
            completed_at=
                _manual_now(),
            post_observation_raw=
                _manual_world_summary(
                    observed_world
                ),
            world_after=
                _manual_world_summary(
                    updated_world
                ),
            reconcile_changes=
                all_reconcile_changes,
        )
        self.session.update(
            latest_world=
                updated_world.snapshot(),
            status=
                "running",
            stage=
                "runtime",
            proposed_state=
                "completed",
            proposed_reason=
                None,
            message=(
                f"Step {step_number}: POST-COMMIT observation 完成，"
                "Runtime 繼續。"
            ),
        )
        self.session.record(
            "step_completed",
            data={
                "step":
                    step_number,
                "post_observation_raw":
                    _manual_world_summary(
                        observed_world
                    ),
                "world_after":
                    _manual_world_summary(
                        updated_world
                    ),
                "reconcile_changes":
                    all_reconcile_changes,
            },
        )

        return (
            updated_world,
            report,
        )


def _apply_manual_force_device(
    task_spec: TaskSpec,
    *,
    world: WorldState,
    tools,
    force_device_id: str | None,
) -> tuple[TaskSpec, list[str]]:
    """
    Apply an explicit hard device constraint for Manual Runtime.

    Free-text instructions are not treated as a hardware guarantee.  This
    helper converts the UI/API force_device_id flag into required DeviceRules
    for every runtime tool whose target type is robot_arm.

    Existing LLM-produced device rules for those robot-arm functions are
    replaced so the explicit flag cannot conflict with an inferred rule.
    """

    if force_device_id is None:
        return task_spec, []

    force_device_id = str(
        force_device_id
    ).strip()

    if not force_device_id:
        return task_spec, []

    if not world.has_entity(
        force_device_id
    ):
        raise ValueError(
            "force_device_id references unknown WorldState entity: "
            f"{force_device_id!r}"
        )

    device_type = world.get(
        force_device_id,
        "type",
    )

    if device_type != "robot_arm":
        raise ValueError(
            "force_device_id 必須指向 robot_arm entity；"
            f"{force_device_id!r} type={device_type!r}"
        )

    forced_functions = sorted({
        tool.name
        for tool
        in tools.values()
        if getattr(
            tool,
            "device_target_type",
            None,
        ) == "robot_arm"
    })

    if not forced_functions:
        raise RuntimeError(
            "runtime tool catalog 沒有 robot_arm tools，"
            "無法套用 force_device_id"
        )

    forced_set = set(
        forced_functions
    )

    preserved_rules = [
        rule
        for rule
        in task_spec.coordination.device_rules
        if rule.function_name
        not in forced_set
    ]

    forced_rules = [
        DeviceRule(
            function_name=
                function_name,
            subject_arg=
                None,
            subject_id=
                None,
            device_id=
                force_device_id,
            mode=
                "required",
        )
        for function_name
        in forced_functions
    ]

    coordination = CoordinationSpec(
        maintain_until=
            list(
                task_spec.coordination.maintain_until
            ),
        device_rules=
            preserved_rules
            + forced_rules,
    )

    raw = deepcopy(
        task_spec.raw
    )

    if not isinstance(
        raw,
        dict,
    ):
        raw = {}

    raw[
        "manual_runtime_force_device"
    ] = {
        "device_id":
            force_device_id,
        "mode":
            "required",
        "functions":
            forced_functions,
    }

    return (
        TaskSpec(
            goals=
                task_spec.goals,
            coordination=
                coordination,
            motion_request=
                deepcopy(
                    task_spec.motion_request
                ),
            summary=
                task_spec.summary,
            raw=
                raw,
        ),
        forced_functions,
    )


def _manual_runtime_worker(
    session: _ManualRuntimeSession,
) -> None:
    try:
        session.update(
            status=
                "running",
            stage=
                "bootstrap",
            message=
                "正在建立 initial multi-region WorldState。",
        )
        session.record(
            "bootstrap_started"
        )

        world = build_runtime_initial_world()

        session.update(
            latest_world=
                world.snapshot(),
            stage=
                "task_interpretation",
            message=
                "Initial WorldState 完成，正在解析 Runtime task。",
        )
        session.record(
            "bootstrap_completed",
            data={
                "world":
                    _manual_world_summary(
                        world
                    ),
            },
        )

        task_spec = interpret_runtime_task(
            session.user_text,
            world=
                world,
            grounded_targets=
                None,
            planning_options=
                session.planning_options,
            openai_api_key=
                session.openai_api_key,
        )

        if task_spec.goals is None:
            raise RuntimeError(
                "Task Interpreter returned no goals"
            )

        tool_catalog = runtime_tools()

        task_spec, forced_functions = (
            _apply_manual_force_device(
                task_spec,
                world=
                    world,
                tools=
                    tool_catalog,
                force_device_id=
                    session.force_device_id,
            )
        )

        if forced_functions:
            session.record(
                "force_device_applied",
                message=(
                    "Manual Runtime hard device constraint applied: "
                    f"{session.force_device_id}"
                ),
                data={
                    "device_id":
                        session.force_device_id,
                    "mode":
                        "required",
                    "functions":
                        forced_functions,
                },
            )

        task_spec_dict = task_spec_to_dict(
            task_spec
        )

        session.update(
            task_spec=
                task_spec_dict,
            stage=
                "runtime_setup",
            message=
                "TaskSpec 完成，正在建立 Runtime Coordinator。",
        )
        session.record(
            "task_interpreted",
            data={
                "task_spec":
                    task_spec_dict,
            },
        )

        normalized = normalize_runtime_planning_options(
            session.planning_options
        )
        brain_options = (
            normalized.get(
                "step_planning"
            )
            or {}
        )

        # Runtime Step Planning is an explicit user-facing phase.
        # Request configuration is authoritative so the UI cannot be
        # silently overridden by stale environment variables.
        brain_provider = str(
            brain_options.get(
                "provider",
                "local",
            )
            or "local"
        ).strip().lower()
        brain_model = brain_options.get(
            "model"
        )

        brain = RuntimeBrain(
            provider=
                brain_provider,
            model=
                brain_model,
            api_key=
                session.openai_api_key,
            temperature=
                0.1,
            max_tokens=
                1024,
            timeout=
                120,
        )

        session.record(
            "runtime_brain_configured",
            data={
                "phase":
                    "step_planning",
                "provider":
                    brain_provider,
                "model":
                    brain_model,
                "api_key_source": (
                    (
                        "request"
                        if session.openai_api_key
                        else (
                            "environment"
                            if os.environ.get(
                                "OPENAI_API_KEY"
                            )
                            else None
                        )
                    )
                    if brain_provider == "openai"
                    else None
                ),
            },
        )

        executor = ManualUIRuntimeExecutor(
            session=
                session,
        )

        execution_memory_path = (
            _MANUAL_RUNTIME_DATA_DIR
            / session.session_id
            / "execution_memory.jsonl"
        )
        execution_memory = ExecutionMemory(
            execution_memory_path,
            max_entries=10,
        )
        session.update(
            execution_memory_path=
                str(execution_memory_path),
        )
        session.record(
            "execution_memory_ready",
            data={
                "path":
                    str(execution_memory_path),
                "max_entries":
                    10,
            },
        )

        coordinator = RuntimeCoordinator(
            brain=
                brain,
            binder=
                StaticWorldDeviceBinder(),
            executor=
                executor,
            tools=
                tool_catalog,
            execution_memory=
                execution_memory,
            max_parallel_actions=
                1,
            max_decision_attempts=
                5,
        )

        session.update(
            stage=
                "runtime",
            message=
                "Manual closed-loop Runtime 已開始。",
        )
        session.record(
            "runtime_started"
        )

        run_result = coordinator.run_until_done(
            world=
                world,
            goals=
                task_spec.goals,
            coordination=
                task_spec.coordination,
            max_iterations=
                session.max_iterations,
        )

        result = {
            "success":
                run_result.success,
            "iterations":
                run_result.iterations,
            "final_world":
                run_result.world.snapshot(),
            "trace": [
                {
                    "iteration":
                        row.iteration,
                    "dispatch":
                        deepcopy(
                            row.dispatch
                        ),
                    "bound_dispatch":
                        deepcopy(
                            row.bound_dispatch
                        ),
                    "execution":
                        deepcopy(
                            row.execution
                        ),
                    "progress_feedback":
                        deepcopy(
                            row.progress_feedback
                        ),
                }
                for row
                in run_result.trace
            ],
        }

        session.update(
            status=(
                "complete"
                if run_result.success
                else "stopped"
            ),
            stage=(
                "complete"
                if run_result.success
                else "max_iterations"
            ),
            message=(
                "Runtime task 完成。"
                if run_result.success
                else "已到 max_iterations，Runtime 尚未完成 task。"
            ),
            instruction=
                None,
            pending_action=
                None,
            latest_world=
                run_result.world.snapshot(),
            result=
                result,
        )
        session.record(
            "runtime_finished",
            data={
                "success":
                    run_result.success,
                "iterations":
                    run_result.iterations,
            },
        )

    except ManualRuntimeCancelled as exc:
        session.update(
            status=
                "cancelled",
            stage=
                "cancelled",
            message=
                str(
                    exc
                ),
            instruction=
                None,
            pending_action=
                None,
        )
        session.record(
            "runtime_cancelled",
            message=
                str(
                    exc
                ),
        )

    except Exception as exc:
        full_message = str(
            exc
        )
        summary = _manual_short_error(
            exc
        )

        session.update(
            status=
                "failed",
            stage=
                "failed",
            message=
                summary,
            instruction=
                None,
            pending_action=
                None,
            error={
                "type":
                    type(
                        exc
                    ).__name__,
                "summary":
                    summary,
                "message":
                    full_message,
            },
        )
        session.record(
            "runtime_failed",
            message=
                summary,
            data={
                "error_type":
                    type(
                        exc
                    ).__name__,
                "full_message":
                    full_message,
                "latest_world":
                    _manual_world_summary(
                        session.latest_world
                    ),
            },
        )


def start_manual_runtime(
    user_text: str,
    *,
    planning_options: dict[str, Any] | None = None,
    openai_api_key: str | None = None,
    max_iterations: int = 10,
    force_device_id: str | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(
            user_text,
            str,
        )
        or not user_text.strip()
    ):
        raise ValueError(
            "user_text 必須是非空字串"
        )

    if (
        not isinstance(
            max_iterations,
            int,
        )
        or max_iterations < 1
    ):
        raise ValueError(
            "max_iterations 必須是 >= 1 的整數"
        )

    normalized = normalize_runtime_planning_options(
        planning_options
    )

    if openai_api_key is not None:
        if (
            not isinstance(
                openai_api_key,
                str,
            )
            or not openai_api_key.strip()
        ):
            raise ValueError(
                "openai_api_key 必須是非空字串或 null"
            )
        openai_api_key = openai_api_key.strip()

    if force_device_id is not None:
        force_device_id = str(
            force_device_id
        ).strip() or None

    global _MANUAL_RUNTIME_SESSION

    with _MANUAL_RUNTIME_LOCK:
        current = _MANUAL_RUNTIME_SESSION

        if (
            current is not None
            and current.status
            in {
                "starting",
                "running",
                "awaiting_confirmation",
            }
        ):
            raise RuntimeError(
                "已有 manual Runtime session 正在執行"
            )

        session = _ManualRuntimeSession(
            session_id=
                uuid4().hex,
            user_text=
                user_text.strip(),
            planning_options=
                normalized,
            max_iterations=
                max_iterations,
            openai_api_key=
                openai_api_key,
            force_device_id=
                force_device_id,
        )
        session.record(
            "session_created",
            data={
                "planning_options":
                    normalized,
                "openai_api_key_supplied":
                    bool(openai_api_key),
                "max_iterations":
                    max_iterations,
                "force_device_id":
                    force_device_id,
            },
        )

        thread = Thread(
            target=
                _manual_runtime_worker,
            args=(
                session,
            ),
            daemon=
                True,
            name=(
                "manual-runtime-"
                + session.session_id[
                    :8
                ]
            ),
        )

        session.thread = thread
        _MANUAL_RUNTIME_SESSION = session
        thread.start()

    return session.snapshot()


def get_manual_runtime_status(
    session_id: str | None = None,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if session is None:
        raise RuntimeError(
            "目前沒有 manual Runtime session"
        )

    if (
        isinstance(
            session_id,
            str,
        )
        and session_id
        and session_id
        != session.session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    return session.snapshot()


def confirm_manual_runtime_step(
    session_id: str,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if (
        session is None
        or session.session_id
        != session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    with session.lock:
        if session.status != "awaiting_confirmation":
            raise RuntimeError(
                "manual Runtime 目前不在等待確認"
            )

        step_number = session.pending_step
        session.status = "running"
        session.stage = "manual_action_confirmed"
        session.message = "已收到人工確認，Runtime 繼續。"

    session.record(
        "confirm_api_received",
        data={
            "step":
                step_number,
        },
    )
    session.continue_event.set()

    return session.snapshot()


def cancel_manual_runtime(
    session_id: str,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if (
        session is None
        or session.session_id
        != session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    with session.lock:
        if session.status in {
            "complete",
            "stopped",
            "failed",
            "cancelled",
        }:
            return session.snapshot()

        session.cancel_requested = True
        session.message = "正在取消 manual Runtime。"

    session.record(
        "cancel_requested"
    )
    session.continue_event.set()

    return session.snapshot()


def get_health() -> dict[str, Any]:
    return response.success(
        MODULE,
        "health",
        data={
            "service":
                "plan",
            "runtime_task_interpreter":
                True,
            "providers":
                sorted(
                    _RUNTIME_ALLOWED_PROVIDERS
                ),
            "executes_robot":
                True,
            "manual_runtime":
                True,
            "physical_manipulation":
                False,
            "task_interpreter_skill":
                "agent/skills/runtime_task_interpreter.md",
        },
    )
