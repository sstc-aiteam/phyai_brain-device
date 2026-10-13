"""
Runtime-specific production observation integration.

This adapter owns the camera/region observation logic needed by the
closed-loop Runtime. Infrastructure services are consumed as external
dependencies; this module does not depend on services.plan_service.
"""

from __future__ import annotations

from typing import Any
from agent.config.scene_config import bootstrap_scene_region_ids
from services import worldmodel_service
from .observation import capture_world

from .observation import (
    ObservationConfig,
    capture_world,
    make_dispatch_observer,
)
from ..domain.world import WorldState
from services import arm_service, worldmodel_service
from copy import deepcopy
from agent.config.scene_config import (
    bootstrap_scene_region_ids,
    get_scene_region,
)
from time import sleep


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
