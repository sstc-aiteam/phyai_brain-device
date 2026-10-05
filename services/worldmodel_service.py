"""
 World Model service facade.

Responsibilities:
- scene description
- occlusion relations
- canonical live WorldState collection
- compact object-relation API payload
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import logging
from typing import Any
import config
from agent.runtime.world_state import WorldState
from agent.vlm import occultation, vlm_describe, material,grasp
from services import arm_service, camera_service, perception_service
from utils import response


MODULE = "vlm"

_DETECTION_CAMERAS = (
    "left",
    "right",
)

_INSTAORDER_PC_URL = (
    "http://192.168.50.37:8021"
)
_MATERIAL_PC_URL = (
    "http://192.168.50.37:8022"
)

_INSTAORDER_RELATION_DISTANCE_PX = 30
_INSTAORDER_CONFIDENCE_THRESHOLD = 0.55

_INSTAORDER_EXCLUDED_CLASSES = (
    "chair_surface",
)

logger = logging.getLogger(__name__)


def assign_object_ids(objects):
    """Assign one consistent set of IDs to a collected object batch."""
    if not isinstance(objects, list):
        raise TypeError("objects 必須是 list")

    assigned = []
    used_ids = set()
    class_counts = {}

    for obj in objects:
        if not isinstance(obj, dict):
            assigned.append(obj)
            continue

        object_id = None
        for key in ("object_id", "track_id", "id", "instance_id"):
            value = obj.get(key)
            if value is not None and str(value).strip():
                object_id = str(value).strip()
                break

        if object_id is None or object_id in used_ids:
            class_name = str(obj.get("class_name") or "object").strip().lower()
            base_name = "".join(
                character if character.isalnum() or character == "_" else "_"
                for character in class_name
            ).strip("_").lower() or "object"
            suffix = class_counts.get(base_name, 0) + 1
            object_id = f"{base_name}_{suffix}"
            while object_id in used_ids:
                suffix += 1
                object_id = f"{base_name}_{suffix}"
            class_counts[base_name] = suffix

        used_ids.add(object_id)
        assigned.append({**obj, "object_id": object_id})

    return assigned


# ============================================================
# Common helpers
# ============================================================

def _utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# Scene Description
# ============================================================

def describe_scene(
    *,
    camera_name=None,
    vlm_backend=None,
):
    return vlm_describe.describe_scene(
        camera_name=camera_name,
        vlm_backend=vlm_backend,
    )

# ============================================================
# Health
# ============================================================

def get_health():
    return response.success(
        MODULE,
        "health",
        data={
            "mode":
                "facade",

            "capabilities": {
                "describe_scene":
                    vlm_describe.get_health(),

                "occlusion":
                    {
                        "implementation":
                            "agent.vlm.occultation",
                        "backend":
                            "instaorder",
                    },
                "grasp":
                    {
                        "implementation":
                            "agent.vlm.grasp",
                        "mode":
                            "rule_based_current_affordance",
                    },
            },

            "implementations": {
                "describe_scene":
                    "agent.vlm.vlm_describe",

                "occlusion":
                    "agent.vlm.occultation",
                "grasp":
                    "agent.vlm.grasp",
            },

            "background_monitor":
                False,
        },
    )


# ============================================================
# Robot entities
# ============================================================

def collect_robot_entities() -> list[
    dict[str, Any]
]:
    response = arm_service.get_arm_status()

    if (
        not isinstance(
            response,
            dict,
        )
        or response.get(
            "result"
        )
        is not True
    ):
        return []

    data = response.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):
        return []

    raw_arms = data.get(
        "arms"
    )

    if not isinstance(
        raw_arms,
        list,
    ):
        return []

    observed_at = _utc_now_iso()
    entities = []

    for row in raw_arms:
        if not isinstance(
            row,
            dict,
        ):
            continue

        arm_name = str(
            row.get(
                "arm_name"
            )
            or ""
        ).strip()

        if not arm_name:
            continue

        status = row.get(
            "status"
        )

        if not isinstance(
            status,
            dict,
        ):
            status = {}

        entity = {
            "entity_id": (
                f"{arm_name}_arm"
                if arm_name
                in {
                    "left",
                    "right",
                }
                else arm_name
            ),
            "type":
                "robot_arm",
            "tags":
                ["manipulator"],
            "arm_name":
                arm_name,
            "driver":
                row.get(
                    "driver"
                ),
            "connected":
                status.get(
                    "connected"
                )
                is True,
            "pose":
                deepcopy(
                    status.get(
                        "pose"
                    )
                ),
            "joints":
                deepcopy(
                    status.get(
                        "joints"
                    )
                ),
            "source":
                "arm_service",
            "observed_at":
                observed_at,
        }

        for field in (
            "holding",
            "controlling",
            "location",
        ):
            if field in status:
                entity[field] = deepcopy(
                    status[field]
                )
            elif field in row:
                entity[field] = deepcopy(
                    row[field]
                )

        entities.append(
            entity
        )

    return entities


# ============================================================
# Camera entities
# ============================================================

def collect_camera_entities() -> list[
    dict[str, Any]
]:
    response = (
        camera_service
        .get_camera_status()
    )

    if (
        not isinstance(
            response,
            dict,
        )
        or response.get(
            "result"
        )
        is not True
    ):
        return []

    data = response.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):
        return []

    raw_cameras = data.get(
        "cameras"
    )

    if not isinstance(
        raw_cameras,
        list,
    ):
        return []

    observed_at = _utc_now_iso()
    entities = []

    for row in raw_cameras:
        if not isinstance(
            row,
            dict,
        ):
            continue

        camera_name = str(
            row.get(
                "camera_name"
            )
            or ""
        ).strip()

        if not camera_name:
            continue

        status = row.get(
            "status"
        )

        if not isinstance(
            status,
            dict,
        ):
            status = {}

        entities.append({
            "entity_id":
                f"{camera_name}_camera",
            "type":
                "camera",
            "tags":
                [
                    "camera",
                    "sensor",
                ],
            "camera_name":
                camera_name,
            "driver":
                row.get(
                    "driver"
                ),
            "connected":
                status.get(
                    "connected"
                )
                is True,
            "running":
                status.get(
                    "running"
                ),
            "width":
                status.get(
                    "width"
                ),
            "height":
                status.get(
                    "height"
                ),
            "fps":
                status.get(
                    "fps"
                ),
            "color_enabled":
                status.get(
                    "color_enabled"
                ),
            "depth_enabled":
                status.get(
                    "depth_enabled"
                ),
            "fault":
                deepcopy(
                    status.get(
                        "fault"
                    )
                ),
            "source":
                "camera_service",
            "observed_at":
                observed_at,
        })

    return entities


# ============================================================
# Detection helpers
# ============================================================

def _canonical_object_entity(
    obj: dict[str, Any],
    *,
    observed_at: str,
) -> dict[str, Any] | None:
    object_id = str(
        obj.get(
            "object_id"
        )
        or ""
    ).strip()

    if not object_id:
        return None

    class_name = str(
        obj.get(
            "class_name"
        )
        or "object"
    ).strip()

    entity = {
        "entity_id":
            object_id,
        "type":
            "object",
        "class_name":
            class_name,
        "tags": [
            "perceived_object",
            class_name,
        ],
        "exists":
            True,
        "camera_source":
            obj.get(
                "camera_source"
            ),
        "confidence":
            obj.get(
                "confidence"
            ),
        "yaw_deg":
            obj.get(
                "yaw_deg"
            ),
        "condition":
            deepcopy(
                obj.get(
                    "condition"
                )
            ),
        "visual_description":
            obj.get(
                "visual_description"
            ),
        "source":
            "vision_yolo",
        "observed_at":
            observed_at,
    }

    bbox = (
        obj.get(
            "bbox"
        )
        or obj.get(
            "box"
        )
    )

    if bbox is not None:
        entity["bbox"] = deepcopy(
            bbox
        )

    xyz = obj.get(
        "robot_xyz"
    )

    if (
        isinstance(
            xyz,
            (
                list,
                tuple,
            ),
        )
        and len(
            xyz
        )
        == 3
    ):
        entity["position"] = [
            float(
                value
            )
            for value in xyz
        ]

    if isinstance(
        obj.get(
            "reachable"
        ),
        bool,
    ):
        entity["reachable"] = (
            obj[
                "reachable"
            ]
        )

    reachable_by = obj.get(
        "reachable_by"
    )

    if isinstance(
        reachable_by,
        list,
    ):
        entity["reachable_by"] = [
            str(
                value
            )
            for value
            in reachable_by
            if (
                isinstance(
                    value,
                    str,
                )
                and value
            )
        ]

    if obj.get(
        "movement"
    ) is not None:
        entity["movement"] = deepcopy(
            obj[
                "movement"
            ]
        )
    # Preserve optional grasp-specific perception outputs.
    for grasp_field in (
        "grasp_point",
        "grasp_xyz",
        "gripper_yaw_deg",
        "grasp_yaw_deg",
        "gripper_width_mm",
    ):
        if obj.get(
            grasp_field
        ) is not None:
            entity[
                grasp_field
            ] = deepcopy(
                obj[
                    grasp_field
                ]
            )

    state_tags = obj.get(
        "state_tags"
    )

    if isinstance(
        state_tags,
        dict,
    ):
        for (
            field,
            value,
        ) in state_tags.items():
            if (
                isinstance(
                    field,
                    str,
                )
                and field
            ):
                entity[field] = deepcopy(
                    value
                )

    if obj.get(
        "open_state"
    ) in {
        "open",
        "closed",
    }:
        entity["open_state"] = (
            obj[
                "open_state"
            ]
        )

    return entity


def collect_detected_entities(
    *,
    camera_names: tuple[str, ...] =
        _DETECTION_CAMERAS,
    include_robot_xyz: bool = True,
    return_observations: bool = False,
) -> (
    list[dict[str, Any]]
    | tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]
):
    """
    Run one fresh structured observation per camera.

    The same observation provides:
    - RGB for InstaOrder
    - YOLO masks/detections
    - canonical object entities
    """

    raw_objects = []
    observations = []

    for camera_name in camera_names:
        try:
            data = (
                perception_service
                .get_detections_value(
                    camera_name=
                        camera_name,
                    draw=False,
                    include_robot_xyz=
                        include_robot_xyz,
                )
            )

        except Exception as exc:
            logger.warning(
                "detection unavailable: "
                "camera=%s, error=%s: %s",
                camera_name,
                type(exc).__name__,
                exc,
            )
            continue

        if not isinstance(
            data,
            dict,
        ):
            logger.warning(
                "invalid detection data: "
                "camera=%s, data=%r",
                camera_name,
                data,
            )
            continue

        detections = data.get(
            "detections"
        )

        if not isinstance(
            detections,
            list,
        ):
            logger.warning(
                "invalid detections payload: "
                "camera=%s, detections=%r",
                camera_name,
                detections,
            )
            continue

        observation = {
            "camera_name":
                camera_name,
            "camera_rgb":
                data.get(
                    "camera_rgb"
                ),
            "timestamp":
                data.get(
                    "timestamp"
                ),
            "objects":
                [],
        }

        observations.append(
            observation
        )

        for detection in detections:
            if not isinstance(
                detection,
                dict,
            ):
                continue

            raw_objects.append({
                **deepcopy(
                    detection
                ),
                "camera_source":
                    camera_name,
            })

    # IDs are assigned once and then reused by both
    # canonical entities and relation producers.
    raw_objects = assign_object_ids(raw_objects)

    observation_by_camera = {
        row["camera_name"]:
            row
        for row in observations
        if row.get(
            "camera_name"
        )
    }

    for obj in raw_objects:
        observation = (
            observation_by_camera.get(
                obj.get(
                    "camera_source"
                )
            )
        )

        if observation is not None:
            observation[
                "objects"
            ].append(
                obj
            )

    observed_at = _utc_now_iso()

    for observation in observations:
        observation[
            "observed_at"
        ] = observed_at

    entities = []

    for obj in raw_objects:
        if not isinstance(
            obj,
            dict,
        ):
            continue

        entity = (
            _canonical_object_entity(
                obj,
                observed_at=
                    observed_at,
            )
        )

        if entity is not None:
            entities.append(
                entity
            )

    if return_observations:
        return (
            entities,
            observations,
        )

    return entities


# ============================================================
# Occlusion Relations
# ============================================================

def collect_occlusion_relations(
    *,
    observations: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    relations = []

    for observation in observations:
        if not isinstance(
            observation,
            dict,
        ):
            continue

        camera_rgb = observation.get(
            "camera_rgb"
        )

        objects = observation.get(
            "objects"
        )

        if (
            camera_rgb is None
            or not isinstance(
                objects,
                list,
            )
            or len(
                objects
            )
            < 2
        ):
            continue

        camera_name = observation.get(
            "camera_name"
        )

        try:
            response = (
                occultation
                .infer_occlusion_on_pc(
                    camera_rgb=
                        camera_rgb,
                    objects=
                        objects,
                    camera_name=
                        camera_name,
                    timestamp=
                        observation.get(
                            "observed_at"
                        ),
                    base_url=
                        _INSTAORDER_PC_URL,
                    relation_distance_px=
                        _INSTAORDER_RELATION_DISTANCE_PX,
                    confidence_threshold=
                        _INSTAORDER_CONFIDENCE_THRESHOLD,
                    exclude_classes=
                        list(
                            _INSTAORDER_EXCLUDED_CLASSES
                        ),
                    timeout_s=
                        5.0,
                    raise_on_error=
                        False,
                )
            )

        except Exception as exc:
            logger.warning(
                "occlusion inference unavailable: "
                "camera=%s, error=%s: %s",
                camera_name,
                type(exc).__name__,
                exc,
            )
            continue

        if (
            not isinstance(
                response,
                dict,
            )
            or response.get(
                "result"
            )
            is not True
        ):
            logger.warning(
                "occlusion inference unavailable: "
                "camera=%s, response=%r",
                camera_name,
                response,
            )
            continue

        rows = (
            occultation
            .relations_from_response(
                response
            )
        )

        for relation in rows:
            if (
                not isinstance(
                    relation,
                    dict,
                )
                or relation.get(
                    "predicate"
                )
                != "occludes"
            ):
                continue

            row = deepcopy(
                relation
            )

            # Normalize relation metadata at the
            # WorldState aggregation boundary.
            row["source"] = (
                "instaorder"
            )
            row["camera_source"] = (
                camera_name
            )
            row["observed_at"] = (
                observation.get(
                    "observed_at"
                )
            )

            relations.append(
                row
            )

    return relations


# ============================================================
# Material States
# ============================================================

def collect_material_states(
    *,
    observations: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    """
    Infer object materials from the SAME
    detection observations.

    Returns:

        {
            "object_id": {
                "plastic": 51.63,
                "paper": 48.14,
                "confidence": 44.55,
            }
        }
    """

    states = {}

    for observation in observations:
        if not isinstance(
            observation,
            dict,
        ):
            continue

        camera_rgb = observation.get(
            "camera_rgb"
        )

        objects = observation.get(
            "objects"
        )

        if (
            camera_rgb is None
            or not isinstance(
                objects,
                list,
            )
            or len(
                objects
            ) < 1
        ):
            continue

        camera_name = observation.get(
            "camera_name"
        )

        try:
            pc_response = (
                material
                .infer_material_on_pc(
                    camera_rgb=
                        camera_rgb,

                    objects=
                        objects,

                    camera_name=
                        camera_name,

                    timestamp=
                        observation.get(
                            "observed_at"
                        ),

                    base_url=
                        _MATERIAL_PC_URL,

                    timeout_s=
                        20.0,

                    raise_on_error=
                        False,
                )
            )

        except Exception as exc:
            logger.warning(
                "material inference unavailable: "
                "camera=%s, error=%s: %s",
                camera_name,
                type(exc).__name__,
                exc,
            )
            continue

        if (
            not isinstance(
                pc_response,
                dict,
            )
            or pc_response.get(
                "result"
            )
            is not True
        ):
            logger.warning(
                "material inference unavailable: "
                "camera=%s, response=%r",
                camera_name,
                pc_response,
            )
            continue

        rows = (
            material
            .material_from_response(
                pc_response
            )
        )

        if not isinstance(
            rows,
            list,
        ):
            continue

        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            object_id = str(
                row.get(
                    "object_id"
                )
                or ""
            ).strip()

            if not object_id:
                continue

            material_state = row.get(
                "material"
            )

            if not isinstance(
                material_state,
                dict,
            ):
                continue

            states[
                object_id
            ] = deepcopy(
                material_state
            )

    return states


# ============================================================
# Grasp Understanding
# ============================================================

def collect_grasp_states(
    *,
    entities: dict[
        str,
        dict[str, Any],
    ],
    relations: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    arm_workspaces = (
        grasp.arm_workspaces_from_config(
            getattr(
                config,
                "ARMS",
                {},
            )
        )
    )

    return grasp.infer_grasp_states(
        entities=
            entities,
        relations=
            relations,
        arm_workspaces=
            arm_workspaces,
    )

 
def _merge_grasp_states(
    *,
    entities: dict[
        str,
        dict[str, Any],
    ],
    grasp_states: dict[
        str,
        dict[str, Any],
    ],
) -> None:
    for (
        object_id,
        grasp_state,
    ) in grasp_states.items():

        entity = entities.get(
            object_id
        )

        if not isinstance(
            entity,
            dict,
        ):
            continue

        state = deepcopy(
            grasp_state
        )

        entity[
            "grasp"
        ] = state

        entity[
            "pickable"
        ] = bool(
            state.get(
                "pickable",
                False,
            )
        )

        entity[
            "pickable_by"
        ] = list(
            state.get(
                "pickable_by",
                [],
            )
        )

        if (
            not entity.get(
                "reachable_by"
            )
            and entity[
                "pickable_by"
            ]
        ):
            entity[
                "reachable_by"
            ] = list(
                entity[
                    "pickable_by"
                ]
            )

        tags = entity.get(
            "tags"
        )

        if not isinstance(
            tags,
            list,
        ):
            tags = []

        tags = [
            tag
            for tag
            in tags
            if (
                isinstance(
                    tag,
                    str,
                )
                and tag
                != "pickable"
            )
        ]

        if entity[
            "pickable"
        ]:
            tags.append(
                "pickable"
            )

        entity[
            "tags"
        ] = tags

# ============================================================
# Live WorldState
# ============================================================

def build_world_state(
    *,
    context: dict[str, Any] | None = None,
    include_robot: bool = True,
    include_cameras: bool = True,
    include_detections: bool = True,
    include_materials: bool = False,
    include_relations: bool = True,
    include_grasp: bool = True,
    camera_names: tuple[str, ...] =
        _DETECTION_CAMERAS,
) -> WorldState:
    if context is None:
        context = {}

    if not isinstance(
        context,
        dict,
    ):
        raise ValueError(
            "context 必須是 dict"
        )

    entities: dict[
        str,
        dict[str, Any],
    ] = {}

    context_entities = context.get(
        "entities"
    )

    if isinstance(
        context_entities,
        dict,
    ):
        for (
            entity_id,
            values,
        ) in context_entities.items():
            if (
                isinstance(
                    entity_id,
                    str,
                )
                and entity_id.strip()
                and isinstance(
                    values,
                    dict,
                )
            ):
                entities[
                    entity_id.strip()
                ] = deepcopy(
                    values
                )

    entity_groups = []
    detection_observations = []

    # ========================================================
    # Robot
    # ========================================================

    if include_robot:
        entity_groups.append(
            collect_robot_entities()
        )

    # ========================================================
    # Cameras
    # ========================================================

    if include_cameras:
        entity_groups.append(
            collect_camera_entities()
        )

    # ========================================================
    # Object detections
    # ========================================================

    if include_detections:
        (
            detected_entities,
            detection_observations,
        ) = collect_detected_entities(
            camera_names=
                camera_names,
            return_observations=
                True,
        )

        entity_groups.append(
            detected_entities
        )

    # ========================================================
    # Merge entities
    # ========================================================

    for rows in entity_groups:
        for entity in rows:
            entity_id = str(
                entity.get(
                    "entity_id"
                )
                or ""
            ).strip()

            if not entity_id:
                continue

            values = deepcopy(
                entity
            )

            values.pop(
                "entity_id",
                None,
            )

            entities[
                entity_id
            ] = values

    # ========================================================
    # Material states
    # ========================================================

    if (
        include_materials
        and detection_observations
    ):
        material_states = (
            collect_material_states(
                observations=
                    detection_observations,
            )
        )

        for (
            object_id,
            material_state,
        ) in material_states.items():

            entity = entities.get(
                object_id
            )

            if not isinstance(
                entity,
                dict,
            ):
                continue

            entity[
                "material"
            ] = deepcopy(
                material_state
            )

    # ========================================================
    # Relations
    #
    # Semantic VLM relation inference has been removed.
    # WorldState relations currently come only from
    # the dedicated occlusion pipeline (InstaOrder).
    # ========================================================

    observed_relations = []

    if (
        include_relations
        or include_grasp
    ):
        observed_relations = (
            collect_occlusion_relations(
                observations=
                    detection_observations,
            )
        )

    # ========================================================
    # Grasp understanding
    # ========================================================

    if include_grasp:
        grasp_states = (
            collect_grasp_states(
                entities=
                    entities,
                relations=
                    observed_relations,
            )
        )

        _merge_grasp_states(
            entities=
                entities,
            grasp_states=
                grasp_states,
        )

    relations = (
        observed_relations
        if include_relations
        else []
    )

    # ========================================================
    # WorldState
    # ========================================================

    return WorldState({
        "entities":
            entities,

        "relations":
            relations,

        "meta": {
            "captured_at":
                _utc_now_iso(),

            "source":
                "services.vlm_narrator_service",
        },
    })
# ============================================================
# Compact Object State API
# ============================================================
def get_object_state(
    *,
    camera_names: tuple[str, ...] =
        _DETECTION_CAMERAS,
):
    """
    Perform one fresh object observation and return
    compact intrinsic object states.

    Output order:
    - object
    - position
    - bbox
    - orientation
    - confidence
    - material

    This API does NOT include:
    - relations
    - handling
    - grasp
    - obstacle state
    - task state
    """

    world = build_world_state(
        include_robot=False,
        include_cameras=False,
        include_detections=True,
        include_materials=True,
        include_relations=False,
        include_grasp=False,
        camera_names=camera_names,
    )

    snapshot = world.snapshot()

    entities = snapshot.get(
        "entities",
        {},
    )

    objects = []

    for (
        entity_id,
        entity,
    ) in entities.items():

        if (
            not isinstance(
                entity,
                dict,
            )
            or entity.get(
                "type"
            )
            != "object"
        ):
            continue

        objects.append({
            "object":
                entity_id,

            "position": {
                "frame":
                    "robot",

                "xyz":
                    deepcopy(
                        entity.get(
                            "position"
                        )
                    ),
            },

            "bbox":
                deepcopy(
                    entity.get(
                        "bbox"
                    )
                ),

            "orientation": {
                "yaw_deg":
                    entity.get(
                        "yaw_deg"
                    ),
            },

            "confidence":
                entity.get(
                    "confidence"
                ),

            "material":
                deepcopy(
                    entity.get(
                        "material"
                    )
                ),
        })

    return response.success(
        MODULE,
        "get_object_state",
        data={
            "objects":
                objects,

            "captured_at":
                snapshot.get(
                    "meta",
                    {},
                ).get(
                    "captured_at"
                ),
        },
    )


# ============================================================
# Compact Object Relation API
# ============================================================

def get_object_relation(
    *,
    camera_names: tuple[str, ...] =
        _DETECTION_CAMERAS,
):


    world = build_world_state(
        include_robot=False,
        include_cameras=False,
        include_detections=True,
        include_materials=False,
        include_relations=True,
        include_grasp=False,
        camera_names=camera_names,
    )

    snapshot = world.snapshot()

    entities = snapshot.get(
        "entities",
        {},
    )

    relations = snapshot.get(
        "relations",
        [],
    )

    objects = {}

    # ========================================================
    # Create object buckets
    # ========================================================

    for entity_id, entity in entities.items():
        if (
            not isinstance(
                entity,
                dict,
            )
            or entity.get(
                "type"
            )
            != "object"
        ):
            continue

        objects[entity_id] = {
            "object":
                entity_id,

            "relations":
                [],
        }

    # ========================================================
    # Organize relations by object
    # ========================================================

    for relation in relations:
        if not isinstance(
            relation,
            dict,
        ):
            continue

        subject = str(
            relation.get(
                "subject"
            )
            or ""
        ).strip()

        predicate = str(
            relation.get(
                "predicate"
            )
            or ""
        ).strip()

        relation_object = str(
            relation.get(
                "object"
            )
            or ""
        ).strip()

        camera_source = relation.get(
            "camera_source"
        )

        if not (
            subject
            and predicate
            and relation_object
        ):
            continue

        # ----------------------------------------------------
        # A --predicate--> B
        # ----------------------------------------------------

        if subject in objects:
            objects[
                subject
            ][
                "relations"
            ].append({
                "direction":
                    "to",

                "predicate":
                    predicate,

                "relation_object":
                    relation_object,

                "camera_source":
                    camera_source,
            })

        # ----------------------------------------------------
        # B <--predicate-- A
        # ----------------------------------------------------

        if relation_object in objects:
            objects[
                relation_object
            ][
                "relations"
            ].append({
                "direction":
                    "from",

                "predicate":
                    predicate,

                "relation_object":
                    subject,

                "camera_source":
                    camera_source,
            })

    return response.success(
        MODULE,
        "get_object_relation",
        data={
            "objects":
                list(
                    objects.values()
                ),

            "captured_at":
                snapshot.get(
                    "meta",
                    {},
                ).get(
                    "captured_at"
                ),
        },
    )
# ============================================================
# Compact Object Grasp API
# ============================================================

def get_object_grasp(
    *,
    camera_names: tuple[str, ...] =
        _DETECTION_CAMERAS,
):
    world = build_world_state(
        include_robot=True,
        include_cameras=False,
        include_detections=True,
        include_materials=False,
        include_relations=False,
        include_grasp=True,
        camera_names=camera_names,
    )

    snapshot = world.snapshot()

    entities = snapshot.get(
        "entities",
        {},
    )

    objects = []

    for (
        entity_id,
        entity,
    ) in entities.items():

        if (
            not isinstance(
                entity,
                dict,
            )
            or entity.get(
                "type"
            )
            != "object"
        ):
            continue

        grasp_state = entity.get(
            "grasp"
        )

        if not isinstance(
            grasp_state,
            dict,
        ):
            continue

        objects.append({
            "object":
                entity_id,

            "pickable":
                bool(
                    grasp_state.get(
                        "pickable",
                        False,
                    )
                ),

            "pickable_by":
                deepcopy(
                    grasp_state.get(
                        "pickable_by",
                        [],
                    )
                ),

            "grasp_point":
                deepcopy(
                    grasp_state.get(
                        "grasp_point"
                    )
                ),

            "gripper_yaw_deg":
                grasp_state.get(
                    "gripper_yaw_deg"
                ),

            "gripper_width_mm":
                grasp_state.get(
                    "gripper_width_mm"
                ),

            "blocked":
                bool(
                    grasp_state.get(
                        "blocked",
                        False,
                    )
                ),

            "blocked_by":
                deepcopy(
                    grasp_state.get(
                        "blocked_by",
                        [],
                    )
                ),

            "reasons":
                deepcopy(
                    grasp_state.get(
                        "reasons",
                        [],
                    )
                ),
        })

    return response.success(
        MODULE,
        "get_object_grasp",
        data={
            "objects":
                objects,

            "captured_at":
                snapshot.get(
                    "meta",
                    {},
                ).get(
                    "captured_at"
                ),
        },
    )