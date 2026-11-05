"""
Runtime scene / observation configuration.

This module contains site-specific scene knowledge:
- semantic observation regions
- concrete robot observation poses
- camera selection
- observation backend selection
- bootstrap scene entities
- static semantic overlays

Generic runtime modules must not contain these concrete values.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# ============================================================
# Region IDs
# ============================================================

PREPARATION_REGION_ID = (
    "preparation_area_1"
)

TRASH_REGION_ID = (
    "trash_area_1"
)


# ============================================================
# Observation poses
# ============================================================

SCENE_OBSERVE_JOINTS = [
    3.395045280456543,
    -2.0729740301715296,
    1.5954980850219727,
    -1.5963285605060022,
    -2.2232630888568323,
    -0.8404019514666956,
]


OBJECT_OBSERVE_JOINTS = [
    4.795281887054443,
    -1.7464926878558558,
    1.486818790435791,
    -1.7543662230121058,
    -2.019101921712057,
    -0.9130390326129358,
]


# ============================================================
# Observation regions
# ============================================================

SCENE_REGIONS: dict[
    str,
    dict[str, Any],
] = {
    PREPARATION_REGION_ID: {
        "semantics": [
            "preparation_area",
            "temporary_placement_area",
        ],

        "device_id":
            "right_arm",

        "camera_names": [
            "right",
        ],

        "joints":
            OBJECT_OBSERVE_JOINTS,

        "observer":
            "world_state",

        "speed":
            0.20,

        "acceleration":
            0.20,

        "settle_seconds":
            0.6,

        # Static site semantics only.
        # Runtime logic does not know what a "chair_surface" is.
        "entity_rules": [
            {
                "match": {
                    "class_name":
                        "chair_surface",
                },

                "tags": [
                    "placement_target",
                    "temporary_placement",
                ],

                "affordances": [
                    "placement_target",
                    "temporary_placement",
                ],
            },
        ],
    },

    TRASH_REGION_ID: {
        "semantics": [
            "trash_area",
            "disposal_area",
        ],

        "device_id":
            "right_arm",

        "camera_names": [
            "right",
        ],

        "joints":
            SCENE_OBSERVE_JOINTS,

        "observer":
            "describe_scene",

        "vlm_backend":
            "remote",

        "speed":
            0.20,

        "acceleration":
            0.20,

        "settle_seconds":
            0.6,

        # Scene-only objects that structured object detection does not create.
        # The VLM observation is matched to this canonical WorldState entity.
        "bootstrap_entities": [
            {
                "entity_id":
                    "trash_can_1",

                "type":
                    "container",

                "class_name":
                    "trash_can",

                "name_match": [
                    "垃圾桶",
                    "有蓋容器",
                ],

                "tags": [
                    "openable",
                    "placement_target",
                ],
            },
        ],
    },
}


# Bootstrap order intentionally ends at the preparation/object view.
BOOTSTRAP_REGION_ORDER = (
    TRASH_REGION_ID,
    PREPARATION_REGION_ID,
)


# ============================================================
# Access helpers
# ============================================================

def has_scene_region(
    region_id: str,
) -> bool:
    return (
        isinstance(
            region_id,
            str,
        )
        and region_id
        in SCENE_REGIONS
    )


def get_scene_region(
    region_id: str,
) -> dict[str, Any]:
    if not isinstance(
        region_id,
        str,
    ) or not region_id:
        raise ValueError(
            "region_id must be a non-empty string"
        )

    config = (
        SCENE_REGIONS.get(
            region_id
        )
    )

    if config is None:
        raise KeyError(
            "Unknown scene region: "
            f"{region_id!r}"
        )

    return deepcopy(
        config
    )


def scene_region_ids() -> tuple[str, ...]:
    return tuple(
        SCENE_REGIONS.keys()
    )


def bootstrap_scene_region_ids() -> tuple[str, ...]:
    return tuple(
        BOOTSTRAP_REGION_ORDER
    )


__all__ = [
    "PREPARATION_REGION_ID",
    "TRASH_REGION_ID",
    "SCENE_OBSERVE_JOINTS",
    "OBJECT_OBSERVE_JOINTS",
    "SCENE_REGIONS",
    "BOOTSTRAP_REGION_ORDER",
    "has_scene_region",
    "get_scene_region",
    "scene_region_ids",
    "bootstrap_scene_region_ids",
]
