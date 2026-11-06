"""
Grasp-understanding helpers for the World Model.

This module contains CURRENT physical grasp affordance logic only.

MVP rule:

    detected object
    + valid grasp point
    + grasp point inside at least one arm workspace
    + object is not currently occluded
    ------------------------------------------------
    => pickable
Notes:
-  A later version can check whether the grasp region itself is occluded instead.
- Workspace checks use configured arm safety ranges as an MVP gate.

"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


_NON_PICKABLE_CLASSES = {
    "chair_surface",
}


def _finite_xyz(
    value: Any,
) -> list[float] | None:
    if isinstance(
        value,
        dict,
    ):
        value = value.get(
            "xyz"
        )

    if (
        not isinstance(
            value,
            (list, tuple),
        )
        or len(value) != 3
    ):
        return None

    try:
        xyz = [
            float(item)
            for item in value
        ]
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not all(
        math.isfinite(item)
        for item in xyz
    ):
        return None

    return xyz


def extract_grasp_point(
    entity: dict[str, Any],
) -> tuple[
    list[float] | None,
    str | None,
]:
    """
    Return the best available grasp point in robot coordinates.

    Priority:
    1. entity["grasp"][...]
    2. explicit grasp_point / grasp_xyz
    3. canonical object position
    """

    grasp_state = entity.get(
        "grasp"
    )

    if isinstance(
        grasp_state,
        dict,
    ):
        for key in (
            "point",
            "grasp_point",
            "xyz",
        ):
            xyz = _finite_xyz(
                grasp_state.get(
                    key
                )
            )

            if xyz is not None:
                return (
                    xyz,
                    f"grasp.{key}",
                )

    for key in (
        "grasp_point",
        "grasp_xyz",
        "position",
    ):
        xyz = _finite_xyz(
            entity.get(
                key
            )
        )

        if xyz is not None:
            return (
                xyz,
                key,
            )

    return (
        None,
        None,
    )


def extract_gripper_yaw_deg(
    entity: dict[str, Any],
) -> float | None:
    """
    Return the best available gripper yaw.

    Prefer grasp-specific yaw when available, otherwise fall back to
    the object's canonical yaw.
    """

    grasp_state = entity.get(
        "grasp"
    )

    candidates = []

    if isinstance(
        grasp_state,
        dict,
    ):
        candidates.extend([
            grasp_state.get(
                "gripper_yaw_deg"
            ),
            grasp_state.get(
                "yaw_deg"
            ),
        ])

    candidates.extend([
        entity.get(
            "gripper_yaw_deg"
        ),
        entity.get(
            "grasp_yaw_deg"
        ),
        entity.get(
            "yaw_deg"
        ),
    ])

    for value in candidates:
        if value is None:
            continue

        try:
            number = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if math.isfinite(
            number
        ):
            return number

    return None


def arm_workspaces_from_config(
    arms_config: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    dict[str, tuple[float, float]],
]:
    """
    Convert config.ARMS[*].safety x/y/z ranges into workspace bounds.

    Returned arm IDs match WorldState entity IDs:
        left  -> left_arm
        right -> right_arm
    """

    workspaces: dict[
        str,
        dict[str, tuple[float, float]],
    ] = {}

    if not isinstance(
        arms_config,
        dict,
    ):
        return workspaces

    for (
        arm_name,
        arm_config,
    ) in arms_config.items():

        if not isinstance(
            arm_config,
            dict,
        ):
            continue

        safety = arm_config.get(
            "safety"
        )

        if not isinstance(
            safety,
            dict,
        ):
            continue

        ranges: dict[
            str,
            tuple[float, float],
        ] = {}

        valid = True

        for axis in (
            "x",
            "y",
            "z",
        ):
            raw_range = safety.get(
                f"{axis}_range"
            )

            if (
                not isinstance(
                    raw_range,
                    (list, tuple),
                )
                or len(raw_range) != 2
            ):
                valid = False
                break

            try:
                low = float(
                    raw_range[0]
                )
                high = float(
                    raw_range[1]
                )
            except (
                TypeError,
                ValueError,
            ):
                valid = False
                break

            if (
                not math.isfinite(
                    low
                )
                or not math.isfinite(
                    high
                )
                or low > high
            ):
                valid = False
                break

            ranges[
                axis
            ] = (
                low,
                high,
            )

        if not valid:
            continue

        arm_id = (
            f"{arm_name}_arm"
            if arm_name
            in {
                "left",
                "right",
            }
            else str(
                arm_name
            )
        )

        workspaces[
            arm_id
        ] = ranges

    return workspaces


def _point_in_workspace(
    xyz: list[float],
    workspace: dict[
        str,
        tuple[float, float],
    ],
) -> bool:
    for (
        index,
        axis,
    ) in enumerate(
        (
            "x",
            "y",
            "z",
        )
    ):
        axis_range = workspace.get(
            axis
        )

        if axis_range is None:
            return False

        if not (
            axis_range[0]
            <= xyz[index]
            <= axis_range[1]
        ):
            return False

    return True


def _blocked_by(
    object_id: str,
    relations: list[
        dict[str, Any]
    ],
) -> list[str]:
    """
    For relation:

        A --occludes--> B

    B is treated as blocked by A in the current MVP.
    """

    blockers = []

    for relation in relations:
        if not isinstance(
            relation,
            dict,
        ):
            continue

        if relation.get(
            "predicate"
        ) != "occludes":
            continue

        target = str(
            relation.get(
                "object"
            )
            or ""
        ).strip()

        if target != object_id:
            continue

        subject = str(
            relation.get(
                "subject"
            )
            or ""
        ).strip()

        if (
            subject
            and subject not in blockers
        ):
            blockers.append(
                subject
            )

    return blockers


def infer_object_grasp(
    object_id: str,
    entity: dict[str, Any],
    *,
    relations: list[
        dict[str, Any]
    ],
    arm_workspaces: dict[
        str,
        dict[str, tuple[float, float]],
    ],
    all_entities: dict[
        str,
        dict[str, Any],
    ] | None = None,
) -> dict[str, Any]:
    """
    Infer the CURRENT grasp affordance for one object.
    """

    xyz, point_source = (
        extract_grasp_point(
            entity
        )
    )

    blocked_by = _blocked_by(
        object_id,
        relations,
    )

    blocked = bool(
        blocked_by
    )

    class_name = str(
        entity.get(
            "class_name"
        )
        or ""
    ).strip().lower()

    class_pickable = (
        class_name
        not in _NON_PICKABLE_CLASSES
    )

    pickable_by: list[str] = []

    explicit_reachable_by = (
        entity.get(
            "reachable_by"
        )
    )

    if (
        isinstance(
            explicit_reachable_by,
            list,
        )
        and explicit_reachable_by
    ):
        candidate_arms = [
            str(item).strip()
            for item
            in explicit_reachable_by
            if (
                isinstance(
                    item,
                    str,
                )
                and item.strip()
            )
        ]
    else:
        candidate_arms = list(
            arm_workspaces.keys()
        )

    if xyz is not None:
        for arm_id in candidate_arms:
            workspace = (
                arm_workspaces.get(
                    arm_id
                )
            )

            if workspace is None:
                reachable = (
                    isinstance(
                        explicit_reachable_by,
                        list,
                    )
                    and arm_id
                    in explicit_reachable_by
                )
            else:
                reachable = (
                    _point_in_workspace(
                        xyz,
                        workspace,
                    )
                )

            if not reachable:
                continue

            if isinstance(
                all_entities,
                dict,
            ):
                arm_entity = (
                    all_entities.get(
                        arm_id
                    )
                )

                if (
                    isinstance(
                        arm_entity,
                        dict,
                    )
                    and arm_entity.get(
                        "connected"
                    )
                    is False
                ):
                    continue

            pickable_by.append(
                arm_id
            )

    detected = (
        entity.get(
            "type"
        )
        == "object"
        and entity.get(
            "exists",
            True,
        )
        is not False
    )

    pickable = bool(
        detected
        and class_pickable
        and xyz is not None
        and pickable_by
        and not blocked
    )

    reasons = []

    if not detected:
        reasons.append(
            "not_detected"
        )

    if not class_pickable:
        reasons.append(
            "non_pickable_class"
        )

    if xyz is None:
        reasons.append(
            "missing_grasp_point"
        )

    if (
        xyz is not None
        and not pickable_by
    ):
        reasons.append(
            "outside_reachable_workspace"
        )

    if blocked:
        reasons.append(
            "occluded"
        )

    grasp_state: dict[
        str,
        Any,
    ] = {
        "pickable":
            pickable,

        "pickable_by":
            pickable_by,

        "grasp_point": {
            "frame":
                "robot",
            "xyz":
                deepcopy(
                    xyz
                ),
            "source":
                point_source,
        },

        "gripper_yaw_deg":
            extract_gripper_yaw_deg(
                entity
            ),

        "blocked":
            blocked,

        "blocked_by":
            blocked_by,

        "reasons":
            reasons,
    }

    width = entity.get(
        "gripper_width_mm"
    )

    if width is not None:
        try:
            width = float(
                width
            )
        except (
            TypeError,
            ValueError,
        ):
            width = None

    if (
        isinstance(
            width,
            float,
        )
        and math.isfinite(
            width
        )
    ):
        grasp_state[
            "gripper_width_mm"
        ] = width

    return grasp_state


def infer_grasp_states(
    *,
    entities: dict[
        str,
        dict[str, Any],
    ],
    relations: list[
        dict[str, Any]
    ],
    arm_workspaces: dict[
        str,
        dict[str, tuple[float, float]],
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    """
    Infer grasp state for every currently perceived object.
    """

    result: dict[
        str,
        dict[str, Any],
    ] = {}

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

        result[
            entity_id
        ] = infer_object_grasp(
            entity_id,
            entity,
            relations=
                relations,
            arm_workspaces=
                arm_workspaces,
            all_entities=
                entities,
        )

    return result


__all__ = [
    "extract_grasp_point",
    "extract_gripper_yaw_deg",
    "arm_workspaces_from_config",
    "infer_object_grasp",
    "infer_grasp_states",
]
