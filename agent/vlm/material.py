from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2


DEFAULT_PC_URL = "http://192.168.50.37:8022"

DEFAULT_RELATION_DISTANCE_PX = 30
DEFAULT_CONFIDENCE_THRESHOLD = 0.55


def _encode_jpeg_b64(
    image,
    quality: int = 90,
) -> str:
    """
    Encode OpenCV BGR image into base64 JPEG.
    """

    quality = max(
        1,
        min(
            100,
            int(quality),
        ),
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            quality,
        ],
    )

    if not ok:
        raise RuntimeError(
            "failed to encode JPEG"
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode(
        "ascii"
    )


def _public_objects(
    objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep only fields required by the
    PC InstaOrder service.
    """

    out = []

    for obj in objects:
        if not isinstance(
            obj,
            dict,
        ):
            continue

        points = obj.get(
            "segmentation_points"
        ) or []

        if len(points) < 3:
            continue

        object_id = (
            obj.get(
                "object_id"
            )
            or obj.get(
                "entity_id"
            )
            or obj.get(
                "track_id"
            )
            or obj.get(
                "id"
            )
        )

        if not object_id:
            continue

        out.append({
            "object_id":
                str(
                    object_id
                ),

            "class_name":
                obj.get(
                    "class_name"
                ),

            "confidence":
                obj.get(
                    "confidence"
                ),

            "box":
                obj.get(
                    "box"
                )
                or obj.get(
                    "bbox"
                ),

            "segmentation_points":
                points,
        })

    return out


def infer_material_on_pc(
    *,
    camera_rgb,
    objects: list[dict[str, Any]],
    camera_name: str | None = None,
    timestamp: str | None = None,
    base_url: str = DEFAULT_PC_URL,
    jpeg_quality: int = 90,
    timeout_s: float = 20.0,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """
    Send one same-frame RGB + object masks
    to the PC MatSpectNet service.

    The PC service should:
    - run MatSpectNet once on the RGB frame
    - reconstruct each object mask
    - calculate material distribution inside each mask
    """

    if camera_rgb is None:
        raise ValueError(
            "camera_rgb is required"
        )

    public_objects = _public_objects(
        objects
    )

    payload = {
        "camera_name":
            camera_name,

        "timestamp":
            timestamp,

        "image_jpeg_b64":
            _encode_jpeg_b64(
                camera_rgb,
                quality=
                    jpeg_quality,
            ),

        "objects":
            public_objects,
    }

    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    request = Request(
        (
            base_url.rstrip("/")
            + "/infer_material"
        ),
        data=body,
        headers={
            "Content-Type":
                "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=float(
                timeout_s
            ),
        ) as response:
            raw = response.read()

        result = json.loads(
            raw.decode(
                "utf-8"
            )
        )

        if (
            raise_on_error
            and (
                not isinstance(
                    result,
                    dict,
                )
                or result.get(
                    "result"
                )
                is not True
            )
        ):
            raise RuntimeError(
                "PC material service error: "
                f"{result}"
            )

        return result

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:

        if raise_on_error:
            raise

        return {
            "status":
                "error",

            "result":
                False,

            "message":
                str(
                    exc
                ),

            "error_type":
                type(
                    exc
                ).__name__,

            "data": {
                "materials": [],
            },
        }


def material_from_response(
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert PC MatSpectNet response into
    compact WorldState-ready material states.

    Output:

    [
        {
            "object_id": "cup_1",
            "material": {
                "ceramic": 91.0,
                "food": 6.0,
                "plastic": 3.0,
                "confidence": 84.0,
            }
        }
    ]

    All values are percentages.
    """

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

    materials = data.get(
        "materials"
    )

    if not isinstance(
        materials,
        list,
    ):
        return []

    result = []

    for row in materials:
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

        distribution = row.get(
            "materials"
        )

        if not isinstance(
            distribution,
            dict,
        ):
            distribution = {}

        compact_material = {}

        # ====================================================
        # Material distribution
        # ====================================================

        for (
            material_name,
            ratio,
        ) in distribution.items():

            try:
                percentage = (
                    float(
                        ratio
                    )
                    * 100.0
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            compact_material[
                str(
                    material_name
                )
            ] = round(
                percentage,
                2,
            )

        # ====================================================
        # Overall MatSpectNet confidence
        # ====================================================

        try:
            confidence = float(
                row.get(
                    "mean_pixel_confidence"
                )
            )

            compact_material[
                "confidence"
            ] = round(
                confidence
                * 100.0,
                2,
            )

        except (
            TypeError,
            ValueError,
        ):
            compact_material[
                "confidence"
            ] = None

        result.append({
            "object_id":
                object_id,

            "material":
                compact_material,
        })

    return result