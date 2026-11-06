from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2


DEFAULT_PC_URL = "http://192.168.50.37:8021"

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


def infer_occlusion_on_pc(
    *,
    camera_rgb,
    objects: list[dict[str, Any]],
    camera_name: str | None = None,
    timestamp: str | None = None,
    base_url: str = DEFAULT_PC_URL,
    relation_distance_px: int = (
        DEFAULT_RELATION_DISTANCE_PX
    ),
    confidence_threshold: float = (
        DEFAULT_CONFIDENCE_THRESHOLD
    ),
    exclude_classes: list[str] | None = None,
    jpeg_quality: int = 90,
    timeout_s: float = 5.0,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """
    Send one same-frame RGB + YOLO-Seg
    observation to the PC InstaOrder service.
    """

    if camera_rgb is None:
        raise ValueError(
            "camera_rgb is required"
        )

    public_objects = _public_objects(
        objects
    )
    print("\n[OCCLUSION PUBLIC OBJECTS]")

    for obj in public_objects:
        print(
            obj.get("object_id"),
            "class=", obj.get("class_name"),
            "seg_points=",
            len(
                obj.get(
                    "segmentation_points"
                )
                or []
            ),
            "box=",
            obj.get("box"),
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

        "relation_distance_px":
            int(
                relation_distance_px
            ),

        "confidence_threshold":
            float(
                confidence_threshold
            ),

        "exclude_classes":
            list(
                exclude_classes
                if exclude_classes
                is not None
                else [
                    "chair_surface",
                ]
            ),
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
            + "/infer_occlusion"
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
        print("\n[OCCLUSION RAW RESPONSE]"        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
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
                "PC occlusion service error: "
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
                "relations": [],
            },
        }


def relations_from_response(
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract WorldState-ready relations
    from PC response.
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

    relations = data.get(
        "relations"
    )

    if not isinstance(
        relations,
        list,
    ):
        return []

    return [
        dict(
            relation
        )
        for relation in relations
        if isinstance(
            relation,
            dict,
        )
    ]