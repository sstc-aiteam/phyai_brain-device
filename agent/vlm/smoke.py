from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.request import (
    Request,
    urlopen,
)

import cv2


DEFAULT_PC_URL = (
    "http://192.168.50.37:8023"
)

DEFAULT_SOURCE_FPS = 15.0


# ============================================================
# Image
# ============================================================

def _encode_jpeg_b64(
    image,
    quality: int = 85,
) -> str:
    """
    OpenCV BGR image -> base64 JPEG.
    """

    if image is None:
        raise ValueError(
            "camera_rgb is required"
        )

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

    return (
        base64
        .b64encode(
            encoded.tobytes()
        )
        .decode("ascii")
    )


# ============================================================
# HTTP helper
# ============================================================

def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:

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
        url,
        data=body,
        headers={
            "Content-Type":
                "application/json",
        },
        method="POST",
    )

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

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "invalid smoke service response"
        )

    return result


# ============================================================
# Smoke inference
# ============================================================

def infer_smoke_on_pc(
    *,
    camera_rgb,
    camera_name: str,
    timestamp=None,
    source_fps: float =
        DEFAULT_SOURCE_FPS,
    base_url: str =
        DEFAULT_PC_URL,
    jpeg_quality: int = 85,
    timeout_s: float = 10.0,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """
    Send one RGB frame to the remote smoke detection service.

    The PC owns:
    - YOLO
    - temporal state
    - PRE-ALERT logic
    - Qwen verification

    Robot only sends frames.
    """

    if camera_rgb is None:
        raise ValueError(
            "camera_rgb is required"
        )

    if (
        not isinstance(
            camera_name,
            str,
        )
        or not camera_name.strip()
    ):
        raise ValueError(
            "camera_name is required"
        )

    try:
        source_fps = float(
            source_fps
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "source_fps must be numeric"
        ) from exc

    if source_fps <= 0:
        raise ValueError(
            "source_fps must be > 0"
        )

    payload = {
        "camera_name":
            camera_name.strip(),

        "timestamp":
            timestamp,

        "source_fps":
            source_fps,

        "image_jpeg_b64":
            _encode_jpeg_b64(
                camera_rgb,
                quality=
                    jpeg_quality,
            ),
    }

    url = (
        base_url.rstrip("/")
        + "/infer_smoke_frame"
    )

    try:
        result = _post_json(
            url=url,
            payload=payload,
            timeout_s=
                timeout_s,
        )

        if (
            raise_on_error
            and (
                result.get(
                    "result"
                )
                is not True
            )
        ):
            raise RuntimeError(
                "PC smoke service error: "
                f"{result}"
            )

        return result

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:

        if raise_on_error:
            raise

        return {
            "status":
                "error",

            "result":
                False,

            "message":
                str(exc),

            "error_type":
                type(
                    exc
                ).__name__,

            "data": {
                "state":
                    "unavailable",

                "camera_name":
                    camera_name,
            },
        }


# ============================================================
# Reset remote camera state
# ============================================================

def reset_smoke_state(
    *,
    camera_name: str,
    base_url: str =
        DEFAULT_PC_URL,
    timeout_s: float = 5.0,
    raise_on_error: bool = False,
) -> dict[str, Any]:

    payload = {
        "camera_name":
            camera_name,
    }

    try:
        return _post_json(
            url=(
                base_url.rstrip("/")
                + "/reset"
            ),
            payload=payload,
            timeout_s=
                timeout_s,
        )

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:

        if raise_on_error:
            raise

        return {
            "status":
                "error",

            "result":
                False,

            "message":
                str(exc),

            "error_type":
                type(
                    exc
                ).__name__,
        }