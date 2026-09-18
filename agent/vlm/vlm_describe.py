from __future__ import annotations

import base64
import io
import json
import time
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from config import OLLAMA_VLM_MODEL
from services import camera_service, llm_service, perception_service
from utils.response import error, success
from agent.vlm import remote_vlm
from agent.vlm.vision_skill_runner import run_vision_skill


MODULE = "vlm"

_MAX_IMAGE_WIDTH = 640
_JPEG_QUALITY = 88
_MAX_TOKENS = 1024
_TIMEOUT_SEC = 60.0
_KEEP_ALIVE = -1

_MAX_SCENE_ELEMENTS = 12
_MAX_SCENE_SUMMARY_LENGTH = 120
_MAX_NAME_LENGTH = 30
_MAX_LOCATION_LENGTH = 40
_MAX_ATTRIBUTE_LENGTH = 20

_SKILLS_DIR = (
    Path(__file__).resolve().parent.parent
    / "skills"
)


# ============================================================
# Scene Contract
# ============================================================

_SCENE_TYPES = {
    "hospital",
    "office",
    "logistics",
    "warehouse",
    "inspection",
}


_OPEN_STATES = {
    "open",
    "closed",
    "partially_open",
    "unknown",
    "not_applicable",
}


_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_type": {
            "type": "string",
            "enum": [
                "hospital",
                "office",
                "logistics",
                "warehouse",
                "inspection",
            ],
        },

        "scene_summary": {
            "type": "string",
            "maxLength":
                _MAX_SCENE_SUMMARY_LENGTH,
        },

        "scene_elements": {
            "type": "array",
            "maxItems":
                _MAX_SCENE_ELEMENTS,

            "items": {
                "type": "object",

                "properties": {
                    "name": {
                        "type": "string",
                        "maxLength":
                            _MAX_NAME_LENGTH,
                    },

                    "count": {
                        "type": "integer",
                        "minimum": 1,
                    },

                    "location": {
                        "type": "string",
                        "maxLength":
                            _MAX_LOCATION_LENGTH,
                    },

                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "maxLength":
                                _MAX_ATTRIBUTE_LENGTH,
                        },
                        "maxItems": 3,
                    },

                    "open_state": {
                        "type": "string",
                        "enum": [
                            "open",
                            "closed",
                            "partially_open",
                            "unknown",
                            "not_applicable",
                        ],
                    },
                },

                "required": [
                    "name",
                    "count",
                    "location",
                    "attributes",
                    "open_state",
                ],

                "additionalProperties":
                    False,
            },
        },
    },

    "required": [
        "scene_type",
        "scene_summary",
        "scene_elements",
    ],

    "additionalProperties":
        False,
}


# ============================================================
# Camera
# ============================================================

def _capture_rgb_jpeg(
    camera_name: str,
) -> bytes:
    """
    取得 RGB JPEG。

    若 camera 尚未啟動，嘗試啟動一次後重試。
    Camera 啟動失敗或 retry 仍失敗時，直接向上拋出錯誤。
    """

    try:
        return (
            perception_service
            .get_camera_rgb_jpeg(
                camera_name=
                    camera_name,
                quality=
                    _JPEG_QUALITY,
            )
        )

    except Exception as first_error:
        start_response = (
            camera_service
            .start_camera(
                camera_name=
                    camera_name,
            )
        )

        if (
            not isinstance(
                start_response,
                dict,
            )
            or start_response.get(
                "result"
            ) is not True
        ):
            raise RuntimeError(
                (
                    start_response.get(
                        "message"
                    )
                    if isinstance(
                        start_response,
                        dict,
                    )
                    else None
                )
                or (
                    "camera 啟動失敗："
                    f"{first_error}"
                )
            ) from first_error

        time.sleep(
            0.3
        )

        return (
            perception_service
            .get_camera_rgb_jpeg(
                camera_name=
                    camera_name,
                quality=
                    _JPEG_QUALITY,
            )
        )


# ============================================================
# Skill
# ============================================================

@lru_cache(maxsize=None)
def _load_skill(
    name: str,
) -> str:
    """
    Load a VLM Markdown skill.

    Markdown:
        VLM instructions

    Python:
        schema / validation / runtime logic
    """

    path = (
        _SKILLS_DIR
        / f"{name}.md"
    )

    if not path.is_file():
        raise RuntimeError(
            f"VLM skill 不存在：{path}"
        )

    content = (
        path.read_text(
            encoding="utf-8"
        )
        .strip()
    )

    # Optional YAML front matter
    if content.startswith("---"):
        parts = content.split(
            "---",
            2,
        )

        if len(parts) == 3:
            content = (
                parts[2]
                .strip()
            )

    if not content:
        raise RuntimeError(
            f"VLM skill 為空：{name}"
        )

    return content


# ============================================================
# Camera / Image Helpers
# ============================================================

def _normalize_camera_name(
    camera_name: Any,
) -> str:
    if camera_name is None:
        return "left"

    if not isinstance(
        camera_name,
        str,
    ):
        raise ValueError(
            "camera_name 必須是字串"
        )

    value = (
        camera_name
        .strip()
    )

    if not value:
        raise ValueError(
            "camera_name 不可為空"
        )

    return value


def _resize_jpeg(
    jpeg_bytes: bytes,
) -> bytes:
    if (
        not isinstance(
            jpeg_bytes,
            (bytes, bytearray),
        )
        or not jpeg_bytes
    ):
        raise RuntimeError(
            "camera RGB JPEG unavailable"
        )

    with Image.open(
        io.BytesIO(
            bytes(jpeg_bytes)
        )
    ) as image:
        image = (
            image.convert(
                "RGB"
            )
        )

        if (
            image.width
            > _MAX_IMAGE_WIDTH
        ):
            target_height = max(
                1,
                round(
                    image.height
                    * _MAX_IMAGE_WIDTH
                    / image.width
                ),
            )

            image = (
                image.resize(
                    (
                        _MAX_IMAGE_WIDTH,
                        target_height,
                    ),
                    Image.Resampling.LANCZOS,
                )
            )

        output = (
            io.BytesIO()
        )

        image.save(
            output,
            format="JPEG",
            quality=_JPEG_QUALITY,
        )

        return (
            output.getvalue()
        )


def _image_data_url(
    jpeg_bytes: bytes,
) -> str:
    encoded = (
        base64.b64encode(
            jpeg_bytes
        )
        .decode(
            "ascii"
        )
    )

    return (
        "data:image/jpeg;base64,"
        f"{encoded}"
    )


# ============================================================
# Structured Response Validation
# ============================================================

def _parse_scene_message(
    message: Any,
) -> dict[str, Any]:

    if not isinstance(
        message,
        dict,
    ):
        raise RuntimeError(
            "VLM response 必須是 message object"
        )

    content = (
        message.get(
            "content"
        )
    )

    if (
        not isinstance(
            content,
            str,
        )
        or not content.strip()
    ):
        raise RuntimeError(
            "VLM response 缺少 content"
        )

    try:
        payload = (
            json.loads(
                content
            )
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "VLM response 不是有效 JSON："
            f"{exc}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "VLM result root 必須是 object"
        )

    # --------------------------------------------------------
    # scene_type
    # --------------------------------------------------------

    scene_type = str(
        payload.get(
            "scene_type"
        )
        or ""
    ).strip()

    if (
        scene_type
        not in _SCENE_TYPES
    ):
        raise RuntimeError(
            "VLM result.scene_type 無效："
            f"{scene_type!r}"
        )

    # --------------------------------------------------------
    # scene_summary
    # --------------------------------------------------------

    scene_summary = (
        payload.get(
            "scene_summary"
        )
    )

    if not isinstance(
        scene_summary,
        str,
    ):
        raise RuntimeError(
            "VLM result.scene_summary "
            "必須是字串"
        )

    scene_summary = (
        scene_summary.strip()
    )

    if not scene_summary:
        raise RuntimeError(
            "VLM result.scene_summary "
            "不可為空"
        )

    if (
        len(scene_summary)
        > _MAX_SCENE_SUMMARY_LENGTH
    ):
        raise RuntimeError(
            "VLM result.scene_summary "
            f"不可超過 {_MAX_SCENE_SUMMARY_LENGTH} 字元"
        )

    # --------------------------------------------------------
    # scene_elements
    # --------------------------------------------------------

    scene_elements = (
        payload.get(
            "scene_elements"
        )
    )

    if not isinstance(
        scene_elements,
        list,
    ):
        raise RuntimeError(
            "VLM result.scene_elements 必須是 list"
        )

    if (
        len(scene_elements)
        > _MAX_SCENE_ELEMENTS
    ):
        raise RuntimeError(
            "VLM result.scene_elements "
            f"最多 {_MAX_SCENE_ELEMENTS} 項"
        )

    normalized_elements = []
    seen_elements = set()

    for index, item in enumerate(
        scene_elements
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                f"VLM result.scene_elements[{index}] "
                "必須是 object"
            )

        name = str(
            item.get(
                "name"
            )
            or ""
        ).strip()

        location = str(
            item.get(
                "location"
            )
            or ""
        ).strip()

        attributes = (
            item.get(
                "attributes"
            )
        )

        count = (
            item.get(
                "count"
            )
        )

        open_state = str(
            item.get(
                "open_state"
            )
            or ""
        ).strip()

        if not name:
            raise RuntimeError(
                f"VLM result.scene_elements[{index}]."
                "name 不可為空"
            )

        if (
            len(name)
            > _MAX_NAME_LENGTH
        ):
            raise RuntimeError(
                f"VLM result.scene_elements[{index}]."
                f"name 不可超過 {_MAX_NAME_LENGTH} 字元"
            )

        if (
            not isinstance(
                count,
                int,
            )
            or isinstance(
                count,
                bool,
            )
            or count < 1
        ):
            raise RuntimeError(
                f"VLM result.scene_elements[{index}]."
                "count 必須是正整數"
            )

        if not location:
            raise RuntimeError(
                f"VLM result.scene_elements[{index}]."
                "location 不可為空"
            )

        if (
            len(location)
            > _MAX_LOCATION_LENGTH
        ):
            raise RuntimeError(
                f"VLM result.scene_elements[{index}]."
                "location 不可超過 "
                f"{_MAX_LOCATION_LENGTH} 字元"
            )

        if not isinstance(
            attributes,
            list,
        ):
            raise RuntimeError(
                f"VLM result.scene_elements[{index}]."
                "attributes 必須是 list"
            )

        if len(
            attributes
        ) > 3:
            raise RuntimeError(
                f"VLM result.scene_elements[{index}]."
                "attributes 最多三項"
            )

        if (
            open_state
            not in _OPEN_STATES
        ):
            raise RuntimeError(
                f"VLM result.scene_elements[{index}]."
                "open_state 無效："
                f"{open_state!r}"
            )

        normalized_attributes = []

        for (
            attr_index,
            attribute,
        ) in enumerate(
            attributes
        ):
            if not isinstance(
                attribute,
                str,
            ):
                raise RuntimeError(
                    f"VLM result.scene_elements[{index}]."
                    f"attributes[{attr_index}] "
                    "必須是字串"
                )

            attribute = (
                attribute.strip()
            )

            if (
                len(attribute)
                > _MAX_ATTRIBUTE_LENGTH
            ):
                raise RuntimeError(
                    f"VLM result.scene_elements[{index}]."
                    f"attributes[{attr_index}] "
                    "不可超過 "
                    f"{_MAX_ATTRIBUTE_LENGTH} 字元"
                )

            if attribute:
                normalized_attributes.append(
                    attribute
                )

        duplicate_key = (
            name,
            count,
            location,
            tuple(
                normalized_attributes
            ),
            open_state,
        )

        if duplicate_key in seen_elements:
            continue

        seen_elements.add(
            duplicate_key
        )

        normalized_elements.append({
            "name":
                name,

            "count":
                count,

            "location":
                location,

            "attributes":
                normalized_attributes,

            "open_state":
                open_state,
        })

    return {
        "scene_type":
            scene_type,

        "scene_summary":
            scene_summary,

        "scene_elements":
            normalized_elements,
    }


# ============================================================
# Public Capability
# ============================================================

def describe_scene(
    *,
    camera_name: Any = None,
    vlm_backend: Any = None,
) -> dict[str, Any]:
    """
    Capture exactly one RGB frame and describe the scene itself.

    Output focuses on:
    - scene type
    - furniture / fixtures / large environment structures
    - visible open / closed state
    - concise scene-level summary

    Small task objects are intentionally not the focus.
    No YOLO and no robot action are executed.
    """

    action = (
        "describe_scene"
    )

    started = (
        perf_counter()
    )

    try:
        camera_name = (
            _normalize_camera_name(
                camera_name
            )
        )

        # Only the explicit string "remote" uses the PC VLM.
        # None / "local" / any other value stays on the Edge.
        use_remote = (
            isinstance(
                vlm_backend,
                str,
            )
            and vlm_backend.strip().lower()
            == "remote"
        )

        backend_name = (
            "remote"
            if use_remote
            else "local"
        )

        model_name = (
            remote_vlm.REMOTE_VLM_MODEL
            if use_remote
            else OLLAMA_VLM_MODEL
        )

        # ----------------------------------------------------
        # RGB Capture
        # ----------------------------------------------------

        capture_started = (
            perf_counter()
        )

        jpeg_bytes = (
            _capture_rgb_jpeg(
                camera_name
            )
        )

        capture_ms = (
            perf_counter()
            - capture_started
        ) * 1000.0

        # ----------------------------------------------------
        # Image Preprocessing
        # ----------------------------------------------------

        jpeg_bytes = (
            _resize_jpeg(
                jpeg_bytes
            )
        )

        # ----------------------------------------------------
        # VLM
        # ----------------------------------------------------

        vlm_started = (
            perf_counter()
        )

        message = (
            run_vision_skill(
                image_jpeg=
                    jpeg_bytes,

                skill=
                    "describe_scene",

                response_schema=
                    _SCENE_SCHEMA,

                user_prompt=(
                    "請觀察目前這張 RGB 畫面，"
                    "描述場景本身、主要家具與固定環境結構，"
                    "並判斷可直接看出的開啟或關閉狀態。"
                ),

                backend=
                    backend_name,

                local_model=
                    OLLAMA_VLM_MODEL,

                temperature=
                    0.0,

                max_tokens=
                    _MAX_TOKENS,

                num_ctx=
                    8192,

                timeout=
                    _TIMEOUT_SEC,

                keep_alive=
                    _KEEP_ALIVE,

                wait=
                    True,

                owner=
                    "vlm_describe_scene",
            )
        )

        vlm_ms = (
            perf_counter()
            - vlm_started
        ) * 1000.0

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        scene = (
            _parse_scene_message(
                message
            )
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return success(
            MODULE,
            action,
            data={
                "camera_name":
                    camera_name,

                "source":
                    "camera_rgb",

                "backend":
                    backend_name,

                "model":
                    model_name,

                "scene":
                    scene,
            },

            timings_ms={
                "capture_rgb":
                    round(
                        capture_ms,
                        2,
                    ),

                "vlm_inference":
                    round(
                        vlm_ms,
                        2,
                    ),

                "total":
                    round(
                        (
                            perf_counter()
                            - started
                        )
                        * 1000.0,
                        2,
                    ),
            },
        )

    except Exception as exc:
        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
            timings_ms={
                "total":
                    round(
                        (
                            perf_counter()
                            - started
                        )
                        * 1000.0,
                        2,
                    ),
            },
        )


def get_health() -> dict[str, Any]:
    """Describe-scene capability metadata."""

    return {
        "capability":
            "describe_scene",

        "default_backend":
            "local",

        "remote_requires_explicit_selection":
            True,

        "local_model":
            OLLAMA_VLM_MODEL,

        "remote_model":
            remote_vlm.REMOTE_VLM_MODEL,

        "skill":
            "agent/skills/describe_scene.md",

        "runs_yolo":
            False,

        "focus":
            "scene_environment",

        "image_max_width":
            _MAX_IMAGE_WIDTH,

        "max_tokens":
            _MAX_TOKENS,

        "max_scene_elements":
            _MAX_SCENE_ELEMENTS,
    }
