from __future__ import annotations
import io
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from threading import Lock
from typing import Any
from PIL import Image
from config import OLLAMA_VLM_MODEL
from services import camera_service, model_service, perception_service
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
# Short-lived per-camera scene continuity memory.
#
# This is only a naming / description-granularity reference for consecutive
# scene observations. It is NOT physical truth and never overrides CURRENT RGB.
_PREVIOUS_SCENE_TTL_SEC = 120.0
_PREVIOUS_SCENE_BY_CAMERA: dict[
    str,
    tuple[
        float,
        dict[str, Any],
    ],
] = {}
_PREVIOUS_SCENE_LOCK = Lock()
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
            .get_camera_image(
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
            .get_camera_image(
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
def _save_scene_debug(
    *,
    jpeg_bytes: bytes,
    scene: dict[str, Any],
    camera_name: str,
) -> None:
    """
    Optional debug output.

    Saves the exact resized JPEG sent to the VLM and the parsed scene JSON.
    No point / bbox / mask visualization is produced.

    Enabled only when:
        VLM_SCENE_DEBUG_DIR=/some/path
    """
    debug_dir = os.getenv(
        "VLM_SCENE_DEBUG_DIR",
        "",
    ).strip()

    if not debug_dir:
        return

    try:
        output_dir = (
            Path(debug_dir)
            .expanduser()
            .resolve()
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_camera_name = (
            camera_name
            .replace("/", "_")
            .replace("\\", "_")
        )

        input_path = output_dir / f"vlm_scene_{safe_camera_name}_input.jpg"
        json_path = output_dir / f"vlm_scene_{safe_camera_name}.json"

        input_path.write_bytes(jpeg_bytes)
        json_path.write_text(
            json.dumps(
                {
                    "camera_name": camera_name,
                    "scene": scene,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("[vlm-scene-debug] input: " f"{input_path}")
        print("[vlm-scene-debug] json : " f"{json_path}")

    except Exception as exc:
        print(
            "[vlm-scene-debug][WARN] "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
def _normalize_previous_scene(
    previous_scene: Any,
) -> dict[str, Any] | None:
    """
    Normalize an optional previous scene reference.
    Accepted forms:
    - the scene dict itself
    - {"scene": {...}}
    - a normal service response containing data.scene
    The returned reference intentionally contains NO open_state values.
    Previous state must never anchor the CURRENT visual state decision.
    """
    if previous_scene is None:
        return None
    if not isinstance(
        previous_scene,
        dict,
    ):
        raise ValueError(
            "previous_scene 必須是 dict 或 None"
        )
    candidate = previous_scene
    data = candidate.get(
        "data"
    )
    if isinstance(
        data,
        dict,
    ) and isinstance(
        data.get(
            "scene"
        ),
        dict,
    ):
        candidate = data[
            "scene"
        ]
    elif isinstance(
        candidate.get(
            "scene"
        ),
        dict,
    ):
        candidate = candidate[
            "scene"
        ]
    scene_type = str(
        candidate.get(
            "scene_type"
        )
        or ""
    ).strip()
    elements = candidate.get(
        "scene_elements"
    )
    if not isinstance(
        elements,
        list,
    ):
        return None
    compact_elements = []
    for item in elements[
        :_MAX_SCENE_ELEMENTS
    ]:
        if not isinstance(
            item,
            dict,
        ):
            continue
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
        if not name:
            continue
        attributes = (
            item.get(
                "attributes"
            )
        )
        if not isinstance(
            attributes,
            list,
        ):
            attributes = []
        count = item.get(
            "count"
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
            count = 1
        compact_elements.append({
            "name":
                name[
                    :_MAX_NAME_LENGTH
                ],
            "count":
                count,
            "location":
                location[
                    :_MAX_LOCATION_LENGTH
                ],
            "attributes": [
                attribute.strip()[
                    :_MAX_ATTRIBUTE_LENGTH
                ]
                for attribute
                in attributes[
                    :3
                ]
                if isinstance(
                    attribute,
                    str,
                )
                and attribute.strip()
            ],
        })
    if not compact_elements:
        return None
    result = {
        "scene_elements":
            compact_elements,
    }
    if scene_type in _SCENE_TYPES:
        result[
            "scene_type"
        ] = scene_type
    return result
def _get_cached_previous_scene(
    camera_name: str,
) -> dict[str, Any] | None:
    """
    Return a fresh-enough previous scene reference for one camera.
    Cache is process-local and short-lived. It is a continuity hint only.
    """
    now = time.monotonic()
    with _PREVIOUS_SCENE_LOCK:
        cached = (
            _PREVIOUS_SCENE_BY_CAMERA
            .get(
                camera_name
            )
        )
        if cached is None:
            return None
        observed_at, scene = cached
        if (
            now
            - observed_at
            > _PREVIOUS_SCENE_TTL_SEC
        ):
            _PREVIOUS_SCENE_BY_CAMERA.pop(
                camera_name,
                None,
            )
            return None
        return json.loads(
            json.dumps(
                scene,
                ensure_ascii=False,
            )
        )
def _remember_scene(
    camera_name: str,
    scene: dict[str, Any],
) -> None:
    """
    Save only continuity-safe scene identity hints.
    open_state is intentionally stripped by _normalize_previous_scene().
    """
    compact = (
        _normalize_previous_scene(
            scene
        )
    )
    if compact is None:
        return
    with _PREVIOUS_SCENE_LOCK:
        _PREVIOUS_SCENE_BY_CAMERA[
            camera_name
        ] = (
            time.monotonic(),
            compact,
        )
def reset_scene_memory(
    camera_name: Any = None,
) -> None:
    """
    Clear process-local scene continuity memory.
    Useful if a camera is physically moved or a test needs a clean start.
    """
    if camera_name is None:
        with _PREVIOUS_SCENE_LOCK:
            _PREVIOUS_SCENE_BY_CAMERA.clear()
        return
    normalized = (
        _normalize_camera_name(
            camera_name
        )
    )
    with _PREVIOUS_SCENE_LOCK:
        _PREVIOUS_SCENE_BY_CAMERA.pop(
            normalized,
            None,
        )
def _build_scene_user_prompt(
    previous_scene: dict[str, Any] | None,
) -> str:
    """
    Build the current-frame instruction with an optional continuity reference.
    """
    prompt = (
        "請觀察目前這張 RGB 畫面，"
        "描述場景本身、主要家具與固定環境結構，"
        "並判斷可直接看出的開啟或關閉狀態。"
    )
    if previous_scene is None:
        return prompt
    reference_json = json.dumps(
        previous_scene,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )
    return (
        prompt
        + "\n\n"
        "以下 previous_scene_reference 是同一相機上一輪的場景描述，"
        "僅用於保持相鄰畫面中同一場景元素的名稱、實體連續性與描述粒度一致。"
        "CURRENT RGB 影像永遠是目前 physical state 的唯一依據。"
        "若目前影像與 previous_scene_reference 衝突，必須相信目前影像。"
        "不要因為門板變得更顯眼、開關狀態改變或視角略變，"
        "就任意把同一櫃體從「櫃子」改名成「門」或其他新實體。"
        "previous_scene_reference 不包含可信的目前 open_state；"
        "所有 open_state 都必須重新從 CURRENT RGB 判斷。"
        "上一輪存在但目前看不到的元素，不得僅因 previous_scene_reference 存在就再次輸出。"
        "\nprevious_scene_reference="
        + reference_json
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
    previous_scene: Any = None,
    use_previous_scene: bool = True,
) -> dict[str, Any]:
    """
    Capture exactly one RGB frame and describe the scene itself.
    Output focuses on:
    - scene type
    - furniture / fixtures / large environment structures
    - visible open / closed state
    - stable structural descriptions for cross-frame continuity
    - concise scene-level summary
    Small task objects are intentionally not the focus.
    No YOLO, segmentation, pixel grounding, or robot action is executed.
    Temporal continuity:
    - previous_scene can be supplied explicitly by the caller.
    - otherwise, when use_previous_scene=True, a short-lived per-camera
      continuity reference from the previous successful call is used.
    - previous scene data is only a naming / identity hint.
    - CURRENT RGB remains authoritative for open_state and current visibility.
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
        if not isinstance(
            use_previous_scene,
            bool,
        ):
            raise ValueError(
                "use_previous_scene 必須是 bool"
            )
        previous_scene_source = (
            "none"
        )
        continuity_reference = None
        if previous_scene is not None:
            continuity_reference = (
                _normalize_previous_scene(
                    previous_scene
                )
            )
            if continuity_reference is not None:
                previous_scene_source = (
                    "provided"
                )
        elif use_previous_scene:
            continuity_reference = (
                _get_cached_previous_scene(
                    camera_name
                )
            )
            if continuity_reference is not None:
                previous_scene_source = (
                    "cache"
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
                user_prompt=
                    _build_scene_user_prompt(
                        continuity_reference
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
        # Optional debug output using the exact image sent to the VLM.
        _save_scene_debug(
            jpeg_bytes=
                jpeg_bytes,
            scene=
                scene,
            camera_name=
                camera_name,
        )
        # Cache only continuity-safe identity hints after CURRENT output
        # passes validation. open_state is never carried forward.
        _remember_scene(
            camera_name,
            scene,
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
                "scene_continuity": {
                    "previous_scene_used":
                        continuity_reference
                        is not None,
                    "previous_scene_source":
                        previous_scene_source,
                },
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
        "returns_pixel_grounding":
            False,
        "image_max_width":
            _MAX_IMAGE_WIDTH,
        "max_tokens":
            _MAX_TOKENS,
        "max_scene_elements":
            _MAX_SCENE_ELEMENTS,
        "temporal_scene_continuity":
            True,
        "previous_scene_ttl_sec":
            _PREVIOUS_SCENE_TTL_SEC,
        "previous_scene_open_state_authoritative":
            False,
    }
