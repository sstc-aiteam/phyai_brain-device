"""Verify visible completion of registered robot actions."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

PROMPT_FILE = Path(__file__).resolve().parents[1] / "skills" / "checkaction.md"
# A single RGB frame cannot prove a motion trajectory or a precise opening distance.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number_env(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _prompt(action: str, target_hint: str | None) -> str | None:
    from agent.tools import get_tool

    try:
        tool = get_tool(action)
    except LookupError:
        return None
    if not isinstance(tool, dict):
        return None
    description = tool.get("description")
    if not isinstance(description, str) or not description.strip():
        return None

    template = PROMPT_FILE.read_text(encoding="utf-8")
    if "{tool_description}" not in template:
        raise ValueError("checkaction.md is missing {tool_description}")
    if target_hint and target_hint.strip():
        hint = target_hint.strip()
    elif action in ("open_drawer", "close_drawer"):
        # Unspecified drawers are evaluated collectively, not resolved to one object.
        hint = "not specified; scene-wide drawer policy"
    else:
        hint = "not specified; require an unambiguous target"
    return (template.replace("{action}", action)
            .replace("{target_hint}", hint)
            .replace("{tool_description}", description.strip()))


def _jpeg_from_image(image_rgb: Any) -> bytes:
    """Encode one camera frame using the color order declared by model_service.

    Despite the historical name ``camera_rgb``, the robot SDK currently
    declares BGR pixels. OpenCV imencode expects BGR; converting those frames
    again with COLOR_RGB2BGR would swap red/blue channels.
    """
    if isinstance(image_rgb, bytes):
        if image_rgb[:3] != b"\xff\xd8\xff":
            raise ValueError("camera image bytes must contain JPEG")
        return image_rgb

    import cv2
    import numpy as np
    from services import model_service

    if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
        raise ValueError("camera_rgb must be a uint8 numpy.ndarray")
    if image_rgb.ndim == 2:
        bgr = image_rgb
    elif image_rgb.ndim == 3 and image_rgb.shape[2] in (3, 4):
        order = os.getenv(
            "CHECKACTION_CAMERA_COLOR_ORDER",
            getattr(model_service, "ACTION_VLA_CAMERA_COLOR_ORDER", "BGR"),
        ).strip().upper()
        if order not in ("RGB", "BGR"):
            raise ValueError("camera pixel order must be RGB or BGR")
        if image_rgb.shape[2] == 3:
            bgr = image_rgb if order == "BGR" else cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_BGRA2BGR if order == "BGR" else cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f"unsupported camera_rgb shape: {image_rgb.shape!r}")

    max_dim = int(_number_env("CHECKACTION_MAX_IMAGE_DIM", 1280, 320, 4096))
    height, width = bgr.shape[:2]
    if min(height, width) < 1:
        raise ValueError("camera frame has zero width/height")
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        bgr = cv2.resize(
            bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("camera image JPEG encoding failed")
    return encoded.tobytes()


def _capture(camera_name: str) -> tuple[bytes, str, str]:
    # Existing main.py owns the D405 pipeline: never create/start a second one.
    from services import perception_service

    observation = perception_service.capture_vision(
        camera_name=camera_name,
        run_yolo=False,
        draw=False,
        include_intrinsics=False,
        include_point_cloud=False,
        include_robot_xyz=False,
    )
    if not isinstance(observation, dict) or observation.get("camera_rgb") is None:
        raise RuntimeError("perception_service.capture_vision returned no camera_rgb")
    stamp = observation.get("timestamp")
    return _jpeg_from_image(observation["camera_rgb"]), str(stamp) if stamp is not None else _now(), "fresh_camera"


# Shared JSON contract for both existing model_service backends.
_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "completed": {"type": ["boolean", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "observed_state": {
            "type": "string",
            "enum": ["target_completed", "target_not_completed", "unknown"],
        },
    },
    "required": ["completed", "confidence", "observed_state"],
    "additionalProperties": False,
}


def _call_vlm(*, jpeg: bytes, prompt: str, backend: str) -> tuple[dict, str]:
    """Call existing robot model_service. No direct HTTP/SDK or custom API URL.

    - openai: model_service.openai.chat(..., api_key=..., schema=...)
    - qwen:   model_service.qwen.chat(..., schema=..., thinking=False)
    Both service clients return a dict with already-parsed ``content`` when a
    schema is passed. All request/connection errors propagate to checkaction()
    and become completed=None rather than a false robot continuation signal.
    """
    from services import model_service

    model_backend = backend.strip().lower()
    tokens = int(_number_env("CHECKACTION_MAX_TOKENS", 128, 32, 16384))
    timeout = _number_env("CHECKACTION_TIMEOUT_SECONDS", 35, 1, 300)
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    messages = [
        {"role": "system", "content": (
            "You verify the visible result of a robot action. "
            "Respond with exactly the requested structured JSON. "
            "Never guess when the target is ambiguous."
        )},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    if model_backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (required by model_service.openai.chat)")
        model = os.getenv("CHECKACTION_OPENAI_MODEL", "").strip() or model_service.OPENAI_MODEL
        result = model_service.openai.chat(
            messages=messages,
            api_key=api_key,
            model=model,
            max_tokens=tokens,
            timeout=timeout,
            schema=_RESULT_SCHEMA,
            reasoning_effort="none",
        )
    elif model_backend == "qwen":
        model = os.getenv("CHECKACTION_QWEN_MODEL", "").strip() or model_service.QWEN_MODEL
        result = model_service.qwen.chat(
            messages=messages,
            model=model,
            temperature=0.0,
            thinking=False,
            max_tokens=tokens,
            timeout=timeout,
            schema=_RESULT_SCHEMA,
        )
    else:
        raise ValueError("CHECKACTION_BACKEND must be 'openai' or 'qwen'; 'local' is no longer a backend")

    if not isinstance(result, dict):
        raise RuntimeError("model_service chat returned a non-dict response")
    content = result.get("content")
    if isinstance(content, str):
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:].strip()
        elif content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        content = json.loads(content)
    if not isinstance(content, dict):
        raise RuntimeError("model_service chat returned no parsed JSON object")
    return content, model


def _verify(parsed: dict, action: str, *, min_confidence: float) -> dict:
    result = {
        "completed": None,
        "confidence": None,
        "observed_state": "unknown",
        "verification_status": "inconclusive",
    }
    state = parsed.get("observed_state")
    confidence = parsed.get("confidence")
    completed = parsed.get("completed")
    if state not in ("target_completed", "target_not_completed", "unknown"):
        return result
    if type(completed) not in (bool, type(None)):
        return result
    if (isinstance(confidence, bool) or not isinstance(confidence, (float, int))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        return result
    result["confidence"] = float(confidence)
    if confidence < min_confidence:
        return result
    if completed is True and state == "target_completed":
        result["completed"] = True
        result["observed_state"] = "target_completed"
        result["verification_status"] = "verified"
    elif completed is False and state == "target_not_completed":
        result["completed"] = False
        result["observed_state"] = "target_not_completed"
        result["verification_status"] = "not_completed"
    return result


def checkaction(*, action: str, camera_name: str = "left", target_hint: str | None = None,
                backend: str | None = None, image_path: str | None = None) -> dict:
    started = time.perf_counter()
    requested_action = str(action or "").strip()
    camera = str(camera_name or "").strip()
    output: dict[str, Any] = {
        "action": requested_action,
        "camera_name": camera,
        "completed": None,
        "confidence": None,
        "observed_state": "unknown",
        "verification_status": "inconclusive",
        "observed_at": None,
        "frame_timestamp": None,
        "source": None,
    }
    try:
        if camera not in ("left", "right", "middle"):
            raise ValueError("camera_name must be left/right/middle")
        if target_hint is not None and (not isinstance(target_hint, str) or len(target_hint) > 300):
            raise ValueError("target_hint must be a string of 300 characters or fewer")
        prompt = _prompt(requested_action, target_hint)
        if prompt is None:
            output["verification_status"] = "unsupported"
            return output
        selected_backend = (backend or os.getenv("CHECKACTION_BACKEND", "local")).strip().lower()
        if selected_backend == "local":
            selected_backend = "qwen"
        min_confidence = _number_env("CHECKACTION_MIN_CONFIDENCE", 0.80, 0, 1)
        if image_path is not None:
            import cv2
            import numpy as np

            source_path = Path(image_path).expanduser()
            if source_path.stat().st_size > 15_000_000:
                raise ValueError("image_path file exceeds 15 MB")
            raw = source_path.read_bytes()
            if raw[:3] == b"\xff\xd8\xff":
                jpeg = raw
            else:
                decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError("image_path must point to a readable JPG/PNG image")
                jpeg = _jpeg_from_image(decoded)
            timestamp, source = _now(), "image_file"
        else:
            jpeg, timestamp, source = _capture(camera)
        output["frame_timestamp"] = timestamp
        output["source"] = source
        output["observed_at"] = _now()
        parsed, model = _call_vlm(jpeg=jpeg, prompt=prompt, backend=selected_backend)
        output.update(_verify(parsed, requested_action, min_confidence=min_confidence))
        output["backend"] = selected_backend
        output["model"] = model
    except Exception as exc:
        output.update(
            completed=None,
            verification_status="error",
            observed_state="unknown",
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )
    finally:
        output["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return output
