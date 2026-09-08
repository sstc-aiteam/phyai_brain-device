"""Resolve semantic world facts from explicit, structured, cached, then VLM data.

Primary APIs for the closed-loop runtime are observe_fact()/observe_facts().
observe_requirement()/observe_requirements() and inspect_visual_state() are
retained for backward compatibility with the existing planning path.
"""

from __future__ import annotations

import json
import base64
import io
import re
from copy import deepcopy
from typing import Any

from PIL import Image

from config import OLLAMA_VLM_MODEL
from services import llm_service, vision_service


_CONTAINER_SUBJECTS = {"trash_can", "second_drawer", "top_cabinet"}
_SEMANTIC_PREDICATES = {"open_state", "semantic_relation", "visual_state"}
_DEFAULT_CAMERA_ROUTES = {
    ("trash_can", "open_state"): "middle",
    ("second_drawer", "open_state"): "middle",
    ("top_cabinet", "open_state"): "middle",
}
_SEMANTIC_VLM_MAX_WIDTH = 960

# Backward-compatible aliases used by older imports/tests.
_PLANNING_CAMERA_ROUTES = _DEFAULT_CAMERA_ROUTES
_PLANNING_VLM_MAX_WIDTH = _SEMANTIC_VLM_MAX_WIDTH


def _resize_planning_vlm_jpeg(jpeg: bytes) -> bytes:
    """Resize a full frame for planning VLM input without cropping."""
    with Image.open(io.BytesIO(jpeg)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width > _SEMANTIC_VLM_MAX_WIDTH:
            target_height = max(
                1,
                round(height * _SEMANTIC_VLM_MAX_WIDTH / width),
            )
            image = image.resize(
                (_SEMANTIC_VLM_MAX_WIDTH, target_height),
                Image.Resampling.LANCZOS,
            )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
        return output.getvalue()


def _unresolved(requirement: dict[str, Any], reason: str, *, source: str = "unresolved") -> dict[str, Any]:
    return {
        "requirement": dict(requirement),
        "source": source,
        "status": "unresolved",
        "fact": None,
        "reason": reason,
    }


def _parse_semantic_fact(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("semantic VLM 回傳空白 JSON")

    payload = content.strip()

    # Some VLMs wrap valid JSON in Markdown code fences.
    # Accept the fence, but still require a real semantic value below.
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, count=1, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload, count=1)
        payload = payload.strip()

    try:
        fact = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"semantic VLM 回傳無效 JSON：{exc}; raw={content!r}"
        ) from exc

    if not isinstance(fact, dict):
        raise ValueError("semantic VLM JSON 根節點必須是 object")

    value = fact.get("value")
    if value is None or (
        isinstance(value, str)
        and value.strip().lower() in {"", "unknown", "uncertain"}
    ):
        raise ValueError(
            f"semantic VLM JSON 缺少可用的 value：{fact!r}"
        )

    return fact


def _fact_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    raw = context.get("task_facts")
    if raw is None:
        raw = context.get("explicit_task_facts")
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if not isinstance(raw, dict):
        return []
    rows = []
    for key, value in raw.items():
        if "." in str(key):
            subject, predicate = str(key).split(".", 1)
            payload = value if isinstance(value, dict) else {"value": value}
            rows.append({"subject": subject, "predicate": predicate, **payload})
        elif isinstance(value, dict):
            for predicate, fact_value in value.items():
                payload = fact_value if isinstance(fact_value, dict) else {"value": fact_value}
                rows.append({"subject": str(key), "predicate": str(predicate), **payload})
    return rows


def _explicit_fact(subject: str, predicate: str, context: dict[str, Any]) -> dict[str, Any] | None:
    for row in _fact_rows(context):
        if str(row.get("subject")) == subject and str(row.get("predicate")) == predicate:
            if "value" not in row:
                continue
            fact = {
                key: deepcopy(value)
                for key, value in row.items()
                if key not in {"subject", "predicate", "source"}
            }
            return {
                "source": str(row.get("source") or "user_instruction"),
                "fact": fact,
            }
    return None


def _structured_context_fact(subject: str, predicate: str, context: dict[str, Any]) -> dict[str, Any] | None:
    for field in ("structured_facts", "sensor_facts"):
        nested_context = {"task_facts": context.get(field)}
        for row in _fact_rows(nested_context):
            if (
                str(row.get("subject")) == subject
                and str(row.get("predicate")) == predicate
                and "value" in row
            ):
                return {
                    key: deepcopy(value)
                    for key, value in row.items()
                    if key not in {"subject", "predicate", "source"}
                }
    return None


def _snapshot_from_context(
    context: dict[str, Any],
    camera_source: str | None = None,
) -> dict[str, Any] | None:
    if camera_source:
        for key in ("vision_snapshots", "scene_snapshots"):
            snapshots = context.get(key)
            value = snapshots.get(camera_source) if isinstance(snapshots, dict) else None
            if isinstance(value, dict):
                return value
    for key in ("vision_snapshot", "scene_snapshot"):
        value = context.get(key)
        if isinstance(value, dict) and (
            not camera_source
            or not value.get("camera_name")
            or value.get("camera_name") == camera_source
        ):
            return value
    detected = context.get("detected_objects")
    if isinstance(detected, dict):
        return {
            "timestamp": detected.get("latest_time"),
            "detections": detected.get("objects") or [],
            "detection_available": detected.get("detection_available"),
        }
    return None


def _capture_snapshot_once(context: dict[str, Any]) -> dict[str, Any] | None:
    supplied = _snapshot_from_context(context)
    if supplied is not None:
        return supplied
    getter = getattr(vision_service, "get_latest_scene_snapshot", None)
    if not callable(getter):
        return None
    snapshot = getter(include_robot_xyz=True)
    return snapshot if isinstance(snapshot, dict) else None


def _detections(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return []
    for key in ("detections", "yolo_detections", "objects"):
        rows = snapshot.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    data = snapshot.get("data")
    if isinstance(data, dict):
        return _detections(data)
    return []


def _binding_subject(requirement: dict[str, Any], context: dict[str, Any]) -> tuple[str | None, str | None]:
    _ = context
    subject = str(requirement.get("subject") or "").strip()
    if subject != "target_object":
        return subject, None
    binding = requirement.get("binding")
    if isinstance(binding, dict) and binding.get("object_id"):
        return str(binding["object_id"]), None
    return None, "target_object 缺少 deterministic object_id binding"


def _find_detection(subject: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if row.get("object_id") is not None and subject == str(row["object_id"]):
            return row
    return None


def _structured_fact(
    subject: str,
    predicate: str,
    snapshot: dict[str, Any] | None,
    context: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    rows = _detections(snapshot)
    detection = _find_detection(subject, rows)

    if predicate == "exists":
        if detection is not None:
            return "vision_yolo", {
                "value": True,
                "object_id": str(detection.get("object_id") or subject),
                "confidence": detection.get("confidence"),
            }
        if subject in _CONTAINER_SUBJECTS:
            return None
        if snapshot is not None:
            return "vision_yolo", {"value": False, "object_id": subject}
        return None

    if predicate == "position" and detection is not None:
        xyz = detection.get("robot_xyz")
        if (
            isinstance(xyz, (list, tuple))
            and len(xyz) == 3
            and all(isinstance(v, (int, float)) for v in xyz)
        ):
            return "vision_depth", {
                "value": [float(v) for v in xyz],
                "object_id": str(detection.get("object_id") or subject),
            }
        return None

    if predicate == "movement" and detection is not None and detection.get("movement") is not None:
        return "tracking", {
            "value": deepcopy(detection["movement"]),
            "object_id": str(detection.get("object_id") or subject),
        }

    if predicate == "open_state" and detection is not None:
        state = detection.get("open_state")
        if state is None and isinstance(detection.get("state_tags"), dict):
            state = detection["state_tags"].get("open_state")
        if state in {"open", "closed"}:
            return "structured_sensor", {
                "value": state,
                "confidence": detection.get("confidence"),
            }

    if predicate == "reachable":
        if detection is not None and isinstance(detection.get("reachable"), bool):
            return "geometry", {
                "value": detection["reachable"],
                "object_id": str(detection.get("object_id") or subject),
            }
        resolver = context.get("reachability_resolver")
        if callable(resolver):
            value = resolver(
                subject=subject,
                detection=deepcopy(detection),
                snapshot=snapshot,
            )
            if isinstance(value, bool):
                return "geometry", {"value": value, "object_id": subject}

    return None


def _cached_fact(subject: str, predicate: str, context: dict[str, Any]) -> dict[str, Any] | None:
    cache = context.get("perception_cache")
    cached = cache.get(f"{subject}.{predicate}") if isinstance(cache, dict) else None
    if (
        not isinstance(cached, dict)
        or not cached.get("fresh", True)
        or "value" not in cached
    ):
        return None
    fact = cached.get("value")
    if isinstance(fact, dict):
        return deepcopy(fact)
    return {
        key: deepcopy(value)
        for key, value in cached.items()
        if key != "fresh"
    }


def observe_requirement(requirement: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    """Resolve one requirement using a caller-supplied shared snapshot when available."""
    if not isinstance(requirement, dict) or not isinstance(context, dict):
        raise ValueError("requirement 與 context 必須是 dict")

    predicate = str(requirement.get("predicate") or "").strip()
    raw_subject = str(requirement.get("subject") or "").strip()
    camera_source = str(requirement.get("camera_source") or "").strip() or None
    if not raw_subject or not predicate:
        raise ValueError("requirement 必須包含 subject 與 predicate")

    subject, binding_error = _binding_subject(requirement, context)
    if binding_error:
        return _unresolved(requirement, binding_error, source="binding")
    assert subject is not None

    explicit = _explicit_fact(subject, predicate, context)
    if explicit is not None:
        return {"requirement": dict(requirement), **explicit}

    snapshot = _snapshot_from_context(context, camera_source)

    structured_context = _structured_context_fact(subject, predicate, context)
    if structured_context is not None:
        return {
            "requirement": dict(requirement),
            "source": "structured_sensor",
            "fact": structured_context,
        }

    structured = _structured_fact(subject, predicate, snapshot, context)
    if structured is not None:
        source, fact = structured
        return {
            "requirement": dict(requirement),
            "source": source,
            "fact": fact,
        }

    cached = _cached_fact(subject, predicate, context)
    if cached is not None:
        return {
            "requirement": dict(requirement),
            "source": "cache",
            "fact": cached,
        }

    if predicate == "reachable":
        return _unresolved(
            requirement,
            "沒有可用的 workspace/reachability rule",
            source="geometry",
        )

    if predicate not in _SEMANTIC_PREDICATES:
        return _unresolved(
            requirement,
            f"structured perception 無法解析 {subject}.{predicate}",
        )

    image_urls = context.get("image_urls")
    image_url = (
        image_urls.get(camera_source)
        if camera_source and isinstance(image_urls, dict)
        else None
    )
    if not image_url:
        image_url = context.get("image_url")
    if not isinstance(image_url, str) or not image_url:
        return _unresolved(
            requirement,
            "semantic perception 缺少 context.image_url",
            source="vlm",
        )

    message = llm_service.chat(
        [
            {
                "role": "system",
                "content": (
                    "Inspect the attached image and answer the requested visual question. "
                    "The subject identifies the object to inspect. "
                    "For predicate=open_state, determine whether the object is open or closed. "
                    "Return ONLY one JSON object with exactly these fields: "
                    '{"value":"open|closed|not_visible","confidence":0.0}. '
                    "Do not repeat the input. "
                    "Do not use Markdown code fences. "
                    "Do not explain your reasoning."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Inspect the attached image itself. "
                            f'Find the object identified as subject="{subject}" and determine '
                            f'the requested predicate="{predicate}". '
                            "Do not repeat the subject/predicate request. "
                            "Answer only with the requested JSON result."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
        model=context.get("model") or OLLAMA_VLM_MODEL,
        temperature=0.0,
        max_tokens=128,
        timeout=context.get("timeout"),
        wait=True,
        owner="task_perception",
    )

    content = message.get("content") if isinstance(message, dict) else None
    return {
        "requirement": dict(requirement),
        "source": "vlm",
        "fact": _parse_semantic_fact(content),
    }


def observe_requirements(
    requirements: list[dict[str, Any]],
    *,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve a batch while acquiring at most one structured vision snapshot."""
    if not isinstance(requirements, list) or not isinstance(context, dict):
        raise ValueError("requirements 必須是 list，context 必須是 dict")

    shared_context = dict(context)

    if _snapshot_from_context(shared_context) is None:
        needs_snapshot = False
        for requirement in requirements:
            if not isinstance(requirement, dict):
                continue
            subject, binding_error = _binding_subject(requirement, shared_context)
            predicate = str(requirement.get("predicate") or "")
            if binding_error or subject is None:
                continue
            if _explicit_fact(subject, predicate, shared_context) is None:
                needs_snapshot = True
                break

        if needs_snapshot:
            snapshot = _capture_snapshot_once(shared_context)
            if snapshot is not None:
                shared_context["vision_snapshot"] = snapshot

    return [
        observe_requirement(requirement, context=shared_context)
        for requirement in requirements
    ]



def observe_fact(
    *,
    subject: str,
    predicate: str,
    context: dict[str, Any] | None = None,
    camera_source: str | None = None,
    freshness: str = "current",
) -> dict[str, Any]:
    """Resolve one semantic fact for WorldState.

    Priority:
      explicit task fact -> structured sensor/vision -> fresh cache -> VLM.

    This function does not mutate WorldState; world_state_service owns that
    update boundary.
    """
    subject = str(subject or "").strip()
    predicate = str(predicate or "").strip()
    if not subject or not predicate:
        raise ValueError("subject 與 predicate 必須是非空字串")

    context = dict(context or {})
    effective_camera = (
        str(camera_source).strip()
        if isinstance(camera_source, str)
        and camera_source.strip()
        else _DEFAULT_CAMERA_ROUTES.get(
            (subject, predicate)
        )
    )

    requirement = {
        "subject": subject,
        "predicate": predicate,
        "timing": "runtime_observation",
        "freshness": freshness,
    }
    if effective_camera:
        requirement["camera_source"] = effective_camera

    observed = observe_requirement(
        requirement,
        context=context,
    )

    fact = observed.get("fact")
    result = {
        "subject": subject,
        "predicate": predicate,
        "source": observed.get("source"),
        "camera_source": effective_camera,
        "status": observed.get("status", "resolved"),
    }

    if isinstance(fact, dict):
        result.update(deepcopy(fact))
    else:
        result["value"] = None

    if observed.get("reason"):
        result["reason"] = observed["reason"]

    return result


def observe_facts(
    requests: list[dict[str, Any]],
    *,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve multiple semantic facts while sharing at most one snapshot."""
    if not isinstance(requests, list):
        raise ValueError("requests 必須是 list")

    context = dict(context or {})
    requirements: list[dict[str, Any]] = []
    normalized: list[tuple[str, str, str | None]] = []

    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("requests item 必須是 dict")

        subject = str(
            request.get("subject") or ""
        ).strip()
        predicate = str(
            request.get("predicate") or ""
        ).strip()
        if not subject or not predicate:
            raise ValueError(
                "每個 request 必須包含 subject/predicate"
            )

        camera_source = request.get("camera_source")
        if not (
            isinstance(camera_source, str)
            and camera_source.strip()
        ):
            camera_source = _DEFAULT_CAMERA_ROUTES.get(
                (subject, predicate)
            )
        else:
            camera_source = camera_source.strip()

        requirement = {
            "subject": subject,
            "predicate": predicate,
            "timing": request.get(
                "timing",
                "runtime_observation",
            ),
            "freshness": request.get(
                "freshness",
                "current",
            ),
        }
        if camera_source:
            requirement["camera_source"] = camera_source

        if isinstance(request.get("binding"), dict):
            requirement["binding"] = deepcopy(
                request["binding"]
            )

        requirements.append(requirement)
        normalized.append(
            (subject, predicate, camera_source)
        )

    observed_rows = observe_requirements(
        requirements,
        context=context,
    )

    results: list[dict[str, Any]] = []
    for (
        subject,
        predicate,
        camera_source,
    ), observed in zip(
        normalized,
        observed_rows,
    ):
        fact = (
            observed.get("fact")
            if isinstance(observed, dict)
            else None
        )
        result = {
            "subject": subject,
            "predicate": predicate,
            "source": (
                observed.get("source")
                if isinstance(observed, dict)
                else "unresolved"
            ),
            "camera_source": camera_source,
            "status": (
                observed.get("status", "resolved")
                if isinstance(observed, dict)
                else "unresolved"
            ),
        }
        if isinstance(fact, dict):
            result.update(deepcopy(fact))
        else:
            result["value"] = None

        if (
            isinstance(observed, dict)
            and observed.get("reason")
        ):
            result["reason"] = observed["reason"]

        results.append(result)

    return results

def inspect_visual_state(
    *,
    subject: str,
    predicate: str,
    context: dict[str, Any] | None = None,
    camera_source: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible semantic inspection helper.

    New closed-loop callers should prefer observe_fact().
    """
    context = dict(context or {})
    effective_camera = (
        str(camera_source).strip()
        if isinstance(camera_source, str)
        and camera_source.strip()
        else _DEFAULT_CAMERA_ROUTES.get(
            (subject, predicate)
        )
    )

    explicit = _explicit_fact(
        subject,
        predicate,
        context,
    )
    if explicit is None and effective_camera:
        image_urls = context.get("image_urls")
        has_image = (
            isinstance(image_urls, dict)
            and bool(image_urls.get(effective_camera))
        )

        if not has_image and not context.get("image_url"):
            jpeg = vision_service.get_camera_rgb_jpeg(
                camera_name=effective_camera
            )
            if (
                not isinstance(jpeg, (bytes, bytearray))
                or not jpeg
            ):
                raise RuntimeError(
                    f"{effective_camera} camera 沒有可用影像"
                )

            jpeg = _resize_planning_vlm_jpeg(
                bytes(jpeg)
            )
            image_urls = dict(image_urls or {})
            image_urls[effective_camera] = (
                "data:image/jpeg;base64,"
                + base64.b64encode(
                    bytes(jpeg)
                ).decode("ascii")
            )
            context["image_urls"] = image_urls

    observed = observe_fact(
        subject=subject,
        predicate=predicate,
        context=context,
        camera_source=effective_camera,
    )

    if (
        observed.get("status") == "unresolved"
        or "value" not in observed
        or observed.get("value") is None
    ):
        raise RuntimeError(
            f"semantic perception 無法解析 "
            f"{subject}.{predicate}："
            f"{observed.get('reason') or observed!r}"
        )

    return observed
