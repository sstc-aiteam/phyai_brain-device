"""
Domain-specific one-shot WorldState observation from an image file.

Intended usage:
    fake/staged image + known entities + selected domain
        -> domain skill
        -> structured observation patch

This module does NOT persist WorldState.
It also does NOT create canonical entity identities.

Place at:
    agent/vlm/vlm_world_observe.py
"""

from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from config import OLLAMA_VLM_MODEL
from services import model_service
from utils.response import error, success


MODULE = "vlm"

_MAX_IMAGE_WIDTH = 640
_JPEG_QUALITY = 88
_MAX_TOKENS = 1536
_TIMEOUT_SEC = 60.0
_KEEP_ALIVE = -1

_MAX_SUMMARY_LENGTH = 160
_MAX_UPDATES = 16
_MAX_RELATIONS = 16
_MAX_UNMATCHED = 8
_MAX_NAME_LENGTH = 40
_MAX_LOCATION_LENGTH = 60
_MAX_ATTRIBUTE_LENGTH = 30

_DOMAINS = {
    "hospital",
    "logistics",
    "dining",
    "inspection",
}

_RELATION_TYPES = {
    "on",
    "inside",
    "attached_to",
    "holding",
    "part_of",
    "near",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
}

_SKILLS_DIR = (
    Path(__file__).resolve().parent.parent
    / "skills"
)


# ============================================================
# Domain field contracts
# ============================================================

_COMMON_VISUAL_FIELDS: dict[str, dict[str, Any]] = {
    "category": {
        "type": "string",
    },
    "material": {
        "type": "string",
    },
    "location": {
        "type": "string",
    },
    "open_state": {
        "type": "string",
        "enum": [
            "open",
            "closed",
        ],
    },
}

_DOMAIN_FIELDS: dict[str, dict[str, dict[str, Any]]] = {
    "hospital": {},

    "logistics": {},

    "dining": {
        "contents": {
            "type": "string",
        },
        "fill_state": {
            "type": "string",
            "enum": [
                "empty",
                "filled",
            ],
        },
        "liquid_type": {
            "type": "string",
        },
        "occupied": {
            "type": "boolean",
        },
        "requested_item": {
            "type": "string",
        },
        "service_status": {
            "type": "string",
        },
        "service_priority": {
            "type": "string",
            "enum": [
                "low",
                "medium",
                "high",
            ],
        },
        "dirty_dishes": {
            "type": "integer",
            "minimum": 0,
        },
        "spill_detected": {
            "type": "boolean",
        },
    },

    "inspection": {
        "inspected": {
            "type": "boolean",
        },
        "health_state": {
            "type": "string",
            "enum": [
                "unknown",
                "normal",
                "anomaly",
            ],
        },
        "smoke_state": {
            "type": "string",
            "enum": [
                "unknown",
                "none",
                "light",
                "dense",
            ],
        },
        "temperature_c": {
            "type": "number",
        },
        "temperature_state": {
            "type": "string",
            "enum": [
                "unknown",
                "normal",
                "high",
                "low",
            ],
        },
        "evidence_captured": {
            "type": "boolean",
        },
        "anomaly_reported": {
            "type": "boolean",
        },
    },
}


def get_allowed_fields(
    domain: str,
) -> set[str]:
    domain = _normalize_domain(domain)

    return set(
        _COMMON_VISUAL_FIELDS
    ) | set(
        _DOMAIN_FIELDS[domain]
    )


# ============================================================
# Schema
# ============================================================

def _build_schema(
    domain: str,
) -> dict[str, Any]:
    domain = _normalize_domain(domain)

    field_properties = {
        **_COMMON_VISUAL_FIELDS,
        **_DOMAIN_FIELDS[domain],
    }

    return {
        "type": "object",
        "properties": {
            "scene_summary": {
                "type": "string",
                "maxLength":
                    _MAX_SUMMARY_LENGTH,
            },

            "entity_updates": {
                "type": "array",
                "maxItems":
                    _MAX_UPDATES,
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                        },
                        "fields": {
                            "type": "object",
                            "properties":
                                field_properties,
                            "additionalProperties":
                                False,
                            "minProperties": 1,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": [
                        "entity_id",
                        "fields",
                        "confidence",
                    ],
                    "additionalProperties":
                        False,
                },
            },

            "relations": {
                "type": "array",
                "maxItems":
                    _MAX_RELATIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                        },
                        "predicate": {
                            "type": "string",
                            "enum":
                                sorted(
                                    _RELATION_TYPES
                                ),
                        },
                        "object": {
                            "type": "string",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": [
                        "subject",
                        "predicate",
                        "object",
                        "confidence",
                    ],
                    "additionalProperties":
                        False,
                },
            },

            "unmatched_objects": {
                "type": "array",
                "maxItems":
                    _MAX_UNMATCHED,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "maxLength":
                                _MAX_NAME_LENGTH,
                        },
                        "location": {
                            "type": "string",
                            "maxLength":
                                _MAX_LOCATION_LENGTH,
                        },
                        "attributes": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {
                                "type": "string",
                                "maxLength":
                                    _MAX_ATTRIBUTE_LENGTH,
                            },
                        },
                    },
                    "required": [
                        "name",
                        "location",
                        "attributes",
                    ],
                    "additionalProperties":
                        False,
                },
            },
        },
        "required": [
            "scene_summary",
            "entity_updates",
            "relations",
            "unmatched_objects",
        ],
        "additionalProperties":
            False,
    }


# ============================================================
# Helpers
# ============================================================

def _normalize_domain(
    domain: Any,
) -> str:
    value = str(
        domain or ""
    ).strip().lower()

    if value not in _DOMAINS:
        raise ValueError(
            "domain 必須為："
            + ", ".join(
                sorted(_DOMAINS)
            )
        )

    return value


@lru_cache(maxsize=8)
def _load_skill(
    domain: str,
) -> str:
    domain = _normalize_domain(domain)

    path = (
        _SKILLS_DIR
        / f"{domain}_world.md"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"找不到 skill：{path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def _normalize_entities(
    entities: Any,
) -> list[dict[str, Any]]:
    """
    Accept:
      - list[entity]
      - canonical dict keyed by entity_id
    """

    if entities is None:
        return []

    if isinstance(
        entities,
        dict,
    ):
        converted = []

        for entity_id, state in (
            entities.items()
        ):
            if not isinstance(
                state,
                dict,
            ):
                raise ValueError(
                    f"entity {entity_id!r} 必須是 object"
                )

            converted.append({
                "entity_id":
                    str(entity_id),

                **state,
            })

        entities = converted

    if not isinstance(
        entities,
        list,
    ):
        raise ValueError(
            "entities 必須是 list 或 dict"
        )

    normalized = []
    seen_ids = set()

    for index, raw in enumerate(
        entities
    ):
        if not isinstance(
            raw,
            dict,
        ):
            raise ValueError(
                f"entities[{index}] 必須是 object"
            )

        entity_id = str(
            raw.get("entity_id")
            or raw.get("object_id")
            or raw.get("device_id")
            or ""
        ).strip()

        if not entity_id:
            raise ValueError(
                f"entities[{index}] 缺少 entity_id"
            )

        if entity_id in seen_ids:
            raise ValueError(
                f"重複 entity_id：{entity_id}"
            )

        seen_ids.add(
            entity_id
        )

        item = {
            "entity_id":
                entity_id,
        }

        # Forward enough information for visual association
        # and state reasoning, but do not convert raw entity
        # text into instructions.
        for key in (
            "name",
            "type",
            "category",
            "class_name",
            "tags",
            "attributes",
            "location",
            "bbox",
            "center",
            "camera_source",
            "open_state",
            "material",
        ):
            if key in raw:
                item[key] = raw[key]

        normalized.append(
            item
        )

    return normalized


def _load_image_jpeg(
    image_path: str | Path,
) -> bytes:
    path = Path(
        image_path
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"找不到測試圖片：{path}"
        )

    with Image.open(path) as image:
        image = image.convert("RGB")

        if image.width > _MAX_IMAGE_WIDTH:
            ratio = (
                _MAX_IMAGE_WIDTH
                / image.width
            )

            image = image.resize(
                (
                    _MAX_IMAGE_WIDTH,
                    max(
                        1,
                        int(
                            image.height
                            * ratio
                        ),
                    ),
                ),
                Image.Resampling.LANCZOS,
            )

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=_JPEG_QUALITY,
        )

        return output.getvalue()


def _image_data_url(
    jpeg_bytes: bytes,
) -> str:
    encoded = base64.b64encode(
        jpeg_bytes
    ).decode("ascii")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def _parse_message(
    message: Any,
    *,
    domain: str,
    valid_entity_ids: set[str],
    known_entities: list[dict[str, Any]],
    system_context: Any = None,
) -> tuple[
    dict[str, Any],
    list[str],
]:
    if not isinstance(
        message,
        dict,
    ):
        raise RuntimeError(
            "VLM response 必須是 message object"
        )

    content = message.get(
        "content"
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
        payload = json.loads(
            content
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

    warnings: list[str] = []

    summary = payload.get(
        "scene_summary"
    )

    if not isinstance(
        summary,
        str,
    ):
        raise RuntimeError(
            "scene_summary 必須是字串"
        )

    summary = summary.strip()

    if not summary:
        raise RuntimeError(
            "scene_summary 不可為空"
        )

    if (
        len(summary)
        > _MAX_SUMMARY_LENGTH
    ):
        raise RuntimeError(
            "scene_summary 過長"
        )

    allowed_fields = (
        get_allowed_fields(
            domain
        )
    )

    known_by_id = {
        item["entity_id"]: item
        for item in known_entities
        if isinstance(item, dict)
        and item.get("entity_id")
    }

    known_names = {
        str(item.get("name") or "").strip()
        for item in known_entities
        if isinstance(item, dict)
        and str(item.get("name") or "").strip()
    }

    allowed_zone_ids = {
        item["entity_id"]
        for item in known_entities
        if isinstance(item, dict)
        and item.get("entity_id")
        and str(item.get("type") or "").strip().lower()
        in {"zone", "area"}
    }

    if isinstance(system_context, dict):
        observation_context = system_context.get(
            "observation_context"
        )
        if isinstance(observation_context, dict):
            zone_id = str(
                observation_context.get("zone_id")
                or ""
            ).strip()
            if zone_id:
                allowed_zone_ids.add(zone_id)

    raw_updates = payload.get(
        "entity_updates"
    )

    if not isinstance(
        raw_updates,
        list,
    ):
        raise RuntimeError(
            "entity_updates 必須是 list"
        )

    updates = []

    for index, raw in enumerate(
        raw_updates
    ):
        if not isinstance(
            raw,
            dict,
        ):
            warnings.append(
                f"entity_updates[{index}] 非 object，已忽略"
            )
            continue

        entity_id = str(
            raw.get("entity_id")
            or ""
        ).strip()

        if entity_id not in (
            valid_entity_ids
        ):
            warnings.append(
                "忽略未知 entity_id："
                f"{entity_id!r}"
            )
            continue

        fields = raw.get(
            "fields"
        )

        if (
            not isinstance(
                fields,
                dict,
            )
            or not fields
        ):
            warnings.append(
                f"{entity_id} fields 為空，已忽略"
            )
            continue

        bad_fields = (
            set(fields)
            - allowed_fields
        )

        if bad_fields:
            warnings.append(
                f"{entity_id} 含不允許欄位："
                + ", ".join(
                    sorted(bad_fields)
                )
            )

            fields = {
                key: value
                for key, value
                in fields.items()
                if key
                in allowed_fields
            }

        if not fields:
            continue

        canonical = known_by_id.get(
            entity_id,
            {}
        )

        # Canonical/static metadata already known by the
        # Initial WorldState must not be re-decided by VLM.
        for static_field in (
            "category",
            "material",
        ):
            if (
                static_field in canonical
                and static_field in fields
            ):
                fields.pop(
                    static_field,
                    None,
                )
                warnings.append(
                    f"{entity_id}.{static_field} "
                    "已有 canonical value，忽略 VLM 更新"
                )

        # location is only allowed to reference a known
        # zone/area. Never accept invented names such as
        # sorting_area.
        if "location" in fields:
            location_value = str(
                fields.get("location")
                or ""
            ).strip()

            canonical_location = str(
                canonical.get("location")
                or ""
            ).strip()

            if canonical_location:
                # Existing canonical location is preserved.
                fields.pop(
                    "location",
                    None,
                )
                warnings.append(
                    f"{entity_id}.location "
                    "已有 canonical value，忽略 VLM 更新"
                )
            elif (
                not location_value
                or location_value
                not in allowed_zone_ids
            ):
                fields.pop(
                    "location",
                    None,
                )
                warnings.append(
                    f"{entity_id}.location "
                    f"引用未知 zone：{location_value!r}，已忽略"
                )

        if not fields:
            continue

        confidence = raw.get(
            "confidence"
        )

        if not isinstance(
            confidence,
            (int, float),
        ):
            warnings.append(
                f"{entity_id} confidence 無效，已忽略"
            )
            continue

        confidence = max(
            0.0,
            min(
                1.0,
                float(
                    confidence
                ),
            ),
        )

        updates.append({
            "entity_id":
                entity_id,

            "fields":
                fields,

            "confidence":
                confidence,
        })

    raw_relations = payload.get(
        "relations"
    )

    if not isinstance(
        raw_relations,
        list,
    ):
        raise RuntimeError(
            "relations 必須是 list"
        )

    relations = []
    relation_keys = set()

    for index, raw in enumerate(
        raw_relations
    ):
        if not isinstance(
            raw,
            dict,
        ):
            warnings.append(
                f"relations[{index}] 非 object，已忽略"
            )
            continue

        subject = str(
            raw.get("subject")
            or ""
        ).strip()

        predicate = str(
            raw.get("predicate")
            or ""
        ).strip()

        object_id = str(
            raw.get("object")
            or ""
        ).strip()

        confidence = raw.get(
            "confidence"
        )

        if (
            subject not in
                valid_entity_ids
            or object_id not in
                valid_entity_ids
        ):
            warnings.append(
                "忽略引用未知 entity 的 relation"
            )
            continue

        if subject == object_id:
            continue

        if predicate not in (
            _RELATION_TYPES
        ):
            continue

        if not isinstance(
            confidence,
            (int, float),
        ):
            continue

        confidence = float(
            confidence
        )

        if confidence < 0.75:
            warnings.append(
                "忽略 confidence < 0.75 的 relation："
                f"{subject} {predicate} {object_id}"
            )
            continue

        key = (
            subject,
            predicate,
            object_id,
        )

        if key in relation_keys:
            continue

        relation_keys.add(
            key
        )

        relations.append({
            "subject":
                subject,

            "predicate":
                predicate,

            "object":
                object_id,

            "confidence":
                min(
                    1.0,
                    confidence,
                ),
        })

    # Remove generic near if a more specific relation already
    # describes the same unordered entity pair.
    specific = {
        "holding",
        "inside",
        "attached_to",
        "part_of",
        "on",
        "left_of",
        "right_of",
        "in_front_of",
        "behind",
    }

    specific_pairs = {
        frozenset({
            item["subject"],
            item["object"],
        })
        for item in relations
        if item["predicate"]
        in specific
    }

    relations = [
        item
        for item in relations
        if not (
            item["predicate"]
            == "near"
            and frozenset({
                item["subject"],
                item["object"],
            })
            in specific_pairs
        )
    ]

    raw_unmatched = payload.get(
        "unmatched_objects"
    )

    if not isinstance(
        raw_unmatched,
        list,
    ):
        raise RuntimeError(
            "unmatched_objects 必須是 list"
        )

    unmatched = []

    for index, raw in enumerate(
        raw_unmatched[
            :_MAX_UNMATCHED
        ]
    ):
        if not isinstance(
            raw,
            dict,
        ):
            continue

        name = str(
            raw.get("name")
            or ""
        ).strip()

        location = str(
            raw.get("location")
            or ""
        ).strip()

        attributes = raw.get(
            "attributes"
        )

        if not name:
            continue

        if name in known_names:
            warnings.append(
                f"unmatched_object {name!r} "
                "已對應 known entity name，已忽略"
            )
            continue

        if not isinstance(
            attributes,
            list,
        ):
            attributes = []

        attributes = [
            str(item).strip()
            for item in attributes
            if str(item).strip()
        ][
            :3
        ]

        unmatched.append({
            "name":
                name[
                    :_MAX_NAME_LENGTH
                ],

            "location":
                location[
                    :_MAX_LOCATION_LENGTH
                ],

            "attributes": [
                item[
                    :_MAX_ATTRIBUTE_LENGTH
                ]
                for item
                in attributes
            ],
        })

    observation = {
        "scene_summary":
            summary,

        "entity_updates":
            updates,

        "relations":
            relations,

        "unmatched_objects":
            unmatched,
    }

    return (
        observation,
        warnings,
    )


# ============================================================
# Public capability
# ============================================================

def observe_world_image(
    *,
    image_path: str | Path,
    domain: str,
    entities: Any,
    task_context: Any = None,
    system_context: Any = None,
) -> dict[str, Any]:
    """
    Test/staging capability using an image file instead of
    a live camera.

    This returns an observation patch only.
    """

    action = (
        "observe_world_image"
    )

    started = perf_counter()

    try:
        domain = _normalize_domain(
            domain
        )

        normalized_entities = (
            _normalize_entities(
                entities
            )
        )

        valid_entity_ids = {
            item["entity_id"]
            for item
            in normalized_entities
        }

        jpeg_bytes = (
            _load_image_jpeg(
                image_path
            )
        )

        entity_context = json.dumps(
            normalized_entities,
            ensure_ascii=False,
            indent=2,
        )

        task_context_text = (
            json.dumps(
                task_context
                if task_context is not None
                else {},
                ensure_ascii=False,
                indent=2,
            )
        )

        system_context_text = (
            json.dumps(
                system_context
                if system_context is not None
                else {},
                ensure_ascii=False,
                indent=2,
            )
        )

        message = (
            model_service.qwen
            .chat_structured(
                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            _load_skill(
                                domain
                            ),
                    },
                    {
                        "role":
                            "user",

                        "content": [
                            {
                                "type":
                                    "text",

                                "text": (
                                    f"目前 domain 已由外部指定為：{domain}\n"
                                    "不要重新分類場域。\n\n"
                                    "以下為已知 entities：\n"
                                    f"{entity_context}\n\n"
                                    "以下為 task_context；只有 skill 明確允許"
                                    "使用的非視覺欄位才能引用：\n"
                                    f"{task_context_text}\n\n"
                                    "以下為 system_context；只有 skill 明確允許"
                                    "使用的系統狀態才能引用：\n"
                                    f"{system_context_text}\n\n"
                                    "請根據這張 RGB 與上述 context，"
                                    "輸出本次 observation patch。"
                                ),
                            },
                            {
                                "type":
                                    "image_url",

                                "image_url": {
                                    "url":
                                        _image_data_url(
                                            jpeg_bytes
                                        ),
                                },
                            },
                        ],
                    },
                ],

                response_schema=
                    _build_schema(
                        domain
                    ),

                model=
                    OLLAMA_VLM_MODEL,

                temperature=0.0,

                max_tokens=
                    _MAX_TOKENS,

                num_ctx=8192,

                timeout=
                    _TIMEOUT_SEC,

                keep_alive=
                    _KEEP_ALIVE,

                wait=True,

                owner=
                    f"vlm_world_{domain}",
            )
        )

        (
            observation,
            warnings,
        ) = _parse_message(
            message,
            domain=domain,
            valid_entity_ids=
                valid_entity_ids,
            known_entities=
                normalized_entities,
            system_context=
                system_context,
        )

        return success(
            MODULE,
            action,
            data={
                "domain":
                    domain,

                "source":
                    "image_file",

                "model":
                    OLLAMA_VLM_MODEL,

                "observation":
                    observation,

                "warnings":
                    warnings,
            },
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
    return {
        "capability":
            "observe_world_image",

        "domains":
            sorted(_DOMAINS),

        "image_max_width":
            _MAX_IMAGE_WIDTH,

        "max_tokens":
            _MAX_TOKENS,
    }
