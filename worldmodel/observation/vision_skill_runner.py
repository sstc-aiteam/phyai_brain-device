"""Structured vision Skill runner through services.model_service only.

Compatible with  model_service.py
Vision clients still receive {'role': 'assistant', 'content': '<JSON string>'},
which preserves the current vlm_describe/vlm_world_observe parsers.
"""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from services import model_service


_SKILLS_DIR = Path(__file__).resolve().parents[2] / "worldmodel" / "skills"
_SUPPORTED_BACKENDS = frozenset({"local", "openai"})


@lru_cache(maxsize=None)
def load_skill(skill: str) -> str:
    """Read existing skill Markdown, keeping old skill path semantics."""
    if not isinstance(skill, str):
        raise TypeError("skill 必須是字串")
    value = skill.strip()
    if not value:
        raise ValueError("skill 不可為空")
    candidate = Path(value).expanduser()
    if candidate.suffix == ".md" or "/" in value:
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / candidate
        path = candidate.resolve()
    else:
        path = (_SKILLS_DIR / f"{value}.md").resolve()
    if not path.is_file():
        raise RuntimeError(f"VLM skill 不存在：{path}")
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) == 3:
            content = parts[2].strip()
    if not content:
        raise RuntimeError(f"VLM skill 為空：{path}")
    return content


def _image_data_url(image_jpeg: bytes) -> str:
    if not isinstance(image_jpeg, (bytes, bytearray)) or not image_jpeg:
        raise ValueError("image_jpeg 必須是非空 JPEG bytes")
    return "data:image/jpeg;base64," + base64.b64encode(bytes(image_jpeg)).decode("ascii")


def _resolve_backend(backend: str | None) -> str:
    if backend is None:
        return "local"
    if not isinstance(backend, str):
        raise TypeError("backend 必須是字串或 None")
    result = backend.strip().lower()
    if result not in _SUPPORTED_BACKENDS:
        raise ValueError(f"不支援 vision backend={result!r}；僅支援 local/openai")
    return result


# The new model_service.openai.chat unconditionally uses strict=True.
# The existing vision schemas use bounds such as maxLength, maxItems,
# minProperties and optional object properties, which OpenAI strict mode does
# not accept. Adapt *only the request schema*; downstream VLM code continues
# enforcing its existing semantic and range constraints.
_OPENAI_STRICT_IGNORED = {
    "maxLength", "minLength", "maxItems", "minItems", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minProperties", "maxProperties",
    "pattern", "format", "multipleOf", "uniqueItems", "default",
}


def _openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise TypeError("response_schema 必須是 dict")

    def walk(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        transformed = {
            key: value
            for key, value in node.items()
            if key not in _OPENAI_STRICT_IGNORED
        }
        if transformed.get("type") == "object":
            original_props = transformed.get("properties", {})
            if not isinstance(original_props, dict):
                raise ValueError("JSON Schema object.properties 必須是 dict")
            original_required = transformed.get("required", [])
            if not isinstance(original_required, list):
                raise ValueError("JSON Schema object.required 必須是 list")
            required_set = set(original_required)
            new_props = {}
            for name, spec in original_props.items():
                new_spec = walk(spec)
                if name not in required_set:
                    new_spec = {"anyOf": [new_spec, {"type": "null"}]}
                new_props[name] = new_spec
            transformed["properties"] = new_props
            transformed["required"] = list(original_props)
            transformed["additionalProperties"] = False
        if "items" in transformed:
            transformed["items"] = walk(transformed["items"])
        if "anyOf" in transformed:
            transformed["anyOf"] = [walk(option) for option in transformed["anyOf"]]
        if "$defs" in transformed:
            transformed["$defs"] = {
                k: walk(v) for k, v in transformed["$defs"].items()
            }
        return transformed

    return walk(schema)


def _prune_optional_nulls(value: Any, schema: dict[str, Any]) -> Any:
    """Undo strict-schema nullable placeholders for formerly optional fields."""
    if not isinstance(schema, dict):
        return value
    if isinstance(value, list):
        item_schema = schema.get("items")
        return [_prune_optional_nulls(v, item_schema) for v in value]
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return value
        required = set(schema.get("required", []))
        cleaned = {}
        for key, subvalue in value.items():
            if subvalue is None and key in properties and key not in required:
                continue
            cleaned[key] = _prune_optional_nulls(subvalue, properties.get(key, {}))
        return cleaned
    return value


def _normalize_message(message: Any, *, original_schema: dict[str, Any]) -> dict[str, str]:
    """Both backends return parsed objects for schema mode; callers expect a string."""
    if not isinstance(message, dict):
        raise RuntimeError("model_service.chat 回傳不是 message object")
    content = message.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("model_service.chat 沒有回傳有效 JSON") from exc
    if not isinstance(content, dict):
        raise RuntimeError("model_service.chat 結構化 content 必須是 JSON object")
    normalized = _prune_optional_nulls(content, original_schema)
    return {"role": "assistant", "content": json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))}


def run_vision_skill(
    *,
    image_jpeg: bytes,
    skill: str,
    response_schema: dict[str, Any],
    user_prompt: str,
    backend: str | None = None,
    local_model: str | None = None,
    remote_model: str | None = None,  # legacy argument, deliberately not used
    temperature: float = 0.0,
    max_tokens: int = 512,
    num_ctx: int = 8192,
    timeout: float = 60.0,
    keep_alive: int | str | None = -1,  # managed centrally by model_service
    wait: bool = True,
    owner: str = "vision_skill",  # legacy log tag, not in new model_service API
    openai_model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Route local/OpenAI vision through model_service, with no hardware actions."""
    target = _resolve_backend(backend)
    if not isinstance(response_schema, dict):
        raise TypeError("response_schema 必須是 dict")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt 必須是非空字串")
    if target == "local" and wait is False:
        raise ValueError("新版 model_service.qwen.chat 不提供 wait=False 非阻塞模式")

    messages = [
        {"role": "system", "content": load_skill(skill)},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt.strip()},
            {"type": "image_url", "image_url": {"url": _image_data_url(image_jpeg)}},
        ]},
    ]

    if target == "local":
        message = model_service.qwen.chat(
            messages=messages,
            schema=response_schema,
            model=local_model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            timeout=timeout,
        )
    else:
        effective_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not effective_key:
            raise ValueError("OpenAI Vision 需要 OPENAI_API_KEY 或 api_key")
        message = model_service.openai.chat(
            messages=messages,
            api_key=effective_key,
            schema=_openai_strict_schema(response_schema),
            model=openai_model or os.environ.get("OPENAI_MODEL") or model_service.OPENAI_MODEL,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    return _normalize_message(message, original_schema=response_schema)
