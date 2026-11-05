"""Shared structured model gateway for Agent and Plan.

All inference is delegated to services.model_service.qwen.chat() or
services.model_service.openai.chat().
This module preserves the prior public contract: message["content"] is a JSON
string.  OpenAI schemas that are incompatible with strict=True use a JSON
schema prompt with the existing downstream application validators.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services import model_service


DEFAULT_PROVIDER = "local"

SUPPORTED_PROVIDERS = {
    "local",
    "openai",
}

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_NUM_CTX = 8192
DEFAULT_TIMEOUT = 60.0
DEFAULT_KEEP_ALIVE = -1


class StructuredProviderError(RuntimeError):
    """Structured LLM provider request failed."""


# ============================================================
# Provider
# ============================================================

def resolve_provider(
    stage: str,
    provider: str | None = None,
) -> str:
    """
    Resolve the LLM provider for one planning stage.

    Default:
        local

    Explicit:
        local
        openai

    No environment-based or automatic fallback is performed here.
    """

    if not isinstance(stage, str) or not stage.strip():
        raise StructuredProviderError(
            "stage 必須是非空字串"
        )

    if provider is None:
        return DEFAULT_PROVIDER

    if not isinstance(provider, str):
        raise StructuredProviderError(
            "provider 必須是字串"
        )

    normalized = provider.strip().lower()

    if not normalized:
        return DEFAULT_PROVIDER

    if normalized not in SUPPORTED_PROVIDERS:
        raise StructuredProviderError(
            f"{stage} 不支援 provider={normalized!r}；"
            "只支援 local/openai（remote 已停用）"
        )

    return normalized


# ============================================================
# Validation
# ============================================================

def _validate_messages(
    messages: Any,
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise StructuredProviderError(
            "messages 必須是非空 list"
        )

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise StructuredProviderError(
                f"messages[{index}] 必須是 object"
            )

        role = message.get("role")

        if (
            not isinstance(role, str)
            or not role.strip()
        ):
            raise StructuredProviderError(
                f"messages[{index}].role 必須是非空字串"
            )

        if "content" not in message:
            raise StructuredProviderError(
                f"messages[{index}] 缺少 content"
            )

    return messages


def _validate_response_schema(
    response_schema: Any,
) -> dict[str, Any]:
    if not isinstance(response_schema, dict):
        raise StructuredProviderError(
            "response_schema 必須是 dict"
        )

    return response_schema


def _validate_model(
    model: Any,
) -> str | None:
    if model is None:
        return None

    if (
        not isinstance(model, str)
        or not model.strip()
    ):
        raise StructuredProviderError(
            "model 必須是非空字串或 None"
        )

    return model.strip()


def _normalize_message(
    message: Any,
    *,
    provider: str,
    stage: str,
) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise StructuredProviderError(
            f"{stage}/{provider} 回傳格式不是 message object"
        )

    content = message.get("content")

    # model_service.{qwen,openai}.chat(schema=...) return parsed JSON.
    # Keep downstream parsers untouched (they call json.loads(content)).
    if isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False)

    if not isinstance(content, str) or not content.strip():
        raise StructuredProviderError(
            f"{stage}/{provider} 回傳缺少有效 content"
        )

    # Text-only structured responses may use Markdown code fences.
    trimmed = content.strip()
    if trimmed.startswith("```") and trimmed.endswith("```"):
        lines = trimmed.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in ("```", "```json"):
            content = "\n".join(lines[1:-1]).strip()

    # Fail closed on malformed JSON rather than silently passing it to a
    # runtime planner or capability executor.
    try:
        json.loads(content)
    except (ValueError, TypeError) as exc:
        raise StructuredProviderError(
            f"{stage}/{provider} 未回傳有效 JSON"
        ) from exc

    normalized = {
        "role": str(
            message.get("role")
            or "assistant"
        ),
        "content": content,
    }
    for key in ("model", "usage", "duration_ns"):
        if key in message:
            normalized[key] = message[key]
    return normalized


def _record_usage(message: dict[str, Any], *, stage: str, provider: str, elapsed_sec: float) -> None:
    """Opt-in metadata-only usage logging; never logs prompts or API keys."""
    log_path = os.getenv("AGENT_LLM_USAGE_LOG", "").strip()
    show = os.getenv("AGENT_LLM_USAGE_PRINT", "").strip().lower() in ("1", "true", "yes")
    if not (log_path or show):
        return
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if type(input_tokens) is not int or type(output_tokens) is not int:
        return
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if type(cached) is not int or cached < 0 or cached > input_tokens:
        cached = 0
    model = message.get("model")
    total = usage.get("total_tokens")
    if type(total) is not int:
        total = input_tokens + output_tokens
    # Only price the verified default GPT-4.1 mini text model; others are unknown.
    cost = None
    if provider == "openai" and isinstance(model, str) and (
        model == "gpt-4.1-mini" or model.startswith("gpt-4.1-mini-")
    ):
        cost = round(((input_tokens - cached) * 0.40 + cached * 0.10 + output_tokens * 1.60) / 1_000_000, 9)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "stage": stage,
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached if provider == "openai" else None,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "latency_sec": round(elapsed_sec, 3),
        "duration_ns": message.get("duration_ns"),
        "estimated_cost_usd": cost,
    }
    if show:
        print("[LLM USAGE] " + json.dumps(row, ensure_ascii=False), flush=True)
    if log_path:
        try:
            path = Path(log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[LLM USAGE WARNING] log write failed: {exc}", flush=True)


# ============================================================
# Local backend
# ============================================================

def _chat_local(
    *,
    stage: str,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any],
    model: str | None,
    temperature: float,
    max_tokens: int,
    num_ctx: int,
    timeout: float,
    keep_alive: int | str,
    wait: bool,
    owner: str | None,
) -> dict[str, Any]:
    """
    Run structured inference on the Edge local LLM service.
    """

    effective_owner = (
        owner.strip()
        if isinstance(owner, str) and owner.strip()
        else f"planning_{stage}"
    )

    started = time.perf_counter()
    try:
        message = model_service.qwen.chat(
            messages=messages,
            schema=response_schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            timeout=timeout
        )

    except Exception as exc:
        raise StructuredProviderError(
            f"{stage}/local structured inference 失敗："
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _record_usage(message, stage=stage, provider="local", elapsed_sec=time.perf_counter() - started)
    return _normalize_message(
        message,
        provider="local",
        stage=stage,
    )


# ============================================================
# OpenAI backend
# ============================================================

def _openai_strict_schema_compatible(schema: dict[str, Any]) -> bool:
    """Conservative check before Model Service's mandatory strict=True mode.

    Strict mode requires all object properties to be required and prohibits
    arbitrary object keys.  The existing Agent/Runtime schemas often use
    optional fields or free-form arguments, so those cannot be sent as-is.
    """
    allowed = {
        "type", "properties", "required", "additionalProperties",
        "items", "enum", "anyOf", "description", "title", "const",
    }

    def valid(node: Any) -> bool:
        if not isinstance(node, dict) or set(node) - allowed:
            return False
        if "anyOf" in node:
            branches = node["anyOf"]
            if not isinstance(branches, list) or not branches:
                return False
            if not all(valid(branch) for branch in branches):
                return False
        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties")
            required = node.get("required")
            if not isinstance(props, dict) or not isinstance(required, list):
                return False
            if set(required) != set(props) or len(required) != len(set(required)):
                return False
            if node.get("additionalProperties") is not False:
                return False
            if not all(valid(value) for value in props.values()):
                return False
        if "items" in node and not valid(node["items"]):
            return False
        return True

    return schema.get("type") == "object" and valid(schema)


def _chat_openai(
    *,
    stage: str,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any],
    model: str | None,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:

    effective_api_key = (
        api_key.strip()
        if isinstance(api_key, str) and api_key.strip()
        else os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not effective_api_key:
        raise StructuredProviderError(
            "provider='openai' 需要 api_key 或環境變數 OPENAI_API_KEY"
        )

    # Model Service provides its own OPENAI_MODEL / default resolution.
    effective_model = (
        model.strip()
        if isinstance(model, str) and model.strip()
        else None
    )
    native_schema = (
        response_schema if _openai_strict_schema_compatible(response_schema)
        else None
    )
    effective_messages = messages
    if native_schema is None:
        effective_messages = [
            {
                "role": "system",
                "content": (
                    "Return exactly one valid JSON object, without Markdown "
                    "fences or any explanation. Follow this JSON Schema "
                    "as closely as possible; all application-level "
                    "constraints are validated afterward:\n"
                    + json.dumps(response_schema, ensure_ascii=False)
                ),
            },
            *messages,
        ]
    started = time.perf_counter()
    try:
        message = model_service.openai.chat(
            messages=effective_messages,
            api_key=effective_api_key,
            model=effective_model,
            schema=native_schema,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as exc:
        raise StructuredProviderError(
            f"{stage}/openai structured inference 失敗："
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _record_usage(message, stage=stage, provider="openai", elapsed_sec=time.perf_counter() - started)
    return _normalize_message(
        message,
        provider="openai",
        stage=stage,
    )


# ============================================================
# Public API
# ============================================================

def chat_structured(
    *,
    stage: str,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    num_ctx: int = DEFAULT_NUM_CTX,
    timeout: float = DEFAULT_TIMEOUT,
    keep_alive: int | str = DEFAULT_KEEP_ALIVE,
    wait: bool = True,
    owner: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Shared structured inference entry point.

    Examples:

        # Default: Edge local LLM
        chat_structured(
            stage="stage1",
            messages=messages,
            response_schema=schema,
        )

        # Explicit OpenAI API
        chat_structured(
            stage="brain",
            messages=messages,
            response_schema=schema,
            provider="openai",
            model="...",
        )

    There is intentionally no automatic fallback:

        local failure  -> raise
        openai failure -> raise

    api_key is only used by provider="openai".
    OPENAI_API_KEY may be used instead.
    ``remote`` is deliberately rejected, never routed to llama.cpp.
    """

    effective_provider = resolve_provider(
        stage,
        provider,
    )

    validated_messages = _validate_messages(
        messages
    )

    validated_schema = _validate_response_schema(
        response_schema
    )

    validated_model = _validate_model(
        model
    )

    if api_key is not None:
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
        ):
            raise StructuredProviderError(
                "api_key 必須是非空字串或 None"
            )

        if effective_provider != "openai":
            raise StructuredProviderError(
                "api_key 只可搭配 provider='openai'"
            )

    try:
        temperature = float(
            temperature
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise StructuredProviderError(
            "temperature 必須是 number"
        ) from exc

    try:
        max_tokens = int(
            max_tokens
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise StructuredProviderError(
            "max_tokens 必須是 integer"
        ) from exc

    if max_tokens <= 0:
        raise StructuredProviderError(
            "max_tokens 必須 > 0"
        )

    try:
        num_ctx = int(
            num_ctx
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise StructuredProviderError(
            "num_ctx 必須是 integer"
        ) from exc

    if num_ctx <= 0:
        raise StructuredProviderError(
            "num_ctx 必須 > 0"
        )

    try:
        timeout = float(
            timeout
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise StructuredProviderError(
            "timeout 必須是 number"
        ) from exc

    if timeout <= 0:
        raise StructuredProviderError(
            "timeout 必須 > 0"
        )

    if not isinstance(wait, bool):
        raise StructuredProviderError(
            "wait 必須是 bool"
        )

    if effective_provider == "local":
        return _chat_local(
            stage=stage,
            messages=validated_messages,
            response_schema=validated_schema,
            model=validated_model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            timeout=timeout,
            keep_alive=keep_alive,
            wait=wait,
            owner=owner,
        )

    if effective_provider == "openai":
        return _chat_openai(
            stage=stage,
            messages=validated_messages,
            response_schema=validated_schema,
            model=validated_model,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
        )

    # resolve_provider() already validates this.
    raise StructuredProviderError(
        f"不支援 provider：{effective_provider}"
    )
