"""
Structured LLM provider used by planning modules.

Supported providers:

- local:
    Use services.model_service on the Edge machine.

- remote:
    Use agent.vlm.remote_vlm to call the remote OpenAI-compatible
    llama.cpp server.

- openai:
    Use the official OpenAI Responses API.

Rules:

- Default provider is always "local".
- "remote" and "openai" are used only when explicitly requested.
- There is NO automatic fallback between providers.

The returned message shape is normalized to:

    {
        "role": "assistant",
        "content": "<JSON string>",
    }

so callers can share the same parser regardless of backend.
"""

from __future__ import annotations

import os
from typing import Any

from agent.vlm import remote_vlm
from services import model_service


DEFAULT_PROVIDER = "local"

SUPPORTED_PROVIDERS = {
    "local",
    "remote",
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
        remote
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
            "只支援 local/remote/openai"
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

    if (
        not isinstance(content, str)
        or not content.strip()
    ):
        raise StructuredProviderError(
            f"{stage}/{provider} 回傳缺少有效 content"
        )

    return {
        "role": str(
            message.get("role")
            or "assistant"
        ),
        "content": content,
    }


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

    try:
        message = model_service.qwen.chat_structured(
            messages=messages,
            response_schema=response_schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            timeout=timeout,
            keep_alive=keep_alive,
            wait=wait,
            owner=effective_owner,
        )

    except Exception as exc:
        raise StructuredProviderError(
            f"{stage}/local structured inference 失敗："
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return _normalize_message(
        message,
        provider="local",
        stage=stage,
    )


# ============================================================
# Remote backend
# ============================================================

def _chat_remote(
    *,
    stage: str,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any],
    model: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """
    Run structured inference on the explicitly selected remote llama.cpp server.

    remote_vlm.py owns:
    - remote URL
    - HTTP request
    - OpenAI-compatible llama.cpp payload
    - remote connectivity errors
    """

    try:
        message = remote_vlm.chat_structured(
            messages=messages,
            response_schema=response_schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    except Exception as exc:
        raise StructuredProviderError(
            f"{stage}/remote structured inference 失敗："
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return _normalize_message(
        message,
        provider="remote",
        stage=stage,
    )


# ============================================================
# OpenAI backend
# ============================================================

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
    """
    Run structured inference through the official OpenAI Responses API.

    This backend is explicit-only:
        provider="openai"

    API key resolution:
        1. api_key argument
        2. OPENAI_API_KEY environment variable

    Model resolution:
        1. model argument
        2. OPENAI_MODEL environment variable

    No provider fallback is performed.
    """

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise StructuredProviderError(
            "OpenAI provider 需要 openai Python package；"
            "請執行：pip install -U openai"
        ) from exc

    effective_api_key = (
        api_key.strip()
        if isinstance(api_key, str) and api_key.strip()
        else os.environ.get(
            "OPENAI_API_KEY",
            "",
        ).strip()
    )

    if not effective_api_key:
        raise StructuredProviderError(
            "provider='openai' 需要 api_key "
            "或環境變數 OPENAI_API_KEY"
        )

    effective_model = (
        model.strip()
        if isinstance(model, str) and model.strip()
        else os.environ.get(
            "OPENAI_MODEL",
            "",
        ).strip()
    )

    if not effective_model:
        raise StructuredProviderError(
            "provider='openai' 需要 model "
            "或環境變數 OPENAI_MODEL"
        )

    try:
        client = OpenAI(
            api_key=effective_api_key,
            timeout=timeout,
        )

        response = client.responses.create(
            model=effective_model,
            input=messages,
            max_output_tokens=max_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "structured_runtime_output",
                    "schema": response_schema,

                    # Existing runtime schemas were written for the current
                    # local/llama.cpp structured provider and may not satisfy
                    # every restriction of OpenAI strict JSON Schema mode.
                    #
                    # Keep the SAME schema for A/B testing, but do not require
                    # OpenAI's strict-subset validation at the API boundary.
                    "strict": False,
                }
            },
        )

        content = response.output_text

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise StructuredProviderError(
                f"{stage}/openai Responses API "
                "沒有回傳有效 output_text"
            )

        message = {
            "role": "assistant",
            "content": content,
        }

    except StructuredProviderError:
        raise

    except Exception as exc:
        raise StructuredProviderError(
            f"{stage}/openai structured inference 失敗："
            f"{type(exc).__name__}: {exc}"
        ) from exc

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

        # Explicit remote: PC llama.cpp
        chat_structured(
            stage="stage1",
            messages=messages,
            response_schema=schema,
            provider="remote",
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
        remote failure -> raise
        openai failure -> raise

    api_key is only used by provider="openai".
    OPENAI_API_KEY may be used instead.
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

    if effective_provider == "remote":
        return _chat_remote(
            stage=stage,
            messages=validated_messages,
            response_schema=validated_schema,
            model=validated_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
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
