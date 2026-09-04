"""Per-request structured LLM provider adapter for abstract planning.

Credentials are accepted in memory for one request only. They are never added
to planner traces or persisted by this module.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from services import llm_service


class StructuredProviderError(RuntimeError):
    """A configured structured-output provider could not be used."""


def resolve_provider(stage: str, provider: str | None = None) -> str:
    value = provider
    if value is None:
        value = os.getenv(f"ABSTRACT_{stage.upper()}_PROVIDER", "local")
    if not isinstance(value, str) or value.strip().lower() not in {"local", "openai"}:
        raise StructuredProviderError(
            f"{stage} provider 只支援 local 或 openai，實際為：{value!r}"
        )
    return value.strip().lower()


def _extract_openai_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text
    raise StructuredProviderError("OpenAI response 缺少 output_text")


def chat_structured(
    *,
    stage: str,
    messages: list[dict[str, str]],
    response_schema: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Call local Ollama or OpenAI without mutating process environment."""
    selected = resolve_provider(stage, provider)
    selected_model = model.strip() if isinstance(model, str) and model.strip() else None
    if selected == "local":
        return llm_service.chat_structured(
            messages=messages,
            response_schema=response_schema,
            model=selected_model,
        )

    key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
    key = key or os.getenv("OPENAI_API_KEY", "").strip()
    selected_model = selected_model or os.getenv(
        f"ABSTRACT_{stage.upper()}_MODEL", ""
    ).strip()
    if not key:
        raise StructuredProviderError(f"{stage} 使用 openai 時必須提供 API key")
    if not selected_model:
        raise StructuredProviderError(f"{stage} 使用 openai 時必須提供 model")

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": selected_model,
                "input": messages,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": f"abstract_{stage.lower()}_output",
                        "schema": response_schema,
                        # Stage 1 has tool-dependent dynamic argument objects;
                        # Python performs the strict tool validation afterward.
                        # Stage 2 is a closed enum and can use strict mode.
                        "strict": stage.lower() == "stage2",
                    }
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f"; response={exc.response.text[:1000]}"
        raise StructuredProviderError(
            f"OpenAI {stage} request 失敗：{type(exc).__name__}: {exc}{detail}"
        ) from exc
    return {"content": _extract_openai_output_text(payload)}
