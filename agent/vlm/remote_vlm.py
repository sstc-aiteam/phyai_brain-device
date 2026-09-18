"""
Generic remote VLM client.

This module knows only how to call the remote OpenAI-compatible
llama.cpp server.  It does not know about scene/relation skills.
"""

from __future__ import annotations

import os
from typing import Any

import requests


REMOTE_VLM_URL = os.getenv(
    "REMOTE_VLM_URL",
    "http://192.168.50.37:8080/v1/chat/completions",
)

REMOTE_VLM_MODEL = os.getenv(
    "REMOTE_VLM_MODEL",
    "Qwen3-VL-8B-Instruct",
)


class RemoteVLMError(RuntimeError):
    """Remote VLM request failed."""


def chat_structured(
    *,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """
    Call the remote llama.cpp OpenAI-compatible endpoint.

    Return shape intentionally matches services.llm_service:
        {
            "role": "assistant",
            "content": "<JSON string>"
        }

    This keeps existing parser code reusable for local and remote VLMs.
    """

    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "messages 必須是非空 list"
        )

    if not isinstance(response_schema, dict):
        raise ValueError(
            "response_schema 必須是 dict"
        )

    payload = {
        "model":
            (
                model.strip()
                if isinstance(model, str)
                and model.strip()
                else REMOTE_VLM_MODEL
            ),

        "messages":
            messages,

        "temperature":
            float(temperature),

        "max_tokens":
            int(max_tokens),

        "stream":
            False,

        "response_format": {
            "type":
                "json_schema",

            "json_schema": {
                "name":
                    "vlm_response",

                "strict":
                    True,

                "schema":
                    response_schema,
            },
        },
    }

    try:
        response = requests.post(
            REMOTE_VLM_URL,
            json=payload,
            timeout=float(timeout),
        )
    except requests.RequestException as exc:
        raise RemoteVLMError(
            "無法連線到 remote VLM："
            f"{exc}"
        ) from exc

    if not response.ok:
        raise RemoteVLMError(
            "Remote VLM API 錯誤，"
            f"HTTP {response.status_code}："
            f"{response.text}"
        )

    try:
        raw = response.json()
    except ValueError as exc:
        raise RemoteVLMError(
            "Remote VLM 回傳內容不是有效 JSON"
        ) from exc

    choices = raw.get(
        "choices"
    )

    if (
        not isinstance(choices, list)
        or not choices
    ):
        raise RemoteVLMError(
            "Remote VLM response 缺少 choices"
        )

    message = (
        choices[0]
        .get(
            "message"
        )
    )

    if not isinstance(message, dict):
        raise RemoteVLMError(
            "Remote VLM response 缺少 message object"
        )

    content = message.get(
        "content"
    )

    if (
        not isinstance(content, str)
        or not content.strip()
    ):
        raise RemoteVLMError(
            "Remote VLM response 缺少 content"
        )

    return {
        "role":
            str(
                message.get(
                    "role"
                )
                or "assistant"
            ),

        "content":
            content,
    }


def health(
    *,
    timeout: float = 3.0,
) -> bool:
    """
    Optional connectivity check.

    It is never used to choose the backend automatically.
    """

    base_url = (
        REMOTE_VLM_URL
        .split(
            "/v1/chat/completions",
            1,
        )[0]
    )

    try:
        response = requests.get(
            f"{base_url}/health",
            timeout=float(timeout),
        )
    except requests.RequestException:
        return False

    return response.ok
