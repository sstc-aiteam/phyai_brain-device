"""
LLM 共用服務層。

所有需要使用 LLM 的模組，
一律透過此 service 呼叫。

目前 backend 使用 Ollama。
"""

import logging
import threading

import requests

import config


MODULE = "llm"

logger = logging.getLogger(__name__)


# ============================================================
# LLM Runtime Lock
# ============================================================

_LLM_LOCK = threading.RLock()


# ============================================================
# Exceptions
# ============================================================

class LLMServiceError(
    RuntimeError
):
    """
    LLM service 執行錯誤。
    """


# ============================================================
# Config Helpers
# ============================================================

def _get_base_url():
    base_url = getattr(
        config,
        "OLLAMA_BASE_URL",
        None,
    )

    if (
        not isinstance(
            base_url,
            str,
        )
        or not base_url.strip()
    ):
        raise LLMServiceError(
            "OLLAMA_BASE_URL "
            "未正確設定"
        )

    return (
        base_url
        .strip()
        .rstrip("/")
    )


def _get_default_model():
    model = getattr(
        config,
        "OLLAMA_MODEL",
        None,
    )

    if (
        not isinstance(
            model,
            str,
        )
        or not model.strip()
    ):
        raise LLMServiceError(
            "OLLAMA_MODEL "
            "未正確設定"
        )

    return model.strip()


def _get_default_timeout():
    timeout = getattr(
        config,
        "OLLAMA_TIMEOUT",
        60,
    )

    try:
        timeout = float(
            timeout
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise LLMServiceError(
            "OLLAMA_TIMEOUT "
            "必須是數值"
        ) from exc

    if timeout <= 0:
        raise LLMServiceError(
            "OLLAMA_TIMEOUT "
            "必須大於 0"
        )

    return timeout


def _get_default_temperature():
    temperature = getattr(
        config,
        "OLLAMA_TEMPERATURE",
        0.0,
    )

    try:
        return float(
            temperature
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise LLMServiceError(
            "OLLAMA_TEMPERATURE "
            "必須是數值"
        ) from exc


# ============================================================
# Validation
# ============================================================

def _validate_messages(
    messages,
):
    if not isinstance(
        messages,
        list,
    ):
        raise ValueError(
            "messages 必須是 list"
        )

    if not messages:
        raise ValueError(
            "messages 不可為空"
        )

    normalized = []

    for index, message in enumerate(
        messages
    ):
        if not isinstance(
            message,
            dict,
        ):
            raise ValueError(
                f"messages[{index}] "
                "必須是 dict"
            )

        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )

        if (
            not isinstance(
                role,
                str,
            )
            or not role.strip()
        ):
            raise ValueError(
                f"messages[{index}].role "
                "必須是非空字串"
            )

        if not isinstance(
            content,
            str,
        ):
            raise ValueError(
                f"messages[{index}].content "
                "必須是字串"
            )

        normalized.append(
            {
                "role":
                    role.strip(),

                "content":
                    content,
            }
        )

    return normalized


# ============================================================
# Ollama Backend
# ============================================================

def _ollama_chat(
    messages,
    model=None,
    response_schema=None,
    temperature=None,
    timeout=None,
):
    messages = (
        _validate_messages(
            messages
        )
    )

    if model is None:
        model = (
            _get_default_model()
        )

    if (
        not isinstance(
            model,
            str,
        )
        or not model.strip()
    ):
        raise ValueError(
            "model 必須是非空字串"
        )

    model = model.strip()

    if temperature is None:
        temperature = (
            _get_default_temperature()
        )

    if timeout is None:
        timeout = (
            _get_default_timeout()
        )

    url = (
        f"{_get_base_url()}"
        "/api/chat"
    )

    payload = {
        "model":
            model,

        "messages":
            messages,

        "stream":
            False,

        "options": {
            "temperature":
                float(
                    temperature
                ),
        },
    }

    if response_schema is not None:

        if not isinstance(
            response_schema,
            dict,
        ):
            raise ValueError(
                "response_schema "
                "必須是 dict"
            )

        payload[
            "format"
        ] = response_schema

    try:

        response = (
            requests.post(
                url,
                json=payload,
                timeout=float(
                    timeout
                ),
            )
        )

    except (
        requests.RequestException
    ) as exc:

        raise LLMServiceError(
            "無法連線到 Ollama："
            f"{exc}"
        ) from exc

    if not response.ok:

        try:

            error_data = (
                response.json()
            )

            error_message = (
                error_data.get(
                    "error",
                    response.text,
                )
            )

        except ValueError:

            error_message = (
                response.text
            )

        raise LLMServiceError(
            "Ollama API 錯誤，"
            f"HTTP "
            f"{response.status_code}："
            f"{error_message}"
        )

    try:

        body = (
            response.json()
        )

    except ValueError as exc:

        raise LLMServiceError(
            "Ollama 回傳內容"
            "不是有效 JSON"
        ) from exc

    message = (
        body.get(
            "message"
        )
    )

    if not isinstance(
        message,
        dict,
    ):
        raise LLMServiceError(
            "Ollama response "
            "缺少 message object"
        )

    return message


# ============================================================
# Public API
# ============================================================

def chat(
    messages,
    model=None,
    temperature=None,
    timeout=None,
):
    """
    一般文字 LLM 對話。

    回傳 Ollama message：
    {
        "role": "...",
        "content": "..."
    }
    """

    with _LLM_LOCK:

        return _ollama_chat(
            messages=
                messages,

            model=
                model,

            temperature=
                temperature,

            timeout=
                timeout,
        )


def chat_structured(
    messages,
    response_schema,
    model=None,
    temperature=None,
    timeout=None,
):
    """
    Structured Output。

    透過 JSON Schema
    限制模型輸出格式。
    """

    with _LLM_LOCK:

        return _ollama_chat(
            messages=
                messages,

            model=
                model,

            response_schema=
                response_schema,

            temperature=
                temperature,

            timeout=
                timeout,
        )