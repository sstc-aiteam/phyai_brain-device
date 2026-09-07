"""Single in-process gateway for the project's local Ollama runtime."""

from __future__ import annotations

import base64
import json
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import requests
import config

MODULE = "llm"
logger = logging.getLogger(__name__)
_LLM_LOCK = threading.RLock()
_OWNER_STATE_LOCK = threading.Lock()
_ACTIVE_OWNER: str | None = None


class LLMServiceError(RuntimeError):
    """The local model request failed."""


class LLMBusyError(LLMServiceError):
    """A non-blocking consumer could not obtain the local runtime."""


def _get_base_url() -> str:
    value = getattr(config, "OLLAMA_BASE_URL", None)
    if not isinstance(value, str) or not value.strip():
        raise LLMServiceError("OLLAMA_BASE_URL 未正確設定")
    value = value.strip().rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def _get_default_model() -> str:
    value = getattr(config, "OLLAMA_MODEL", None)
    if not isinstance(value, str) or not value.strip():
        raise LLMServiceError("OLLAMA_MODEL 未正確設定")
    return value.strip()


def _positive_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMServiceError(f"{name} 必須是數值") from exc
    if number <= 0:
        raise LLMServiceError(f"{name} 必須大於 0")
    return number


def _get_default_timeout() -> float:
    return _positive_number(getattr(config, "OLLAMA_TIMEOUT", 60), "OLLAMA_TIMEOUT")


def _get_default_temperature() -> float:
    try:
        return float(getattr(config, "OLLAMA_TEMPERATURE", 0.0))
    except (TypeError, ValueError) as exc:
        raise LLMServiceError("OLLAMA_TEMPERATURE 必須是數值") from exc


def _image_data(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image_url.url 必須是非空字串")
    value = value.strip()
    if value.startswith("data:"):
        if "," not in value:
            raise ValueError("image data URL 格式錯誤")
        return value.split(",", 1)[1]
    try:
        base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("image_url 必須是 data URL 或 base64") from exc
    return value


def _validate_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必須是非空 list")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] 必須是 dict")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"messages[{index}].role 必須是非空字串")
        row: dict[str, Any] = {"role": role.strip()}
        if isinstance(content, str):
            row["content"] = content
        elif isinstance(content, list):
            texts, images = [], []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError(f"messages[{index}].content item 必須是 dict")
                if part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    images.append(_image_data(part.get("image_url")))
                else:
                    raise ValueError(f"不支援的 multimodal content type：{part.get('type')!r}")
            row["content"] = "\n".join(texts)
            if images:
                row["images"] = images
        else:
            raise ValueError(f"messages[{index}].content 必須是字串或 list")
        normalized.append(row)
    return normalized


@contextmanager
def runtime_session(*, wait: bool = True, owner: str = "anonymous") -> Iterator[None]:
    """Lease the local runtime. RLock permits nested calls in one planner lease."""
    global _ACTIVE_OWNER
    if not _LLM_LOCK.acquire(blocking=bool(wait)):
        raise LLMBusyError(f"local LLM runtime busy (owner={_ACTIVE_OWNER or 'unknown'})")
    with _OWNER_STATE_LOCK:
        previous_owner, _ACTIVE_OWNER = _ACTIVE_OWNER, str(owner or "anonymous")
    try:
        yield
    finally:
        with _OWNER_STATE_LOCK:
            _ACTIVE_OWNER = previous_owner
        _LLM_LOCK.release()


def runtime_busy() -> bool:
    acquired = _LLM_LOCK.acquire(blocking=False)
    if acquired:
        _LLM_LOCK.release()
        return False
    return True


def _raise_request_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("error", response.text)
    except ValueError:
        detail = response.text
    raise LLMServiceError(f"Ollama API 錯誤，HTTP {response.status_code}：{detail}")


def _ollama_chat(messages: Any, *, model: str | None = None,
                 response_schema: dict[str, Any] | None = None,
                 temperature: float | None = None, top_p: float | None = None,
                 max_tokens: int | None = None, timeout: float | None = None,
                 keep_alive: int | str | None = None,
                 stream_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": _get_default_temperature() if temperature is None else float(temperature)}
    if top_p is not None:
        options["top_p"] = float(top_p)
    if max_tokens is not None:
        options["num_predict"] = int(max_tokens)
    streaming = callable(stream_callback)
    payload: dict[str, Any] = {
        "model": model.strip() if isinstance(model, str) and model.strip() else _get_default_model(),
        "messages": _validate_messages(messages), "stream": streaming, "options": options,
    }
    if response_schema is not None:
        if not isinstance(response_schema, dict):
            raise ValueError("response_schema 必須是 dict")
        payload["format"] = response_schema
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    try:
        response = requests.post(f"{_get_base_url()}/api/chat", json=payload,
                                 timeout=_get_default_timeout() if timeout is None else _positive_number(timeout, "timeout"),
                                 stream=streaming)
    except requests.RequestException as exc:
        raise LLMServiceError(f"無法連線到 Ollama：{exc}") from exc
    if not response.ok:
        _raise_request_error(response)
    if not streaming:
        try:
            message = response.json().get("message")
        except ValueError as exc:
            raise LLMServiceError("Ollama 回傳內容不是有效 JSON") from exc
        if not isinstance(message, dict):
            raise LLMServiceError("Ollama response 缺少 message object")
        return message
    parts: list[str] = []
    try:
        for line in response.iter_lines():
            if not line:
                continue
            piece = (json.loads(line).get("message") or {}).get("content")
            if piece:
                parts.append(str(piece))
                stream_callback(str(piece))
    except (ValueError, requests.RequestException) as exc:
        raise LLMServiceError(f"Ollama streaming response 無效：{exc}") from exc
    return {"role": "assistant", "content": "".join(parts)}


def chat(messages: Any, model: str | None = None, temperature: float | None = None,
         timeout: float | None = None, *, top_p: float | None = None,
         max_tokens: int | None = None, keep_alive: int | str | None = None,
         stream_callback: Callable[[str], None] | None = None,
         wait: bool = True, owner: str = "llm_chat") -> dict[str, Any]:
    with runtime_session(wait=wait, owner=owner):
        return _ollama_chat(messages, model=model, temperature=temperature, top_p=top_p,
                            max_tokens=max_tokens, timeout=timeout, keep_alive=keep_alive,
                            stream_callback=stream_callback)


def chat_structured(messages: Any, response_schema: dict[str, Any], model: str | None = None,
                    temperature: float | None = None, timeout: float | None = None, *,
                    top_p: float | None = None, max_tokens: int | None = None,
                    keep_alive: int | str | None = None, wait: bool = True,
                    owner: str = "llm_structured") -> dict[str, Any]:
    with runtime_session(wait=wait, owner=owner):
        return _ollama_chat(messages, model=model, response_schema=response_schema,
                            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                            timeout=timeout, keep_alive=keep_alive)


def keepalive(*, model: str | None = None, keep_alive: int | str = -1,
              timeout: float | None = None, wait: bool = False,
              owner: str = "llm_keepalive") -> dict[str, Any]:
    with runtime_session(wait=wait, owner=owner):
        payload = {"model": model or _get_default_model(), "prompt": "", "stream": False,
                   "keep_alive": keep_alive}
        try:
            response = requests.post(f"{_get_base_url()}/api/generate", json=payload,
                                     timeout=timeout or _get_default_timeout())
        except requests.RequestException as exc:
            raise LLMServiceError(f"無法連線到 Ollama：{exc}") from exc
        if not response.ok:
            _raise_request_error(response)
        try:
            return response.json()
        except ValueError as exc:
            raise LLMServiceError("Ollama keepalive 回傳不是有效 JSON") from exc
