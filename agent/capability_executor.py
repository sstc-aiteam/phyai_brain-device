"""Execute a selected Agent capability through its registered Service.

This is intentionally separate from LLM selection. The Registry, not an LLM
response, is authoritative for the Service target and execution permission.

Step 3 only executes read-only information capabilities. Robot-moving Runtime
is unconditionally blocked here, even if a selection payload is forged.
"""
from __future__ import annotations

import importlib
from typing import Any

from agent.capability_registry import get_capability
from agent.capability_selector import CapabilitySelectionError, _validate_arguments


class CapabilityExecutionError(RuntimeError):
    """The selection or selected Service could not be executed correctly."""


class CapabilityExecutionBlocked(CapabilityExecutionError):
    """Execution denied by the registry / actuator safety boundary."""


def execute_capability(selection: dict[str, Any]) -> dict[str, Any]:
    """Call ONE registered information Service selected by the Agent.

    Parameters
    ----------
    selection:
        Result of ``select_capability()`` (tool_name + arguments). Caller-
        supplied permission fields, if any, are intentionally NOT trusted.

    Returns
    -------
    dict:
        ``tool_name``, ``executed``, ``arguments`` and the raw
        ``service_response``. No natural-language answer is invented here.

    Note
    ----
    This function does not start Manual Runtime, invoke an actuator, or call
    the LLM. The future manual execution path needs a separate authorization
    and session lifecycle integration.
    """
    if not isinstance(selection, dict):
        raise CapabilityExecutionError("selection 必須是 dict")

    tool_name = selection.get("tool_name")
    arguments = selection.get("arguments")
    if not isinstance(tool_name, str) or not tool_name:
        raise CapabilityExecutionError("缺少有效 tool_name")
    if not isinstance(arguments, dict):
        raise CapabilityExecutionError("arguments 必須是 dict")

    if tool_name == "none":
        if arguments:
            raise CapabilityExecutionError("tool_name='none' 時 arguments 必須為空")
        return {
            "tool_name": "none",
            "executed": False,
            "arguments": {},
            "service_response": None,
        }

    try:
        spec = get_capability(tool_name)
    except KeyError as exc:
        raise CapabilityExecutionError(
            f"未註冊的 Capability：{tool_name!r}"
        ) from exc

    # Do NOT trust selection['execution_enabled'] or selection['side_effect'].
    # Enforce both gates before importing the Service module.
    if not spec.execution_enabled or spec.side_effect != "none":
        raise CapabilityExecutionBlocked(
            f"Capability 禁止由資訊查詢執行層呼叫：{tool_name} "
            f"(execution_enabled={spec.execution_enabled}, "
            f"side_effect={spec.side_effect!r})"
        )

    try:
        checked_arguments = _validate_arguments(tool_name, arguments)
    except CapabilitySelectionError as exc:
        raise CapabilityExecutionError(f"Capability 參數驗證失敗：{exc}") from exc

    # Confirmed Service signature: get_object_state(*, camera_names: tuple).
    # JSON/LLM selection uses arrays; preserve Service's Python contract.
    call_arguments = dict(checked_arguments)
    if tool_name == "get_object_state" and "camera_names" in call_arguments:
        call_arguments["camera_names"] = tuple(call_arguments["camera_names"])

    try:
        service = importlib.import_module(spec.service_module)
        function = getattr(service, spec.service_function)
        if not callable(function):
            raise TypeError(f"{spec.service_function} 不是可呼叫函式")
        service_response = function(**call_arguments)
    except Exception as exc:
        raise CapabilityExecutionError(
            f"{tool_name} Service 呼叫失敗：{type(exc).__name__}: {exc}"
        ) from exc

    # These three confirmed Services use utils.response's result/data envelope.
    # A disconnected arm within a successful status query is NOT an API error.
    if not isinstance(service_response, dict):
        raise CapabilityExecutionError(f"{tool_name} Service 回傳值不是 dict")
    if service_response.get("result") is not True:
        raise CapabilityExecutionError(
            f"{tool_name} Service 回報失敗："
            f"{service_response.get('message') or service_response.get('error') or service_response!r}"
        )

    return {
        "tool_name": tool_name,
        "executed": True,
        "arguments": checked_arguments,
        "service_response": service_response,
    }


__all__ = [
    "CapabilityExecutionError",
    "CapabilityExecutionBlocked",
    "execute_capability",
]
