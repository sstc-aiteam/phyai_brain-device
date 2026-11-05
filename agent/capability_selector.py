"""Agent capability selection (no service execution).

This module uses the shared structured LLM provider and registered metadata.
It deliberately does NOT execute robot actions, camera requests, or Runtime.
The legacy agent.agent.route_command() remains untouched.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.capability_registry import get_capability, routing_catalog
from agent.providers.structured import chat_structured


class CapabilitySelectionError(RuntimeError):
    """Invalid capability selection or incompatible LLM output."""


_SKILL_PATH = Path(__file__).with_name("skills") / "capability_selection.md"


@lru_cache(maxsize=1)
def _load_skill() -> str:
    if not _SKILL_PATH.is_file():
        raise CapabilitySelectionError(f"缺少 Capability Selection Skill：{_SKILL_PATH}")
    content = _SKILL_PATH.read_text(encoding="utf-8").strip()
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) == 3:
            content = parts[2].strip()
    if not content:
        raise CapabilitySelectionError("capability_selection.md 不可為空")
    return content


def _response_schema(names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string", "enum": [*names, "none"]},
            "arguments": {"type": "object"},
            "reason": {"type": "string"},
        },
        "required": ["tool_name", "arguments", "reason"],
        "additionalProperties": False,
    }


def _validate_parameter(key: str, value: Any, rule: dict[str, Any]) -> None:
    """Validate the JSON Schema features currently used by the registry.

    Intentionally not a full JSON Schema implementation; no silent coercions.
    """
    expected = rule.get("type")
    if expected == "string":
        if not isinstance(value, str):
            raise CapabilitySelectionError(f"{key} 必須是 string")
        if len(value) < rule.get("minLength", 0):
            raise CapabilitySelectionError(f"{key} 長度不足")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CapabilitySelectionError(f"{key} 必須是 integer")
        if "minimum" in rule and value < rule["minimum"]:
            raise CapabilitySelectionError(f"{key} 小於允許下限")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CapabilitySelectionError(f"{key} 必須是 number")
    elif expected == "object":
        if not isinstance(value, dict):
            raise CapabilitySelectionError(f"{key} 必須是 object")
    elif expected == "array":
        if not isinstance(value, list):
            raise CapabilitySelectionError(f"{key} 必須是 array")
        if len(value) < rule.get("minItems", 0):
            raise CapabilitySelectionError(f"{key} 元素數量不足")
        if rule.get("uniqueItems") and len(value) != len(set(map(repr, value))):
            raise CapabilitySelectionError(f"{key} 不可包含重複項目")
        item_rule = rule.get("items")
        if isinstance(item_rule, dict):
            for index, item in enumerate(value):
                _validate_parameter(f"{key}[{index}]", item, item_rule)
    else:
        raise CapabilitySelectionError(f"Registry 不支援的參數型別：{key}={expected!r}")


def _validate_arguments(tool_name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise CapabilitySelectionError("arguments 必須是 JSON object")
    spec = get_capability(tool_name)
    schema = spec.parameters_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    unexpected = set(arguments) - set(properties)
    if unexpected:
        raise CapabilitySelectionError(
            f"{tool_name} 出現未註冊參數：{sorted(unexpected)}"
        )
    missing = required - set(arguments)
    if missing:
        raise CapabilitySelectionError(
            f"{tool_name} 缺少必要參數：{sorted(missing)}"
        )
    for key, value in arguments.items():
        _validate_parameter(key, value, properties[key])
    return dict(arguments)


def _reject_obvious_device_mismatch(user_text: str, tool_name: str) -> None:
    """Fail closed for *unambiguous equipment* mismatch; never reroute silently.

    This is a narrow safety check, not a full intent parser. If multiple device
    categories appear in the question, leave the decision to the LLM.
    """
    text = user_text.lower()
    vocabulary = {
        "camera": ("相機", "攝影機", "camera", "webcam", "鏡頭"),
        "arm": ("手臂", "機械手", "robot arm", "robotic arm"),
        "gripper": ("夾爪", "夾具", "gripper"),
    }
    explicit = {
        domain for domain, words in vocabulary.items()
        if any(word in text for word in words)
    }
    tools_by_domain = {
        "get_camera_status": "camera",
        "get_arm_status": "arm",
        "get_arm_pose": "arm",
        "get_arm_joints": "arm",
        "get_gripper_status": "gripper",
    }
    expected = tools_by_domain.get(tool_name)
    if len(explicit) == 1 and expected is not None and expected not in explicit:
        raise CapabilitySelectionError(
            f"Capability 類別與問題不符：使用者詢問 {next(iter(explicit))}，"
            f"但 LLM 選擇 {tool_name}；未執行任何 Service"
        )


def select_capability(
    user_text: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Select one registered capability; never invoke the selected service.

    `provider=None` uses `local`. Explicit `openai` remains available. No automatic provider fallback is used.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text 必須是非空字串")
    user_text = user_text.strip()

    effective_provider = "local" if provider is None else provider
    if effective_provider not in ("local", "openai"):
        raise CapabilitySelectionError(
            "Agent 目前只支援 provider='local' 或 provider='openai'"
        )

    catalog = routing_catalog()
    names = [row["name"] for row in catalog]
    message = chat_structured(
        stage="agent_capability_selection",
        messages=[
            {"role": "system", "content": _load_skill()},
            {
                "role": "user",
                "content": json.dumps(
                    {"user_text": user_text, "capabilities": catalog},
                    ensure_ascii=False,
                ),
            },
        ],
        response_schema=_response_schema(names),
        provider=effective_provider,
        model=model,
        temperature=0.0,
        max_tokens=512,
        num_ctx=4096,
        timeout=60.0,
        owner="agent_capability_selection",
        api_key=api_key,
    )

    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise CapabilitySelectionError("Structured Provider 回傳格式錯誤")
    try:
        selected = json.loads(message["content"])
    except json.JSONDecodeError as exc:
        raise CapabilitySelectionError(f"LLM 回傳無效 JSON：{exc}") from exc
    if not isinstance(selected, dict):
        raise CapabilitySelectionError("LLM 結果必須是 JSON object")
    if set(selected) != {"tool_name", "arguments", "reason"}:
        raise CapabilitySelectionError("LLM 結果欄位不符合能力選擇 Schema")

    tool_name = selected["tool_name"]
    reason = selected["reason"]
    if not isinstance(tool_name, str) or tool_name not in {*names, "none"}:
        raise CapabilitySelectionError(f"未知 Capability：{tool_name!r}")
    if not isinstance(reason, str):
        raise CapabilitySelectionError("reason 必須是 string")
    if tool_name == "none":
        if selected["arguments"] != {}:
            raise CapabilitySelectionError("tool_name='none' 時 arguments 必須是空物件")
        return {
            "tool_name": "none",
            "arguments": {},
            "reason": reason,
            "execution_enabled": False,
            "side_effect": "none",
        }

    _reject_obvious_device_mismatch(user_text, tool_name)

    # Robot task text must be the original command, not an LLM paraphrase.
    arguments = selected["arguments"]
    if not isinstance(arguments, dict):
        raise CapabilitySelectionError("arguments 必須是 JSON object")
    arguments = dict(arguments)
    if tool_name in ("run_manual_runtime", "plan_single_action"):
        arguments["user_text"] = user_text
    arguments = _validate_arguments(tool_name, arguments)
    spec = get_capability(tool_name)
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "reason": reason,
        "execution_enabled": spec.execution_enabled,
        "side_effect": spec.side_effect,
    }


__all__ = ["CapabilitySelectionError", "select_capability"]
