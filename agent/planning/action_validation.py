"""Deterministic validation for Stage 1 atomic action extraction.

This module validates tool arguments, duplicate subplans and high-confidence
coverage of explicit user actions. It does not call an LLM or compile a DAG.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from math import isclose
from typing import Any



class SemanticIRError(ValueError):
    """Raised when semantic parser output is inconsistent or unsafe."""

    def __init__(self, code: str, message: str, path: str | None = None):
        self.code = code
        self.path = path
        super().__init__(message)


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
        "null": value is None,
    }.get(expected, False)


def _validate_value(value, definition, path):
    expected = definition.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]
    if not expected_types or not any(_matches_type(value, item) for item in expected_types):
        raise SemanticIRError("INVALID_ARGUMENT_TYPE", f"{path} 參數型別錯誤", path)
    if value is None:
        return
    if "enum" in definition and value not in definition["enum"]:
        raise SemanticIRError("INVALID_ARGUMENT_VALUE", f"{path} 不在允許範圍", path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise SemanticIRError("INVALID_ARGUMENT_VALUE", f"{path} 必須是有限數值", path)
        if "minimum" in definition and value < definition["minimum"]:
            raise SemanticIRError("ARGUMENT_OUT_OF_RANGE", f"{path} 小於最小值", path)
        if "maximum" in definition and value > definition["maximum"]:
            raise SemanticIRError("ARGUMENT_OUT_OF_RANGE", f"{path} 超過最大值", path)
    if isinstance(value, list):
        if len(value) < definition.get("minItems", 0):
            raise SemanticIRError("INVALID_ARGUMENT_LENGTH", f"{path} 項目數不足", path)
        if "maxItems" in definition and len(value) > definition["maxItems"]:
            raise SemanticIRError("INVALID_ARGUMENT_LENGTH", f"{path} 項目數過多", path)
        if isinstance(definition.get("items"), dict):
            for index, item in enumerate(value):
                _validate_value(item, definition["items"], f"{path}[{index}]")


def _validate_arguments(action, tool, path):
    arguments = action["arguments"]
    parameters = tool.get("parameters", {})
    required = tool.get("required", [])
    unknown = set(arguments) - set(parameters)
    if unknown:
        raise SemanticIRError(
            "UNKNOWN_ARGUMENT", f"{path} 包含未授權參數：{sorted(unknown)}", path
        )
    missing = [name for name in required if arguments.get(name) is None]
    if missing:
        raise SemanticIRError(
            "MISSING_ARGUMENT", f"{path} 缺少必要參數：{missing}", path
        )
    for name, value in arguments.items():
        _validate_value(value, parameters[name], f"{path}.{name}")


def _detect_duplicated_subplan(actions):
    signatures = [
        (action["function_name"], json.dumps(action["arguments"], sort_keys=True))
        for action in actions
    ]
    count = len(signatures)
    if count >= 4 and count % 2 == 0 and signatures[: count // 2] == signatures[count // 2 :]:
        raise SemanticIRError(
            "DUPLICATED_SUBPLAN",
            "同一完整 action sequence 被重複產生；除非使用者明確要求，不可複製整份任務",
            "actions",
        )


def validate_atomic_action_output(
    output: Any,
    tool_catalog: list[dict[str, Any]],
    *,
    allow_duplicated_subplan: bool = False,
) -> list[dict[str, Any]]:
    """Validate Stage 1 output and assign deterministic local action keys."""
    if not isinstance(output, dict) or set(output) != {"actions"}:
        raise SemanticIRError(
            "INVALID_ACTION_OUTPUT",
            "Stage 1 輸出只能包含 actions",
            "actions",
        )
    actions = output["actions"]
    if not isinstance(actions, list) or not actions:
        raise SemanticIRError("INVALID_ACTIONS", "actions 必須是非空陣列", "actions")
    tools = {
        item.get("api_function"): item
        for item in tool_catalog
        if isinstance(item, dict) and isinstance(item.get("api_function"), str)
    }
    normalized = []
    for index, raw_action in enumerate(actions):
        path = f"actions[{index}]"
        if not isinstance(raw_action, dict) or set(raw_action) != {"function_name", "arguments"}:
            raise SemanticIRError(
                "INVALID_ACTION",
                f"{path} 只能包含 function_name 與 arguments",
                path,
            )
        function_name = raw_action["function_name"]
        arguments = raw_action["arguments"]
        if function_name not in tools:
            raise SemanticIRError("UNKNOWN_TOOL", f"{path} 使用不存在工具：{function_name}", path)
        if not isinstance(arguments, dict):
            raise SemanticIRError("INVALID_ARGUMENTS", f"{path}.arguments 必須是 object", path)
        action = {
            "key": f"a{index + 1}",
            "function_name": function_name,
            "arguments": deepcopy(arguments),
        }
        _validate_arguments(action, tools[function_name], f"{path}.arguments")
        normalized.append(action)
    if not allow_duplicated_subplan:
        _detect_duplicated_subplan(normalized)
    return normalized


_ENTITY_REFERENCE_ARGUMENTS = {
    "object_id",
    "source_id",
    "target_id",
    "destination_id",
    "payload_id",
    "container_id",
    "location_id",
    "waypoint_id",
}


def _collect_known_entity_ids(context):
    if not isinstance(context, dict):
        return set()

    known = set()

    # Explicit deterministic grounding.
    for row in context.get("grounded_targets") or []:
        if not isinstance(row, dict):
            continue

        entity_id = row.get("object_id") or row.get("id")

        if isinstance(entity_id, str) and entity_id.strip():
            known.add(entity_id.strip())

    # Structured scene snapshot.
    for snapshot_key in ("scene_snapshot", "fake_scene_snapshot"):
        snapshot = context.get(snapshot_key)

        if not isinstance(snapshot, dict):
            continue

        for row in snapshot.get("entities") or []:
            if not isinstance(row, dict):
                continue

            entity_id = row.get("id") or row.get("object_id")

            if isinstance(entity_id, str) and entity_id.strip():
                known.add(entity_id.strip())

    # YOLO / structured detections fallback.
    for yolo_key in ("yolo", "fake_yolo"):
        yolo = context.get(yolo_key)

        if not isinstance(yolo, dict):
            continue

        for row in yolo.get("detections") or []:
            if not isinstance(row, dict):
                continue

            entity_id = row.get("object_id") or row.get("id")

            if isinstance(entity_id, str) and entity_id.strip():
                known.add(entity_id.strip())

    return known


def validate_grounded_action_targets(actions, context):
    """Reject scene entity IDs invented by Stage1.

    Entity-reference arguments may refer to any entity that exists in the
    deterministic grounding / structured scene.  Do not require every
    grounded mention to become a placement action.
    """
    known_entity_ids = _collect_known_entity_ids(context)

    if not known_entity_ids:
        return

    for action_index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue

        function_name = str(action.get("function_name") or "")
        arguments = action.get("arguments")

        if not isinstance(arguments, dict):
            continue

        for argument_name, value in arguments.items():
            if argument_name not in _ENTITY_REFERENCE_ARGUMENTS:
                continue

            if not isinstance(value, str) or not value.strip():
                continue

            if value not in known_entity_ids:
                raise SemanticIRError(
                    "UNGROUNDED_OBJECT_ID",
                    (
                        f"{function_name}.{argument_name} 使用不存在於 "
                        f"grounded scene 的 entity ID：{value!r}；"
                        f"known={sorted(known_entity_ids)}"
                    ),
                    f"actions[{action_index}].arguments.{argument_name}",
                )



def _collect_initially_held_object_ids(context: Any) -> set[str]:
    """Collect optional planner-known held objects without touching hardware."""
    if not isinstance(context, dict):
        return set()

    held: set[str] = set()

    raw_ids = context.get("initially_held_object_ids")
    if isinstance(raw_ids, (list, tuple, set)):
        for object_id in raw_ids:
            if isinstance(object_id, str) and object_id.strip():
                held.add(object_id.strip())

    resource_state = context.get("resource_holding_state")
    if isinstance(resource_state, dict):
        for object_id in resource_state.values():
            if isinstance(object_id, str) and object_id.strip():
                held.add(object_id.strip())

    return held


def validate_manipulation_preconditions(
    actions: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> None:
    """Validate deterministic acquire/release semantics.

    pick_object(X) acquires X.
    place_object(X) requires X to already be held, then releases X.

    This is tool-level planning semantics only. Device ownership continuity is
    handled later by the compiler/allocator, while end-effector capacity is
    handled by the scheduler.
    """
    held_objects = _collect_initially_held_object_ids(context)

    for action_index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue

        function_name = action.get("function_name")
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            continue

        object_id = arguments.get("object_id")
        if not isinstance(object_id, str) or not object_id.strip():
            continue
        object_id = object_id.strip()

        if function_name == "pick_object":
            held_objects.add(object_id)
            continue

        if function_name != "place_object":
            continue

        if object_id not in held_objects:
            raise SemanticIRError(
                "UNSATISFIED_ACTION_PRECONDITION",
                (
                    f"place_object({object_id!r}) 需要該物件先處於 held 狀態；"
                    f"目前 actions 中沒有先前的 pick_object({object_id!r})，"
                    "且 task-start context 也未標記為已持有"
                ),
                f"actions[{action_index}]",
            )

        held_objects.remove(object_id)


def _available_tools(tool_catalog):
    return {
        tool.get("api_function")
        for tool in tool_catalog
        if isinstance(tool, dict) and isinstance(tool.get("api_function"), str)
    }


def derive_explicit_action_expectations(user_text, tool_catalog):
    """Return only actions stated with an unambiguous surface form."""
    compact = re.sub(r"\s+", "", user_text.lower())
    available = _available_tools(tool_catalog)
    expected = []

    phrase_rules = (
        ("open_trash_can", r"(?:打開|開啟)垃圾桶"),
        ("close_trash_can", r"(?:關閉|關上|關)垃圾桶"),
        ("move_arm_default", r"(?:回到|回|移動到)(?:手臂)?(?:的)?(?:預設位置|預設姿態|待命姿態)"),
    )
    for function_name, pattern in phrase_rules:
        if function_name not in available:
            continue
        expected.extend(
            {"function_name": function_name, "arguments": {}}
            for _ in re.finditer(pattern, compact)
        )

    if "move_arm_step" in available:
        direction_pattern = re.compile(
            r"([xyz])(?:軸)?(?:的)?(正|負|\+|-)(?:向|方向)?(?:移動|運動)"
            r"(?:約)?(?:(\d+(?:\.\d+)?)(公尺|米|公分|厘米|cm|m)?)?"
        )
        for match in direction_pattern.finditer(compact):
            sign = "+" if match.group(2) in {"正", "+"} else "-"
            arguments = {"direction": f"{match.group(1)}{sign}"}
            if match.group(3) is not None:
                distance = float(match.group(3))
                if match.group(4) in {"公分", "厘米", "cm"}:
                    distance /= 100.0
                arguments["distance"] = distance
            expected.append({
                "function_name": "move_arm_step",
                "arguments": arguments,
            })

    # Exact function names in technical prompts are also unambiguous.
    for function_name in sorted(available):
        literal_count = compact.count(function_name.lower())
        already_count = sum(
            item["function_name"] == function_name for item in expected
        )
        for _ in range(max(0, literal_count - already_count)):
            expected.append({"function_name": function_name, "arguments": {}})
    return expected


def validate_semantic_action_coverage(user_text: str, ir: Any, tool_catalog):
    """Reject IR that omits an explicitly requested, high-confidence action."""
    if not isinstance(ir, dict) or not isinstance(ir.get("actions"), list):
        return
    expected = derive_explicit_action_expectations(user_text, tool_catalog)
    actual = ir["actions"]

    expected_default_count = sum(
        item["function_name"] == "move_arm_default"
        for item in expected
    )
    actual_default_count = sum(
        isinstance(item, dict)
        and item.get("function_name") == "move_arm_default"
        for item in actual
    )
    if actual_default_count > expected_default_count:
        raise SemanticIRError(
            "UNREQUESTED_ACTION",
            "使用者未要求回預設／待命姿態，不可新增 move_arm_default",
            "actions",
        )

    def value_matches(expected_value, actual_value):
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            return isclose(float(expected_value), float(actual_value), rel_tol=1e-9, abs_tol=1e-9)
        return expected_value == actual_value

    def action_matches(expected_action, actual_action):
        if not isinstance(actual_action, dict):
            return False
        if expected_action["function_name"] != actual_action.get("function_name"):
            return False
        actual_arguments = actual_action.get("arguments")
        if not isinstance(actual_arguments, dict):
            return False
        return all(
            key in actual_arguments and value_matches(value, actual_arguments[key])
            for key, value in expected_action["arguments"].items()
        )

    # Match the most specific expectations first. This prevents a generic literal
    # function-name expectation from consuming an action that must also carry an
    # explicitly requested direction/distance.
    remaining = list(actual)
    missing = []
    for expected_action in sorted(
        expected, key=lambda item: len(item["arguments"]), reverse=True
    ):
        match_index = next(
            (
                index for index, actual_action in enumerate(remaining)
                if action_matches(expected_action, actual_action)
            ),
            None,
        )
        if match_index is not None:
            remaining.pop(match_index)
            continue
        arguments = expected_action["arguments"]
        details = ", ".join(f"{key}={value}" for key, value in arguments.items())
        label = expected_action["function_name"]
        if details:
            label += f"({details})"
        missing.append(label)
    if missing:
        raise SemanticIRError(
            "MISSING_EXPLICIT_ACTION",
            f"semantic IR 漏掉使用者明確要求的 actions：{missing}",
            "actions",
        )
