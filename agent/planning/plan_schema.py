"""通用任務計畫 Schema 與確定性驗證。

本模組定義 requirements、roles、steps、dependencies、constraints
與 assignment_preferences 的最小穩定格式，並在基礎 Schema 通過後
比對 Tool Catalog 與 Device Registry。它不操作硬體或執行計畫。
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = "1.0"
TOP_LEVEL_FIELDS = {
    "schema_version",
    "requirements",
    "roles",
    "steps",
    "constraints",
    "assignment_preferences",
}
BINDING_MODES = {"automatic", "preferred", "required"}
DEPENDENCY_CONDITIONS = {"completed", "succeeded"}
CONSTRAINT_TYPES = {"same_assignment", "distinct_assignment"}
PREFERENCE_TYPES = {
    "prefer_same_assignment",
    "prefer_distinct_assignment",
}
CONSTRAINT_FIELDS = {"id", "type", "roles", "hard", "source"}
PREFERENCE_FIELDS = {"id", "type", "roles", "weight", "reason", "source"}
BINDING_FIELDS = {"mode", "target_id", "source"}


class PlanValidationError(ValueError):
    """Raised when an abstract plan violates the deterministic schema."""

    def __init__(self, code: str, message: str, path: str | None = None):
        self.code = code
        self.path = path
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "message": str(self)}
        if self.path is not None:
            result["path"] = self.path
        return result


def _error(code: str, message: str, path: str | None = None):
    raise PlanValidationError(code, message, path)


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _error("INVALID_TYPE", f"{path} 必須是 list", path)
    return value


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _error("INVALID_TYPE", f"{path} 必須是 object", path)
    return value


def _require_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error("INVALID_ID", f"{path} 必須是非空字串", path)
    return value.strip()


def _collect_unique_ids(items: list[Any], section: str) -> dict[str, dict[str, Any]]:
    collected = {}
    for index, raw_item in enumerate(items):
        path = f"{section}[{index}]"
        item = _require_object(raw_item, path)
        item_id = _require_id(item.get("id"), f"{path}.id")
        if item_id in collected:
            _error(
                "DUPLICATE_ID",
                f"{section} 的 id 重複：{item_id}",
                f"{path}.id",
            )
        collected[item_id] = item
    return collected


def _normalize_unique_strings(value: Any, path: str) -> list[str]:
    items = _require_list(value, path)
    normalized = []
    seen = set()
    for index, item in enumerate(items):
        item = _require_id(item, f"{path}[{index}]")
        if item in seen:
            _error("DUPLICATE_VALUE", f"{path} 不可重複：{item}", path)
        seen.add(item)
        normalized.append(item)
    return normalized


def _normalize_role_relation(
    value: Any,
    path: str,
    role_ids: set[str],
) -> list[str]:
    roles = _normalize_unique_strings(value, path)
    if len(roles) < 2:
        _error("INSUFFICIENT_ROLES", f"{path} 至少需要兩個 role", path)
    unknown_roles = set(roles) - role_ids
    if unknown_roles:
        _error(
            "UNKNOWN_ROLE",
            f"{path} 引用不存在的 role：{sorted(unknown_roles)}",
            path,
        )
    return roles


def _normalize_dependencies(raw_dependencies: Any, path: str) -> list[dict[str, str]]:
    dependencies = _require_list(raw_dependencies, path)
    normalized = []
    seen = set()
    for index, raw_dependency in enumerate(dependencies):
        dependency_path = f"{path}[{index}]"
        if isinstance(raw_dependency, str):
            step_id = _require_id(raw_dependency, dependency_path)
            condition = "completed"
        else:
            dependency = _require_object(raw_dependency, dependency_path)
            unknown = set(dependency) - {"step_id", "condition"}
            if unknown:
                _error(
                    "UNKNOWN_FIELD",
                    f"{dependency_path} 包含未支援欄位：{sorted(unknown)}",
                    dependency_path,
                )
            step_id = _require_id(
                dependency.get("step_id"),
                f"{dependency_path}.step_id",
            )
            condition = dependency.get("condition", "completed")
            if condition not in DEPENDENCY_CONDITIONS:
                _error(
                    "INVALID_DEPENDENCY_CONDITION",
                    f"{dependency_path}.condition 不合法：{condition}",
                    f"{dependency_path}.condition",
                )
        key = (step_id, condition)
        if key in seen:
            _error(
                "DUPLICATE_DEPENDENCY",
                f"{path} 重複依賴步驟 {step_id}",
                dependency_path,
            )
        seen.add(key)
        normalized.append({"step_id": step_id, "condition": condition})
    return normalized


def _validate_cycle(step_dependencies: dict[str, set[str]]) -> None:
    visiting = set()
    visited = set()

    def visit(step_id: str):
        if step_id in visiting:
            _error("DEPENDENCY_CYCLE", "步驟依賴形成循環", "steps")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency_id in step_dependencies[step_id]:
            visit(dependency_id)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in step_dependencies:
        visit(step_id)


def validate_plan(plan: Any) -> dict[str, Any]:
    """Validate and return a normalized deep copy of an abstract plan."""
    normalized = deepcopy(_require_object(plan, "plan"))
    unknown_fields = set(normalized) - TOP_LEVEL_FIELDS
    missing_fields = TOP_LEVEL_FIELDS - set(normalized)
    if unknown_fields:
        _error(
            "UNKNOWN_FIELD",
            f"plan 包含未支援欄位：{sorted(unknown_fields)}",
            "plan",
        )
    if missing_fields:
        _error(
            "MISSING_FIELD",
            f"plan 缺少欄位：{sorted(missing_fields)}",
            "plan",
        )
    if normalized["schema_version"] != SCHEMA_VERSION:
        _error(
            "UNSUPPORTED_SCHEMA_VERSION",
            f"不支援 schema_version={normalized['schema_version']}",
            "schema_version",
        )

    requirements = _require_list(normalized["requirements"], "requirements")
    roles = _require_list(normalized["roles"], "roles")
    steps = _require_list(normalized["steps"], "steps")
    _require_list(normalized["constraints"], "constraints")
    _require_list(normalized["assignment_preferences"], "assignment_preferences")

    requirement_by_id = _collect_unique_ids(requirements, "requirements")
    role_by_id = _collect_unique_ids(roles, "roles")
    step_by_id = _collect_unique_ids(steps, "steps")

    for index, role in enumerate(roles):
        path = f"roles[{index}]"
        _require_id(role.get("target_type"), f"{path}.target_type")
        capabilities = _normalize_unique_strings(
            role.get("required_capabilities", []),
            f"{path}.required_capabilities",
        )
        required_zones = _normalize_unique_strings(
            role.get("required_zones", []),
            f"{path}.required_zones",
        )
        binding = _require_object(
            role.get("binding", {"mode": "automatic"}),
            f"{path}.binding",
        )
        unknown_binding_fields = set(binding) - BINDING_FIELDS
        if unknown_binding_fields:
            _error(
                "UNKNOWN_FIELD",
                f"{path}.binding 包含未支援欄位：{sorted(unknown_binding_fields)}",
                f"{path}.binding",
            )
        mode = binding.get("mode", "automatic")
        if mode not in BINDING_MODES:
            _error(
                "INVALID_BINDING_MODE",
                f"{path}.binding.mode 不合法：{mode}",
                f"{path}.binding.mode",
            )
        if mode in {"required", "preferred"}:
            _require_id(binding.get("target_id"), f"{path}.binding.target_id")
        elif "target_id" in binding:
            _error(
                "UNEXPECTED_TARGET_ID",
                f"{path}.binding 在 automatic 模式不可指定 target_id",
                f"{path}.binding.target_id",
            )
        role["required_capabilities"] = capabilities
        role["required_zones"] = required_zones
        role["binding"] = binding

    role_ids = set(role_by_id)
    normalized_constraints = []
    relation_types_by_roles = {}
    for index, raw_constraint in enumerate(normalized["constraints"]):
        path = f"constraints[{index}]"
        constraint = _require_object(raw_constraint, path)
        unknown = set(constraint) - CONSTRAINT_FIELDS
        if unknown:
            _error(
                "UNKNOWN_FIELD",
                f"{path} 包含未支援欄位：{sorted(unknown)}",
                path,
            )
        constraint_type = constraint.get("type")
        if constraint_type not in CONSTRAINT_TYPES:
            _error(
                "UNKNOWN_CONSTRAINT_TYPE",
                f"{path}.type 不支援：{constraint_type}",
                f"{path}.type",
            )
        roles_in_constraint = _normalize_role_relation(
            constraint.get("roles"),
            f"{path}.roles",
            role_ids,
        )
        hard = constraint.get("hard", True)
        if not isinstance(hard, bool):
            _error("INVALID_TYPE", f"{path}.hard 必須是 boolean", f"{path}.hard")
        constraint = deepcopy(constraint)
        constraint["roles"] = roles_in_constraint
        constraint["hard"] = hard
        normalized_constraints.append(constraint)
        if hard:
            relation_key = frozenset(roles_in_constraint)
            previous_type = relation_types_by_roles.get(relation_key)
            if previous_type is not None and previous_type != constraint_type:
                _error(
                    "CONTRADICTORY_CONSTRAINT",
                    f"同一組 roles 不可同時要求 same 與 distinct：{sorted(relation_key)}",
                    path,
                )
            relation_types_by_roles[relation_key] = constraint_type
    normalized["constraints"] = normalized_constraints

    normalized_preferences = []
    for index, raw_preference in enumerate(normalized["assignment_preferences"]):
        path = f"assignment_preferences[{index}]"
        preference = _require_object(raw_preference, path)
        unknown = set(preference) - PREFERENCE_FIELDS
        if unknown:
            _error(
                "UNKNOWN_FIELD",
                f"{path} 包含未支援欄位：{sorted(unknown)}",
                path,
            )
        preference_type = preference.get("type")
        if preference_type not in PREFERENCE_TYPES:
            _error(
                "UNKNOWN_PREFERENCE_TYPE",
                f"{path}.type 不支援：{preference_type}",
                f"{path}.type",
            )
        roles_in_preference = _normalize_role_relation(
            preference.get("roles"),
            f"{path}.roles",
            role_ids,
        )
        weight = preference.get("weight", 1.0)
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or weight < 0
        ):
            _error(
                "INVALID_WEIGHT",
                f"{path}.weight 必須是有限的非負數",
                f"{path}.weight",
            )
        preference = deepcopy(preference)
        preference["roles"] = roles_in_preference
        preference["weight"] = float(weight)
        normalized_preferences.append(preference)
    normalized["assignment_preferences"] = normalized_preferences

    covered_requirements = set()
    step_dependencies = {}
    for index, step in enumerate(steps):
        path = f"steps[{index}]"
        step_id = step["id"].strip()
        role_id = _require_id(step.get("role"), f"{path}.role")
        if role_id not in role_by_id:
            _error(
                "UNKNOWN_ROLE",
                f"步驟 {step_id} 引用不存在的 role：{role_id}",
                f"{path}.role",
            )
        action = _require_object(step.get("action"), f"{path}.action")
        _require_id(action.get("function_name"), f"{path}.action.function_name")
        _require_object(action.get("arguments"), f"{path}.action.arguments")
        dependencies = _normalize_dependencies(
            step.get("depends_on", []),
            f"{path}.depends_on",
        )
        dependency_ids = {dependency["step_id"] for dependency in dependencies}
        if step_id in dependency_ids:
            _error(
                "SELF_DEPENDENCY",
                f"步驟 {step_id} 不可依賴自己",
                f"{path}.depends_on",
            )
        unknown_dependencies = dependency_ids - set(step_by_id)
        if unknown_dependencies:
            _error(
                "UNKNOWN_DEPENDENCY",
                f"步驟 {step_id} 依賴不存在的步驟：{sorted(unknown_dependencies)}",
                f"{path}.depends_on",
            )
        satisfies = _require_list(step.get("satisfies", []), f"{path}.satisfies")
        normalized_satisfies = []
        for requirement_index, requirement_id in enumerate(satisfies):
            requirement_id = _require_id(
                requirement_id,
                f"{path}.satisfies[{requirement_index}]",
            )
            if requirement_id not in requirement_by_id:
                _error(
                    "UNKNOWN_REQUIREMENT",
                    f"步驟 {step_id} 引用不存在的 requirement：{requirement_id}",
                    f"{path}.satisfies[{requirement_index}]",
                )
            normalized_satisfies.append(requirement_id)
            covered_requirements.add(requirement_id)
        step["depends_on"] = dependencies
        step["satisfies"] = normalized_satisfies
        step_dependencies[step_id] = dependency_ids

    _validate_cycle(step_dependencies)
    required_ids = {
        requirement_id
        for requirement_id, requirement in requirement_by_id.items()
        if requirement.get("required", True) is True
    }
    uncovered = required_ids - covered_requirements
    if uncovered:
        _error(
            "UNCOVERED_REQUIREMENT",
            f"必要需求沒有對應步驟：{sorted(uncovered)}",
            "requirements",
        )
    return normalized


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    raise PlanValidationError(
        "UNSUPPORTED_PARAMETER_TYPE",
        f"不支援的工具參數型別：{expected_type}",
    )


def _validate_argument_value(function_name, name, value, definition, path):
    if not isinstance(definition, dict):
        raise PlanValidationError(
            "INVALID_TOOL_SCHEMA",
            f"工具 {function_name}.{name} 定義必須是 object",
            path,
        )
    expected = definition.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]
    if not expected_types or any(not isinstance(item, str) for item in expected_types):
        raise PlanValidationError(
            "INVALID_TOOL_SCHEMA",
            f"工具 {function_name}.{name} 缺少有效 type",
            path,
        )
    if not any(_matches_type(value, item) for item in expected_types):
        raise PlanValidationError(
            "INVALID_ARGUMENT_TYPE",
            f"工具 {function_name}.{name} 參數型別錯誤",
            path,
        )
    if value is None:
        return
    enum_values = definition.get("enum")
    if enum_values is not None and value not in enum_values:
        raise PlanValidationError(
            "INVALID_ARGUMENT_VALUE",
            f"工具 {function_name}.{name} 的值不在允許範圍",
            path,
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise PlanValidationError(
                "INVALID_ARGUMENT_VALUE",
                f"工具 {function_name}.{name} 必須是有限數值",
                path,
            )
        if "minimum" in definition and value < definition["minimum"]:
            raise PlanValidationError(
                "ARGUMENT_OUT_OF_RANGE",
                f"工具 {function_name}.{name} 小於最小值",
                path,
            )
        if "maximum" in definition and value > definition["maximum"]:
            raise PlanValidationError(
                "ARGUMENT_OUT_OF_RANGE",
                f"工具 {function_name}.{name} 超過最大值",
                path,
            )
    if isinstance(value, list):
        if "minItems" in definition and len(value) < definition["minItems"]:
            raise PlanValidationError(
                "INVALID_ARGUMENT_LENGTH",
                f"工具 {function_name}.{name} 項目數不足",
                path,
            )
        if "maxItems" in definition and len(value) > definition["maxItems"]:
            raise PlanValidationError(
                "INVALID_ARGUMENT_LENGTH",
                f"工具 {function_name}.{name} 項目數過多",
                path,
            )
        item_definition = definition.get("items")
        if item_definition is not None:
            for index, item in enumerate(value):
                _validate_argument_value(
                    function_name,
                    f"{name}[{index}]",
                    item,
                    item_definition,
                    f"{path}[{index}]",
                )


def _validate_tool_arguments(step, tool):
    function_name = step["action"]["function_name"]
    arguments = step["action"]["arguments"]
    parameters = tool.get("parameters", {})
    required = tool.get("required", [])
    if not isinstance(parameters, dict) or not isinstance(required, list):
        raise PlanValidationError(
            "INVALID_TOOL_SCHEMA",
            f"工具 {function_name} 的 parameters/required 格式錯誤",
        )
    unknown = set(arguments) - set(parameters)
    if unknown:
        raise PlanValidationError(
            "UNKNOWN_ARGUMENT",
            f"工具 {function_name} 包含未授權參數：{sorted(unknown)}",
            f"steps.{step['id']}.action.arguments",
        )
    missing = [
        name for name in required
        if name not in arguments or arguments[name] is None
    ]
    if missing:
        raise PlanValidationError(
            "MISSING_ARGUMENT",
            f"工具 {function_name} 缺少必要參數：{missing}",
            f"steps.{step['id']}.action.arguments",
        )
    for name, value in arguments.items():
        _validate_argument_value(
            function_name,
            name,
            value,
            parameters[name],
            f"steps.{step['id']}.action.arguments.{name}",
        )


def validate_plan_against_catalogs(
    plan: Any,
    tool_catalog: dict[str, dict[str, Any]],
    device_registry: Any,
) -> dict[str, Any]:
    normalized = validate_plan(plan)
    if not isinstance(tool_catalog, dict):
        raise PlanValidationError("INVALID_TOOL_CATALOG", "tool_catalog 必須是 dict")

    role_by_id = {role["id"]: role for role in normalized["roles"]}
    inferred_capabilities = {
        role_id: set(role.get("required_capabilities", []))
        for role_id, role in role_by_id.items()
    }
    inferred_zones = {
        role_id: set(role.get("required_zones", []))
        for role_id, role in role_by_id.items()
    }
    for step in normalized["steps"]:
        function_name = step["action"]["function_name"]
        tool = tool_catalog.get(function_name)
        if not isinstance(tool, dict):
            raise PlanValidationError(
                "UNKNOWN_TOOL",
                f"步驟 {step['id']} 使用未授權工具：{function_name}",
                f"steps.{step['id']}.action.function_name",
            )
        role = role_by_id[step["role"]]
        target_types = tool.get("target_types", [])
        if target_types and role["target_type"] not in target_types:
            raise PlanValidationError(
                "INCOMPATIBLE_TARGET_TYPE",
                f"工具 {function_name} 不支援 {role['target_type']}",
                f"steps.{step['id']}.role",
            )
        _validate_tool_arguments(step, tool)
        tool_capabilities = tool.get("required_capabilities", [function_name])
        tool_zones = tool.get("required_zones", [])
        if not isinstance(tool_capabilities, list) or not all(
            isinstance(item, str) and item for item in tool_capabilities
        ):
            raise PlanValidationError(
                "INVALID_TOOL_SCHEMA",
                f"工具 {function_name}.required_capabilities 格式錯誤",
            )
        if not isinstance(tool_zones, list) or not all(
            isinstance(item, str) and item for item in tool_zones
        ):
            raise PlanValidationError(
                "INVALID_TOOL_SCHEMA",
                f"工具 {function_name}.required_zones 格式錯誤",
            )
        inferred_capabilities[role["id"]].update(tool_capabilities)
        inferred_zones[role["id"]].update(tool_zones)

    for role in normalized["roles"]:
        role["required_capabilities"] = sorted(inferred_capabilities[role["id"]])
        role["required_zones"] = sorted(inferred_zones[role["id"]])

    available_devices = [
        device for device in device_registry.list() if device.get("enabled") is True
    ]
    for role in normalized["roles"]:
        required = set(role.get("required_capabilities", []))
        required_zones = set(role.get("required_zones", []))
        candidates = [
            device
            for device in available_devices
            if device["type"] == role["target_type"]
            and required.issubset(set(device.get("capabilities", [])))
            and required_zones.issubset(set(device.get("reachable_zones", [])))
        ]
        binding = role.get("binding", {})
        if binding.get("mode") == "required":
            candidates = [
                device
                for device in candidates
                if device["id"] == binding.get("target_id")
            ]
        if not candidates:
            raise PlanValidationError(
                "NO_CAPABLE_DEVICE",
                f"角色 {role['id']} 沒有可用設備",
                f"roles.{role['id']}",
            )
    return normalized
