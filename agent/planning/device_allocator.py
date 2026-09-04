"""抽象角色與實際設備之間的分配、驗證與解析。

本模組負責 logical resource role 到 physical device binding 的完整流程：
先依設備狀態與 Plan 規則分配角色，再驗證 assignment，最後將 concrete
target 寫入 resolved steps。它不呼叫 executor 或操作硬體。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.planning.device_registry import DeviceRegistry
from agent.planning.plan_schema import PlanValidationError, validate_plan


class AllocationError(RuntimeError):
    """Raised when no assignment satisfies all hard constraints."""

    def __init__(self, code: str, message: str, role_id: str | None = None):
        self.code = code
        self.role_id = role_id
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "message": str(self)}
        if self.role_id is not None:
            result["role_id"] = self.role_id
        return result


class AssignmentValidationError(RuntimeError):
    """Raised when role-to-device assignments violate the plan or registry."""

    def __init__(
        self,
        code: str,
        message: str,
        role_id: str | None = None,
        target_id: str | None = None,
    ):
        self.code = code
        self.role_id = role_id
        self.target_id = target_id
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "message": str(self)}
        if self.role_id is not None:
            result["role_id"] = self.role_id
        if self.target_id is not None:
            result["target_id"] = self.target_id
        return result


def _constraint_pairs(plan, constraint_type):
    pairs = []
    for constraint in plan["constraints"]:
        if not isinstance(constraint, dict) or constraint.get("type") != constraint_type:
            continue
        if constraint.get("hard", True) is not True:
            continue
        roles = constraint.get("roles")
        if not isinstance(roles, list) or len(roles) < 2:
            raise AllocationError(
                "INVALID_CONSTRAINT",
                f"{constraint_type} 必須至少包含兩個 roles",
            )
        pairs.append(tuple(roles))
    return pairs


def _satisfies_hard_constraints(assignments, same_groups, distinct_groups):
    for roles in same_groups:
        assigned = {assignments[role] for role in roles if role in assignments}
        if len(assigned) > 1:
            return False
    for roles in distinct_groups:
        assigned = [assignments[role] for role in roles if role in assignments]
        if len(assigned) != len(set(assigned)):
            return False
    return True


def _preference_score(plan, assignments):
    score = 0.0
    for preference in plan["assignment_preferences"]:
        if not isinstance(preference, dict):
            continue
        roles = preference.get("roles")
        if not isinstance(roles, list) or not roles or any(
            role not in assignments for role in roles
        ):
            continue
        weight = preference.get("weight", 1)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            weight = 1
        assigned = [assignments[role] for role in roles]
        preference_type = preference.get("type")
        if preference_type == "prefer_distinct_assignment" and len(assigned) == len(set(assigned)):
            score += float(weight)
        elif preference_type == "prefer_same_assignment" and len(set(assigned)) == 1:
            score += float(weight)
    for role in plan["roles"]:
        binding = role.get("binding", {})
        if (
            binding.get("mode") == "preferred"
            and assignments.get(role["id"]) == binding.get("target_id")
        ):
            score += 1.0
    return score


def allocate_devices(
    plan: Any,
    device_registry: DeviceRegistry,
    device_statuses: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(device_registry, DeviceRegistry):
        raise AllocationError(
            "INVALID_DEVICE_REGISTRY",
            "device_registry 必須是 DeviceRegistry",
        )
    try:
        normalized = validate_plan(plan)
    except PlanValidationError as exc:
        raise AllocationError(
            "INVALID_PLAN",
            f"無法分配未通過驗證的 plan：{exc}",
        ) from exc
    if device_statuses is None:
        device_statuses = {}
    if not isinstance(device_statuses, dict):
        raise AllocationError(
            "INVALID_DEVICE_STATUSES",
            "device_statuses 必須是 dict",
        )
    for device_id, status in device_statuses.items():
        if not isinstance(device_id, str) or not isinstance(status, dict):
            raise AllocationError(
                "INVALID_DEVICE_STATUS",
                "每個 device status 必須使用字串 ID 與 object 內容",
            )
    devices = device_registry.list()
    role_candidates = {}

    for role in normalized["roles"]:
        required_capabilities = set(role.get("required_capabilities", []))
        required_zones = set(role.get("required_zones", []))
        candidates = []
        for device in devices:
            status = device_statuses.get(device["id"], {})
            if device.get("enabled") is not True:
                continue
            if status and (
                status.get("connected", True) is not True
                or status.get("available", True) is not True
            ):
                continue
            if device["type"] != role["target_type"]:
                continue
            if not required_capabilities.issubset(set(device.get("capabilities", []))):
                continue
            if not required_zones.issubset(set(device.get("reachable_zones", []))):
                continue
            binding = role.get("binding", {})
            if binding.get("mode") == "required" and device["id"] != binding.get("target_id"):
                continue
            candidates.append(device["id"])
        if not candidates:
            raise AllocationError(
                "NO_CAPABLE_DEVICE",
                f"角色 {role['id']} 沒有可用設備",
                role_id=role["id"],
            )
        role_candidates[role["id"]] = sorted(candidates)

    same_groups = _constraint_pairs(normalized, "same_assignment")
    distinct_groups = _constraint_pairs(normalized, "distinct_assignment")
    role_ids = sorted(role_candidates, key=lambda role_id: len(role_candidates[role_id]))
    solutions = []

    def search(index, assignments):
        if index == len(role_ids):
            solutions.append((
                _preference_score(normalized, assignments),
                assignments.copy(),
            ))
            return
        role_id = role_ids[index]
        for device_id in role_candidates[role_id]:
            assignments[role_id] = device_id
            if _satisfies_hard_constraints(assignments, same_groups, distinct_groups):
                search(index + 1, assignments)
            assignments.pop(role_id, None)

    search(0, {})
    if not solutions:
        raise AllocationError(
            "UNSATISFIABLE_CONSTRAINTS",
            "沒有任何分配能滿足所有硬性限制",
        )
    solutions.sort(key=lambda item: (-item[0], sorted(item[1].items())))
    score, assignments = solutions[0]
    return {
        "status": "resolved",
        "assignments": assignments,
        "preference_score": score,
    }


def validate_assignments(
    plan: Any,
    assignments: Any,
    device_registry: DeviceRegistry,
    device_statuses: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Validate and normalize a complete role-to-device assignment."""
    if not isinstance(device_registry, DeviceRegistry):
        raise AssignmentValidationError(
            "INVALID_DEVICE_REGISTRY",
            "device_registry 必須是 DeviceRegistry",
        )
    try:
        normalized_plan = validate_plan(plan)
    except PlanValidationError as exc:
        raise AssignmentValidationError(
            "INVALID_PLAN",
            f"無法驗證未通過驗證的 plan：{exc}",
        ) from exc
    if not isinstance(assignments, dict):
        raise AssignmentValidationError(
            "INVALID_ASSIGNMENTS",
            "assignments 必須是 object",
        )
    if device_statuses is None:
        device_statuses = {}
    if not isinstance(device_statuses, dict):
        raise AssignmentValidationError(
            "INVALID_DEVICE_STATUSES",
            "device_statuses 必須是 dict",
        )

    role_by_id = {role["id"]: role for role in normalized_plan["roles"]}
    role_ids = set(role_by_id)
    assignment_role_ids = set(assignments)
    missing_roles = role_ids - assignment_role_ids
    unknown_roles = assignment_role_ids - role_ids
    if missing_roles:
        raise AssignmentValidationError(
            "MISSING_ROLE_ASSIGNMENT",
            f"缺少 role assignments：{sorted(missing_roles)}",
        )
    if unknown_roles:
        raise AssignmentValidationError(
            "UNKNOWN_ROLE_ASSIGNMENT",
            f"assignments 包含未知 roles：{sorted(unknown_roles)}",
        )

    normalized_assignments = {}
    for role_id, raw_target_id in assignments.items():
        if not isinstance(raw_target_id, str) or not raw_target_id.strip():
            raise AssignmentValidationError(
                "INVALID_TARGET_ID",
                f"role {role_id} 的 target ID 必須是非空字串",
                role_id=role_id,
            )
        target_id = raw_target_id.strip()
        device = device_registry.get(target_id)
        if device is None:
            raise AssignmentValidationError(
                "UNKNOWN_TARGET",
                f"role {role_id} 分配到不存在的設備：{target_id}",
                role_id=role_id,
                target_id=target_id,
            )
        if device.get("enabled") is not True:
            raise AssignmentValidationError(
                "TARGET_DISABLED",
                f"設備 {target_id} 未啟用",
                role_id=role_id,
                target_id=target_id,
            )
        status = device_statuses.get(target_id, {})
        if status and (
            status.get("connected", True) is not True
            or status.get("available", True) is not True
        ):
            raise AssignmentValidationError(
                "TARGET_UNAVAILABLE",
                f"設備 {target_id} 目前不可用",
                role_id=role_id,
                target_id=target_id,
            )
        role = role_by_id[role_id]
        if device["type"] != role["target_type"]:
            raise AssignmentValidationError(
                "INCOMPATIBLE_TARGET_TYPE",
                f"role {role_id} 需要 {role['target_type']}，"
                f"但 {target_id} 是 {device['type']}",
                role_id=role_id,
                target_id=target_id,
            )
        missing_capabilities = set(role.get("required_capabilities", [])) - set(
            device.get("capabilities", [])
        )
        if missing_capabilities:
            raise AssignmentValidationError(
                "TARGET_MISSING_CAPABILITY",
                f"設備 {target_id} 缺少能力：{sorted(missing_capabilities)}",
                role_id=role_id,
                target_id=target_id,
            )
        unreachable_zones = set(role.get("required_zones", [])) - set(
            device.get("reachable_zones", [])
        )
        if unreachable_zones:
            raise AssignmentValidationError(
                "TARGET_ZONE_UNREACHABLE",
                f"設備 {target_id} 無法到達區域：{sorted(unreachable_zones)}",
                role_id=role_id,
                target_id=target_id,
            )
        binding = role.get("binding", {})
        if binding.get("mode") == "required" and binding.get("target_id") != target_id:
            raise AssignmentValidationError(
                "REQUIRED_BINDING_VIOLATION",
                f"role {role_id} 必須分配給 {binding.get('target_id')}",
                role_id=role_id,
                target_id=target_id,
            )
        normalized_assignments[role_id] = target_id

    for constraint in normalized_plan["constraints"]:
        if constraint.get("hard", True) is not True:
            continue
        roles = constraint["roles"]
        assigned_targets = [normalized_assignments[role_id] for role_id in roles]
        if constraint["type"] == "same_assignment" and len(set(assigned_targets)) != 1:
            raise AssignmentValidationError(
                "SAME_ASSIGNMENT_VIOLATION",
                f"roles {roles} 必須分配給同一設備",
            )
        if (
            constraint["type"] == "distinct_assignment"
            and len(set(assigned_targets)) != len(assigned_targets)
        ):
            raise AssignmentValidationError(
                "DISTINCT_ASSIGNMENT_VIOLATION",
                f"roles {roles} 必須分配給不同設備",
            )
    return normalized_assignments


def resolve_assignments(
    plan: Any,
    assignments: Any,
    device_registry: DeviceRegistry,
) -> dict[str, Any]:
    """Write validated role assignments into concrete resolved plan steps."""
    if not isinstance(device_registry, DeviceRegistry):
        raise AssignmentValidationError(
            "INVALID_DEVICE_REGISTRY",
            "device_registry 必須是 DeviceRegistry",
        )
    try:
        normalized_plan = validate_plan(plan)
    except PlanValidationError as exc:
        raise AssignmentValidationError(
            "INVALID_PLAN",
            f"無法解析未通過驗證的 plan：{exc}",
        ) from exc
    if not isinstance(assignments, dict):
        raise AssignmentValidationError(
            "INVALID_ASSIGNMENTS",
            "assignments 必須是 object",
        )

    resolved_steps = []
    for step in normalized_plan["steps"]:
        role_id = step["role"]
        target_id = assignments.get(role_id)
        device = device_registry.get(target_id) if isinstance(target_id, str) else None
        if device is None:
            raise AssignmentValidationError(
                "INVALID_VALIDATED_ASSIGNMENTS",
                f"role {role_id} 缺少已驗證的 target device",
                role_id=role_id,
                target_id=target_id,
            )
        resolved_steps.append({
            "id": step["id"],
            "role": role_id,
            "target": {"type": device["type"], "id": target_id},
            "function_name": step["action"]["function_name"],
            "arguments": deepcopy(step["action"]["arguments"]),
            "depends_on": deepcopy(step["depends_on"]),
            "satisfies": deepcopy(step["satisfies"]),
        })

    return {
        "schema_version": "1.0",
        "source_plan": {"schema_version": normalized_plan["schema_version"]},
        "assignments": deepcopy(assignments),
        "resolved_steps": resolved_steps,
    }
