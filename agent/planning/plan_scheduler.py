"""Build a deterministic execution preview from a resolved Plan.

This module owns scheduling semantics only: dependency readiness, per-device
mutual exclusion and failure-policy metadata.  It is deliberately side-effect
free and never calls an executor, service or hardware driver.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SCHEDULER_VERSION = "1.0"
SUPPORTED_FAILURE_POLICIES = {"stop_all"}


class SchedulerError(ValueError):
    """Raised when a resolved Plan cannot be scheduled safely."""

    def __init__(self, code: str, message: str, step_id: str | None = None):
        self.code = code
        self.step_id = step_id
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "message": str(self)}
        if self.step_id is not None:
            result["step_id"] = self.step_id
        return result


def _normalize_resolved_steps(resolved_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(resolved_plan, dict):
        raise SchedulerError("INVALID_RESOLVED_PLAN", "resolved_plan 必須是 object")
    raw_steps = resolved_plan.get("resolved_steps")
    if not isinstance(raw_steps, list):
        raise SchedulerError(
            "INVALID_RESOLVED_STEPS",
            "resolved_plan.resolved_steps 必須是 list",
        )

    normalized = []
    step_ids = set()
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise SchedulerError(
                "INVALID_RESOLVED_STEP",
                f"resolved_steps[{index}] 必須是 object",
            )
        step_id = raw_step.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise SchedulerError(
                "INVALID_STEP_ID",
                f"resolved_steps[{index}].id 必須是非空字串",
            )
        step_id = step_id.strip()
        if step_id in step_ids:
            raise SchedulerError("DUPLICATE_STEP_ID", f"step id 重複：{step_id}", step_id)

        target = raw_step.get("target")
        if not isinstance(target, dict):
            raise SchedulerError("INVALID_TARGET", f"步驟 {step_id} 缺少 target", step_id)
        target_id = target.get("id")
        target_type = target.get("type")
        if not isinstance(target_id, str) or not target_id.strip():
            raise SchedulerError(
                "INVALID_TARGET_ID", f"步驟 {step_id} target.id 無效", step_id
            )
        if not isinstance(target_type, str) or not target_type.strip():
            raise SchedulerError(
                "INVALID_TARGET_TYPE", f"步驟 {step_id} target.type 無效", step_id
            )

        function_name = raw_step.get("function_name")
        arguments = raw_step.get("arguments")
        if not isinstance(function_name, str) or not function_name.strip():
            raise SchedulerError(
                "INVALID_FUNCTION", f"步驟 {step_id} function_name 無效", step_id
            )
        if not isinstance(arguments, dict):
            raise SchedulerError(
                "INVALID_ARGUMENTS", f"步驟 {step_id} arguments 必須是 object", step_id
            )

        raw_dependencies = raw_step.get("depends_on", [])
        if not isinstance(raw_dependencies, list):
            raise SchedulerError(
                "INVALID_DEPENDENCIES", f"步驟 {step_id} depends_on 必須是 list", step_id
            )
        dependencies = []
        seen_dependencies = set()
        for dependency_index, raw_dependency in enumerate(raw_dependencies):
            if not isinstance(raw_dependency, dict):
                raise SchedulerError(
                    "INVALID_DEPENDENCY",
                    f"步驟 {step_id} depends_on[{dependency_index}] 必須是 object",
                    step_id,
                )
            dependency_id = raw_dependency.get("step_id")
            condition = raw_dependency.get("condition", "completed")
            if not isinstance(dependency_id, str) or not dependency_id.strip():
                raise SchedulerError(
                    "INVALID_DEPENDENCY",
                    f"步驟 {step_id} dependency step_id 無效",
                    step_id,
                )
            dependency_id = dependency_id.strip()
            if condition not in {"completed", "succeeded"}:
                raise SchedulerError(
                    "INVALID_DEPENDENCY_CONDITION",
                    f"步驟 {step_id} dependency condition 無效：{condition}",
                    step_id,
                )
            key = (dependency_id, condition)
            if key in seen_dependencies:
                raise SchedulerError(
                    "DUPLICATE_DEPENDENCY",
                    f"步驟 {step_id} 重複依賴 {dependency_id}",
                    step_id,
                )
            seen_dependencies.add(key)
            dependencies.append({"step_id": dependency_id, "condition": condition})

        normalized.append({
            "id": step_id,
            "role": raw_step.get("role"),
            "target": {
                "type": target_type.strip(),
                "id": target_id.strip(),
            },
            "function_name": function_name.strip(),
            "arguments": deepcopy(arguments),
            "depends_on": dependencies,
        })
        step_ids.add(step_id)

    for step in normalized:
        dependency_ids = {item["step_id"] for item in step["depends_on"]}
        if step["id"] in dependency_ids:
            raise SchedulerError(
                "SELF_DEPENDENCY",
                f"步驟 {step['id']} 不可依賴自己",
                step["id"],
            )
        unknown = dependency_ids - step_ids
        if unknown:
            raise SchedulerError(
                "UNKNOWN_DEPENDENCY",
                f"步驟 {step['id']} 依賴不存在步驟：{sorted(unknown)}",
                step["id"],
            )
    return normalized



def _normalize_initial_holding(
    initial_holding: dict[str, str | None] | None,
) -> dict[str, str | None]:
    """Normalize planning-only end-effector occupancy state.

    Mapping format:
        {
            "device_id": None,            # end effector free
            "device_id_2": "object_id",   # currently holding this object
        }

    This state is planner metadata only. It does not query or modify hardware.
    """
    if initial_holding is None:
        return {}
    if not isinstance(initial_holding, dict):
        raise SchedulerError(
            "INVALID_RESOURCE_STATE",
            "initial_holding 必須是 dict",
        )

    normalized: dict[str, str | None] = {}
    for raw_device_id, raw_object_id in initial_holding.items():
        if not isinstance(raw_device_id, str) or not raw_device_id.strip():
            raise SchedulerError(
                "INVALID_RESOURCE_STATE",
                "initial_holding device id 必須是非空字串",
            )

        device_id = raw_device_id.strip()

        if raw_object_id is None:
            normalized[device_id] = None
            continue

        if not isinstance(raw_object_id, str) or not raw_object_id.strip():
            raise SchedulerError(
                "INVALID_RESOURCE_STATE",
                f"initial_holding[{device_id!r}] 必須是 object id 或 None",
            )

        normalized[device_id] = raw_object_id.strip()

    return normalized


def _resource_allows_step(
    step: dict[str, Any],
    holding: dict[str, str | None],
) -> bool:
    """Return whether planning resource state allows this step now."""
    device_id = step["target"]["id"]
    function_name = step["function_name"]
    arguments = step.get("arguments", {})

    held_object_id = holding.get(device_id)

    if function_name == "pick_object":
        # Capacity-1 end effector must be empty before acquiring another object.
        return held_object_id is None

    if function_name == "place_object":
        # A device may only place the exact object it currently owns.
        object_id = arguments.get("object_id")
        return (
            isinstance(object_id, str)
            and held_object_id == object_id
        )

    return True


def _apply_resource_effect(
    step: dict[str, Any],
    holding: dict[str, str | None],
) -> None:
    """Apply the successful planning-side acquire/release effect."""
    device_id = step["target"]["id"]
    function_name = step["function_name"]
    arguments = step.get("arguments", {})

    if function_name == "pick_object":
        object_id = arguments.get("object_id")
        if isinstance(object_id, str) and object_id:
            holding[device_id] = object_id
        return

    if function_name == "place_object":
        object_id = arguments.get("object_id")
        if (
            isinstance(object_id, str)
            and holding.get(device_id) == object_id
        ):
            holding[device_id] = None


def _resource_aware_topological_order(
    steps: list[dict[str, Any]],
    *,
    initial_holding: dict[str, str | None] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build a stable DAG order while respecting end-effector occupancy.

    No semantic dependency is invented.

    Example with one capacity-1 arm:
        pick(A) -> place(A)
        pick(B) -> place(B)

    Stage2 may correctly say there is no semantic relation between the two
    chains. If allocation maps both chains to the same arm, planning state
    prevents pick(B) while that arm still holds A, so the safe order becomes:

        pick(A), place(A), pick(B), place(B)

    If the two chains are mapped to different arms, both picks remain eligible.
    """
    source_order = {
        step["id"]: index
        for index, step in enumerate(steps)
    }
    step_by_id = {
        step["id"]: step
        for step in steps
    }
    remaining = {
        step["id"]: {
            item["step_id"]
            for item in step["depends_on"]
        }
        for step in steps
    }

    holding = _normalize_initial_holding(initial_holding)
    ordered: list[str] = []
    completed: set[str] = set()
    decisions: list[dict[str, Any]] = []

    while len(ordered) < len(steps):
        dependency_ready = sorted(
            (
                step_id
                for step_id, dependencies in remaining.items()
                if (
                    step_id not in completed
                    and dependencies.issubset(completed)
                )
            ),
            key=source_order.get,
        )

        if not dependency_ready:
            raise SchedulerError(
                "DEPENDENCY_CYCLE",
                "resolved Plan dependencies 形成 cycle",
            )

        selected = None
        blocked = []

        for step_id in dependency_ready:
            step = step_by_id[step_id]

            if _resource_allows_step(step, holding):
                selected = step_id
                break

            blocked.append({
                "step_id": step_id,
                "device_id": step["target"]["id"],
                "function_name": step["function_name"],
                "object_id": (
                    step.get("arguments") or {}
                ).get("object_id"),
                "held_object_id": holding.get(
                    step["target"]["id"]
                ),
            })

        if selected is None:
            raise SchedulerError(
                "RESOURCE_STATE_DEADLOCK",
                (
                    "dependencies 已滿足，但所有 ready steps 都被 "
                    f"end-effector occupancy 阻擋：{blocked}"
                ),
            )

        selected_step = step_by_id[selected]

        decisions.append({
            "step_id": selected,
            "device_id": selected_step["target"]["id"],
            "function_name": selected_step["function_name"],
            "held_before": holding.get(
                selected_step["target"]["id"]
            ),
        })

        _apply_resource_effect(
            selected_step,
            holding,
        )

        decisions[-1]["held_after"] = holding.get(
            selected_step["target"]["id"]
        )

        ordered.append(selected)
        completed.add(selected)

    return ordered, decisions

def _stable_topological_order(steps: list[dict[str, Any]]) -> list[str]:
    source_order = {step["id"]: index for index, step in enumerate(steps)}
    remaining = {
        step["id"]: {item["step_id"] for item in step["depends_on"]}
        for step in steps
    }
    ordered = []
    completed = set()
    while len(ordered) < len(steps):
        ready = sorted(
            (
                step_id
                for step_id, dependencies in remaining.items()
                if step_id not in completed and dependencies.issubset(completed)
            ),
            key=source_order.get,
        )
        if not ready:
            raise SchedulerError(
                "DEPENDENCY_CYCLE",
                "resolved Plan dependencies 形成 cycle",
            )
        ordered.extend(ready)
        completed.update(ready)
    return ordered


def build_execution_preview(
    resolved_plan: Any,
    *,
    failure_policy: str = "stop_all",
    initial_holding: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Return scheduler initial state without dispatching any step."""
    if failure_policy not in SUPPORTED_FAILURE_POLICIES:
        raise SchedulerError(
            "UNSUPPORTED_FAILURE_POLICY",
            f"不支援 failure_policy：{failure_policy}",
        )
    steps = _normalize_resolved_steps(resolved_plan)

    # First validate the semantic DAG itself. Resource occupancy must never
    # "fix" or hide a real dependency cycle.
    _stable_topological_order(steps)

    topological_order, resource_decisions = (
        _resource_aware_topological_order(
            steps,
            initial_holding=initial_holding,
        )
    )

    step_by_id = {step["id"]: step for step in steps}

    initial_ready = [
        step_id
        for step_id in topological_order
        if not step_by_id[step_id]["depends_on"]
    ]
    # At most one root step per concrete device can be dispatched initially.
    # Other roots stay ready and will be selected after that device is free.
    claimed_devices = set()
    initial_dispatchable = []
    initial_holding_state = _normalize_initial_holding(
        initial_holding
    )

    for step_id in initial_ready:
        step = step_by_id[step_id]
        device_id = step["target"]["id"]

        if device_id in claimed_devices:
            continue

        if not _resource_allows_step(
            step,
            initial_holding_state,
        ):
            continue

        claimed_devices.add(device_id)
        initial_dispatchable.append(step_id)

    device_workloads = {}
    for step_id in topological_order:
        device_id = step_by_id[step_id]["target"]["id"]
        device_workloads.setdefault(device_id, []).append(step_id)

    preview_steps = []
    for step_id in topological_order:
        step = step_by_id[step_id]
        waiting_for = [item["step_id"] for item in step["depends_on"]]
        preview_steps.append({
            "id": step_id,
            "role": step["role"],
            "target": deepcopy(step["target"]),
            "function_name": step["function_name"],
            "arguments": deepcopy(step["arguments"]),
            "depends_on": deepcopy(step["depends_on"]),
            "initial_state": "pending" if waiting_for else "ready",
            "waiting_for": waiting_for,
            "initially_dispatchable": step_id in initial_dispatchable,
        })

    return {
        "scheduler_version": SCHEDULER_VERSION,
        "mode": "dry_run",
        "will_execute": False,
        "failure_policy": {
            "mode": failure_policy,
            "halt_new_dispatch_on_failure": True,
            "pending_or_ready_steps": "cancelled",
            "running_steps": "cancel_requested",
            "note": "實際停止 running hardware 仍需各 device service 提供安全 stop/cancel",
        },
        "dispatch_policy": {
            "mode": "event_driven",
            "max_concurrent_steps_per_device": 1,
            "ready_tiebreak": "resource_aware_resolved_step_order",
            "end_effector_capacity": 1,
            "pick_requires_free_end_effector": True,
            "place_requires_matching_held_object": True,
        },
        "resource_state": {
            "initial_holding": deepcopy(
                _normalize_initial_holding(initial_holding)
            ),
            "planning_only": True,
            "decisions": deepcopy(resource_decisions),
        },
        "state_model": [
            "pending",
            "ready",
            "running",
            "succeeded",
            "failed",
            "cancel_requested",
            "cancelled",
        ],
        "topological_order": topological_order,
        "initial_ready_steps": initial_ready,
        "initial_dispatchable_steps": initial_dispatchable,
        "device_workloads": device_workloads,
        "steps": preview_steps,
    }
