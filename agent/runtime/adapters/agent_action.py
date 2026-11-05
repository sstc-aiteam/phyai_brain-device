"""Registry-driven Agent -> Action Service adapter.

`agent.tools` is the ONLY Action allowlist. The function with the same name
must exist on `services.action_service` and accept the registered arguments.
Robot motion safety, per-action readiness, batching, end-effector limits and
completion determination belong to Action Service, NOT this adapter.

This gateway keeps *software* boundaries: exact registered arguments, an
explicit Real authorization, and no automatic success without a verified
completion returned by Action Service. It is not a trajectory safety checker.
"""
from __future__ import annotations

import inspect
import math
import os
import time
from typing import Any

# An upper bound passed to services whose signature exposes `max_batches`.
# No forced loop: Action Service may stop earlier and owns all batch checks.
ACTION_BATCH_LIMIT = 20

# Kept solely for the explicit `python -m agent.auto --home ...` route.
_HOME_MAX_DELTA_RAD = 1.80
_HOME_MAX_EXPLICIT_DELTA_RAD = 3.14
_HOME_PATH_APPROVAL_ENV = "AGENT_REAL_HOME_PATH_APPROVAL"
_HOME_POSTFLIGHT_TOLERANCE_RAD = 0.05
_HOME_POSTFLIGHT_TIMEOUT_SECONDS = 3.0
_HOME_POSTFLIGHT_POLL_SECONDS = 0.20


class AgentActionRejected(ValueError):
    """Invalid Registry action or incompatible Action Service contract."""


def _check_value(key: str, value: Any, spec: dict[str, Any]) -> None:
    """Validate the small JSON-Schema subset defined in agent/tools.py."""
    kind = spec.get("type")
    types = {
        "string": lambda x: isinstance(x, str) and bool(x.strip()),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "boolean": lambda x: type(x) is bool,
        "array": lambda x: isinstance(x, list),
        "object": lambda x: isinstance(x, dict),
    }
    if kind not in types or not types[kind](value):
        raise AgentActionRejected(f"Action 參數 {key!r} 不符合 Registry type={kind!r}")
    if "enum" in spec and value not in spec["enum"]:
        raise AgentActionRejected(f"Action 參數 {key!r} 不符合 Registry enum")
    if kind in ("number", "integer"):
        if "minimum" in spec and value < spec["minimum"]:
            raise AgentActionRejected(f"Action 參數 {key!r} 小於 minimum")
        if "maximum" in spec and value > spec["maximum"]:
            raise AgentActionRejected(f"Action 參數 {key!r} 大於 maximum")


def validate_action_step(step: dict[str, Any]) -> dict[str, Any]:
    """Plan must refer to exactly one trusted tool and only its declared args."""
    if not isinstance(step, dict) or set(step) != {"step_id", "action_name", "arguments", "status"}:
        raise AgentActionRejected("Action Step 欄位不符合規格")
    if not isinstance(step["step_id"], str) or not step["step_id"].strip():
        raise AgentActionRejected("step_id 無效")
    if step["status"] != "planned":
        raise AgentActionRejected("只接受 status=planned 的 Step")
    name, args = step["action_name"], step["arguments"]
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise AgentActionRejected("Action 名稱無效")
    if not isinstance(args, dict):
        raise AgentActionRejected("arguments 必須是 dict")

    from agent.tools import get_tool
    tool = get_tool(name)
    if not isinstance(tool, dict) or tool.get("module") != "action":
        raise AgentActionRejected(f"Action 未在 agent.tools 註冊：{name!r}")
    specs, required = tool.get("parameters"), tool.get("required")
    if not isinstance(specs, dict) or not isinstance(required, list) or any(
        not isinstance(k, str) for k in required
    ) or not set(required) <= set(specs):
        raise AgentActionRejected(f"Action Registry 規格無效：{name!r}")
    if set(args) - set(specs) or not set(required) <= set(args):
        raise AgentActionRejected(f"Action {name!r} 的參數不符合 Registry")
    for key, value in args.items():
        spec = specs[key]
        if not isinstance(spec, dict):
            raise AgentActionRejected(f"Action Registry 參數規格無效：{key}")
        _check_value(key, value, spec)
    return {"step_id": step["step_id"], "action_name": name,
            "arguments": dict(args), "status": "planned"}


def _resolve_action(step: dict[str, Any]):
    """Resolve ONLY a Registry-approved public Action Service function.

    The explicit function signature is the source of service-side compatibility.
    `max_batches` is passed only to services which declare it by name.
    """
    checked = validate_action_step(step)
    from services import action_service
    name = checked["action_name"]
    function = getattr(action_service, name, None)
    if not callable(function):
        raise AgentActionRejected(f"Action Service 不存在可呼叫的 {name}()")
    try:
        signature = inspect.signature(function)
    except (ValueError, TypeError) as exc:
        raise AgentActionRejected(f"無法檢查 {name}() 函式簽名") from exc
    kwargs = dict(checked["arguments"])
    # Service default is not relied on when an explicit batch limit is supported.
    if "max_batches" in signature.parameters:
        param = signature.parameters["max_batches"]
        if param.kind not in (inspect.Parameter.KEYWORD_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise AgentActionRejected(f"{name}.max_batches 不能以關鍵字指定")
        kwargs["max_batches"] = ACTION_BATCH_LIMIT
    try:
        signature.bind(**kwargs)
    except TypeError as exc:
        raise AgentActionRejected(f"{name}() 與 agent.tools 參數不相容：{exc}") from exc
    return checked, function, kwargs


def _assert_real_allowed(step: dict[str, Any]) -> None:
    """Compatibility entry for Task Runner: trust Registry + callable contract."""
    _resolve_action(step)


def inspect_real_action_step(step: dict[str, Any]) -> dict[str, Any]:
    """Read-only software contract check; NOT a robot motion preflight."""
    preview = {
        "read_only": True, "service_invoked": False, "action_dispatch_count": 0,
        "gateway_preflight_passed": False, "trajectory_collision_checked": False,
        "hardware_safety_checked_by_agent": False,
        "safety_owner": "Action Service", "error": None,
    }
    try:
        checked, _, kwargs = _resolve_action(step)
    except (AgentActionRejected, ValueError, TypeError, ImportError) as exc:
        preview["error"] = f"{type(exc).__name__}: {exc}"
    else:
        preview.update(gateway_preflight_passed=True,
                       action_name=checked["action_name"],
                       service_function=f"action_service.{checked['action_name']}",
                       service_arguments=kwargs,
                       max_batches=kwargs.get("max_batches"))
    return preview


def _completion_from_service(raw: dict[str, Any]) -> dict[str, Any]:
    """Trust Action Service's explicit *verified* outcome, not result=True.

    Accepted service contracts (root or data):
      - task_completed_verified: true -> verified complete
      - physical_completed: bool, physical_completion_verified: true
      - completed: bool, completion_verified: true
    Anything else: dispatched, unverified; no auto-success or auto-retry.
    Agent does NOT run CheckAction or check end-effector displacement.
    """
    containers = [raw]
    if isinstance(raw.get("data"), dict):
        containers.append(raw["data"])
    observations = []
    for item in containers:
        if item.get("task_completed_verified") is True:
            observations.append(True)
        if item.get("physical_completion_verified") is True and type(item.get("physical_completed")) is bool:
            observations.append(item["physical_completed"])
        if item.get("completion_verified") is True and type(item.get("completed")) is bool:
            observations.append(item["completed"])
    if observations and len(set(observations)) == 1:
        return {"physical_completed": observations[0],
                "physical_completion_verified": True,
                "completion_source": "action_service"}
    return {"physical_completed": None, "physical_completion_verified": False,
            "completion_source": "conflicting_fields" if observations else "no_verified_completion"}


def execute_action(step: dict[str, Any], *, mode: str = "mock",
                   operator_authorized: bool = False) -> dict[str, Any]:
    """Dispatch one Action; side effects only in explicit, authorized Real mode."""
    checked = validate_action_step(step)
    if mode == "mock":
        return {"result": True, "simulated": True, "physical_completed": False,
                "physical_completion_verified": False,
                "action_name": checked["action_name"], "arguments": checked["arguments"],
                "message": "Mock Action：未呼叫實機或宣稱物理完成"}
    if mode != "real":
        raise AgentActionRejected("mode 僅支援 mock/real")
    if operator_authorized is not True:
        raise AgentActionRejected("缺少本次 Real Action 授權")

    checked, function, kwargs = _resolve_action(checked)
    # Action Service owns movement safety, batch loop, completion and stopping.
    raw = function(**kwargs)
    if not isinstance(raw, dict) or raw.get("result") is not True:
        raise RuntimeError(f"Action Service {checked['action_name']}() 未確認成功；禁止自動重試：{raw!r}")
    completion = _completion_from_service(raw)
    return {
        "result": True, "simulated": False, **completion,
        "action_name": checked["action_name"], "arguments": checked["arguments"],
        "service_arguments": kwargs, "service_response": raw,
        "message": ("Action Service 已確認動作完成" if completion["physical_completed"] is True
                    and completion["physical_completion_verified"] else
                    "Action Service 已確認動作未完成" if completion["physical_completed"] is False
                    and completion["physical_completion_verified"] else
                    "Action Service 已回覆，但沒有明確的完成驗證，不自動重試"),
    }


# --- Legacy, explicitly-invoked HOME route only; not a Registry Action. ---

def _read_arm_ready(arm_name: str) -> dict[str, Any]:
    from services import arm_service
    envelope = arm_service.get_arm_status(arm_name=arm_name)
    if not isinstance(envelope, dict) or envelope.get("result") is not True:
        raise AgentActionRejected("無法確認手臂狀態，禁止 Home")
    arms = (envelope.get("data") or {}).get("arms")
    if not isinstance(arms, list):
        raise AgentActionRejected("Arm Service 回傳格式錯誤")
    matching = [item for item in arms if isinstance(item, dict) and item.get("arm_name") == arm_name]
    if len(matching) != 1 or not isinstance(matching[0].get("status"), dict):
        raise AgentActionRejected("指定手臂狀態無效")
    status = matching[0]["status"]
    if (status.get("connected") is not True or status.get("ready") is not True
            or status.get("moving") is not False or status.get("safety_mode") != 1
            or any(status.get(key) is not False for key in ("emergency_stop", "protective_stop", "fault"))
            or status.get("data_valid") is False or status.get("data_stale") is True):
        raise AgentActionRejected("Home 手臂狀態不安全或未知")
    age = status.get("data_age_seconds")
    if age is not None:
        try:
            age_value = float(age)
        except (ValueError, TypeError) as exc:
            raise AgentActionRejected("Home 遙測時間無效") from exc
        if not math.isfinite(age_value) or age_value < 0 or age_value > 2.0:
            raise AgentActionRejected("Home 遙測已過期")
    joints = status.get("joints")
    if not isinstance(joints, (list, tuple)) or len(joints) != 6:
        raise AgentActionRejected("Home 目前關節無效")
    try:
        joints = [float(x) for x in joints]
    except (ValueError, TypeError) as exc:
        raise AgentActionRejected("Home 關節數值無效") from exc
    if not all(math.isfinite(x) for x in joints):
        raise AgentActionRejected("Home 關節包含非有限數值")
    return {"connected": True, "ready": True, "moving": False,
            "safety_mode": 1, "joints": joints}


def _target_default_joints(arm_name: str) -> list[float]:
    import config
    arms = getattr(config, "ARMS", None)
    item = arms.get(arm_name) if isinstance(arms, dict) else None
    poses = item.get("poses") if isinstance(item, dict) else None
    target = poses.get("default_joints") if isinstance(poses, dict) else None
    if not isinstance(target, (list, tuple)) or len(target) != 6:
        raise AgentActionRejected("Home 目標關節無效")
    try:
        target = [float(x) for x in target]
    except (ValueError, TypeError) as exc:
        raise AgentActionRejected("Home 目標數值無效") from exc
    if not all(math.isfinite(x) for x in target):
        raise AgentActionRejected("Home 目標包含非有限數值")
    return target


def _home_move_delta_gate(arm_name: str) -> tuple[float, bool]:
    approval = os.environ.get(_HOME_PATH_APPROVAL_ENV, "").strip()
    if not approval:
        return _HOME_MAX_DELTA_RAD, False
    parts = approval.split(":")
    if len(parts) != 2 or parts[0] != arm_name:
        raise AgentActionRejected("Home 授權應為 arm:max_delta")
    try:
        bound = float(parts[1])
    except ValueError as exc:
        raise AgentActionRejected("Home 授權數值無效") from exc
    if not math.isfinite(bound) or not (_HOME_MAX_DELTA_RAD < bound <= _HOME_MAX_EXPLICIT_DELTA_RAD):
        raise AgentActionRejected("Home 授權超出限制")
    return bound, True


def _read_real_move_details(step: dict[str, Any]) -> dict[str, Any]:
    if step.get("action_name") != "move_arm_default":
        raise AgentActionRejected("此診斷僅供 --home 使用")
    arm_name = (step.get("arguments") or {}).get("arm_name")
    if arm_name not in ("left", "right"):
        raise AgentActionRejected("Home 手臂名稱無效")
    status = _read_arm_ready(arm_name)
    target = _target_default_joints(arm_name)
    deltas = [abs(dst - src) for src, dst in zip(status["joints"], target)]
    limit, exception_approved = _home_move_delta_gate(arm_name)
    return {"arm_name": arm_name, "current_joints": status["joints"],
            "target_joints": target, "joint_deltas_rad": [round(x, 6) for x in deltas],
            "max_joint_delta_rad": max(deltas), "max_allowed_joint_delta_rad": limit,
            "large_home_move_exception": exception_approved, "arm_status": status,
            "trajectory_collision_checked": False}


def _assert_joint_delta_gate(details: dict[str, Any]) -> None:
    if details["max_joint_delta_rad"] > details["max_allowed_joint_delta_rad"]:
        raise AgentActionRejected("Home 關節位移超過當次允許上限")


def _postflight_check(arm_name: str, target: list[float]) -> dict[str, Any]:
    deadline = time.monotonic() + _HOME_POSTFLIGHT_TIMEOUT_SECONDS
    consecutive = 0
    last: dict[str, Any] | None = None
    last_error: str | None = None
    while True:
        try:
            snapshot = _read_arm_ready(arm_name)
            errors = [abs(actual - wanted) for actual, wanted in zip(snapshot["joints"], target)]
            last = {"observed_joints": snapshot["joints"],
                    "joint_errors_rad": [round(x, 6) for x in errors],
                    "max_joint_error_rad": max(errors)}
            if max(errors) <= _HOME_POSTFLIGHT_TOLERANCE_RAD:
                consecutive += 1
                if consecutive >= 2:
                    return {"verified": True, "reached": True,
                            "tolerance_rad": _HOME_POSTFLIGHT_TOLERANCE_RAD, **last}
            else:
                consecutive = 0
        except AgentActionRejected as exc:
            consecutive = 0
            last_error = str(exc)
        if time.monotonic() >= deadline:
            break
        time.sleep(_HOME_POSTFLIGHT_POLL_SECONDS)
    return {"verified": False, "reached": False if last is not None else None,
            "tolerance_rad": _HOME_POSTFLIGHT_TOLERANCE_RAD,
            "error": last_error or "Home 遙測驗證逾時", **(last or {})}
