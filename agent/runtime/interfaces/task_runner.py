"""Independent Agent task runner: Plan -> preflight -> Action.

- Existing HTTP agent_service.command uses run_task(), which remains MOCK ONLY.
- Real motion is only reachable via operator-controlled prepare/approve/execute.
- Manual Runtime is never started by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from copy import deepcopy
from contextlib import contextmanager
from threading import Lock
from typing import Any

from agent.runtime.adapters.agent_action import execute_action, validate_action_step

_TASK_LOCK = Lock()
_APPROVAL_LOCK = Lock()
_APPROVALS: dict[str, tuple[str, float]] = {}
_APPROVAL_TTL_SECONDS = 90


class AgentTaskError(RuntimeError):
    pass


def _checked_plan(user_text: str, envelope: dict[str, Any]) -> dict[str, Any]:
    """Preflight the entire Plan before dispatching any Action."""
    if not isinstance(envelope, dict) or envelope.get("result") is not True:
        raise AgentTaskError("Plan Service 規劃失敗")
    plan = envelope.get("data")
    if not isinstance(plan, dict):
        raise AgentTaskError("Plan Service 缺少 data object")
    if plan.get("mode") != "mock" or plan.get("physical_execution_allowed") is not False:
        raise AgentTaskError("Plan 權限宣告無效，拒絕執行")
    if plan.get("user_text") != user_text.strip():
        raise AgentTaskError("Plan 的 user_text 不符合原始指令")
    if not isinstance(plan.get("plan_id"), str) or not plan["plan_id"]:
        raise AgentTaskError("Plan ID 無效")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not (
            1 <= len(steps) <= 3 or (len(steps) == 0 and plan.get("complete") is True)):
        raise AgentTaskError("Plan Steps 數量無效")
    checked = [validate_action_step(step) for step in steps]
    ids = [step["step_id"] for step in checked]
    if len(ids) != len(set(ids)):
        raise AgentTaskError("Plan Step IDs 重複")
    result = deepcopy(plan)
    result["steps"] = checked
    return result


def prepare_task(
    user_text: str, *, provider: str = "local",
    planning_options: dict[str, Any] | None = None,
    observation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the real Plan LLM (and its WorldModel) but do not move hardware."""
    if not isinstance(user_text, str) or not user_text.strip():
        raise AgentTaskError("user_text 必須是非空字串")
    if provider not in ("local", "openai"):
        raise AgentTaskError("provider 僅支援 local/openai")
    if not _TASK_LOCK.acquire(blocking=False):
        raise AgentTaskError("目前已有 Agent Task 正在執行")
    try:
        from services import plan_service
        options = {"user_text": user_text.strip(), "provider": provider,
                   "planning_options": planning_options}
        if observation_context is not None:
            options["observation_context"] = observation_context
        envelope = plan_service.plan_task(**options)
        return _checked_plan(user_text, envelope)
    finally:
        _TASK_LOCK.release()


def real_confirmation_phrase(plan: dict[str, Any]) -> str:
    """Operator must type the exact action / arm / plan_id displayed in CLI."""
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or len(plan["steps"]) != 1:
        raise AgentTaskError("Real 僅支援單一步驟計畫")
    step = validate_action_step(plan["steps"][0])
    args = step["arguments"]
    # The Registry may contain actions which do not require an arm_name.
    extra = f" {args['direction']}" if step["action_name"] == "observe_world" and "direction" in args else ""
    return f"EXECUTE {plan.get('plan_id')} {step['action_name']} {args.get('arm_name', '-')}{extra}"


def _plan_fingerprint(plan: dict[str, Any]) -> str:
    subset = {"plan_id": plan.get("plan_id"), "user_text": plan.get("user_text"),
              "steps": plan.get("steps"), "mode": plan.get("mode"),
              "worldmodel": plan.get("worldmodel")}
    payload = json.dumps(subset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def approve_real_plan(plan: dict[str, Any], *, operator_input: str) -> str:
    """One-time 90s permit, ONLY after a full local operator confirmation."""
    if os.getenv("AGENT_REAL_ACTION_ENABLED") != "1":
        raise AgentTaskError("實機動作尚未透過 AGENT_REAL_ACTION_ENABLED=1 授權")
    if not isinstance(plan, dict) or plan.get("physical_execution_allowed") is not False or plan.get("mode") != "mock":
        raise AgentTaskError("Plan 來源權限宣告無效")
    expected = real_confirmation_phrase(plan)
    if operator_input != expected:
        raise AgentTaskError("操作員確認字串不相符；禁止執行 Real Action")
    step = validate_action_step(plan["steps"][0])
    # One trusted Registry + callable contract; adapter rechecks at dispatch.
    from agent.runtime.adapters.agent_action import _assert_real_allowed
    _assert_real_allowed(step)
    token = secrets.token_urlsafe(32)
    with _APPROVAL_LOCK:
        _APPROVALS.clear()  # Only the latest approval can be used.
        _APPROVALS[token] = (_plan_fingerprint(plan), time.monotonic() + _APPROVAL_TTL_SECONDS)
    return token



# AGENT_SINGLE_MODE_REQUEST_SCOPE_V1
from contextvars import ContextVar as _AgentContextVar
from contextlib import contextmanager as _agent_contextmanager

_AGENT_AUTO_REAL_REQUEST = _AgentContextVar("agent_auto_real_request", default=False)

@_agent_contextmanager
def _agent_auto_real_request():
    """Grant autonomous Real Execute intent to this request only.

    ContextVar avoids toggling process-global environment variables in Flask.
    Existing plan-token validation, Action preflight and hardware gates remain.
    """
    token = _AGENT_AUTO_REAL_REQUEST.set(True)
    try:
        yield
    finally:
        _AGENT_AUTO_REAL_REQUEST.reset(token)


def authorize_autonomous_plan(plan: dict[str, Any]) -> str:
    """Machine-generated one-shot permit for dedicated unattended CLI only.

    No operator phrase is forged. The request-scoped intent, one-shot token,
    trusted Action Registry and service callable contract are enforced here.
    Action Service owns any hardware movement checks and stopping conditions.
    HTTP Agent run_task remains Mock-only.
    """
    # The request must enter via auto.py's explicit Real Execute branch.
    # Dry Run / information previews never enter this context.
    if not _AGENT_AUTO_REAL_REQUEST.get():
        raise AgentTaskError("目前請求未選擇 Real Execute，不允許實機派送")
    if (not isinstance(plan, dict) or plan.get("physical_execution_allowed") is not False
            or plan.get("mode") != "mock"):
        raise AgentTaskError("Plan 來源權限宣告無效")
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise AgentTaskError("自動實機僅支援單一步驟")
    step = validate_action_step(steps[0])
    from agent.runtime.adapters.agent_action import _assert_real_allowed
    _assert_real_allowed(step)
    token = secrets.token_urlsafe(32)
    with _APPROVAL_LOCK:
        _APPROVALS.clear()
        _APPROVALS[token] = (_plan_fingerprint(plan), time.monotonic() + _APPROVAL_TTL_SECONDS)
    return token


def _consume_approval(plan: dict[str, Any], token: str | None) -> None:
    if not isinstance(token, str) or not token:
        raise AgentTaskError("Real Action 缺少一次性操作員授權")
    with _APPROVAL_LOCK:
        value = _APPROVALS.pop(token, None)
    if value is None:
        raise AgentTaskError("Real Action 授權不存在或已被使用")
    fingerprint, expires = value
    if time.monotonic() > expires or fingerprint != _plan_fingerprint(plan):
        raise AgentTaskError("Real Action 授權逾時或 Plan 已變更")


def _execute_checked(plan: dict[str, Any], *, mode: str, approval_token: str | None = None) -> dict[str, Any]:
    steps = [validate_action_step(step) for step in plan["steps"]]
    if not steps and plan.get("complete") is True and mode == "mock":
        return {"plan_id": plan["plan_id"], "mode": "mock",
                "planner": plan.get("planner"), "worldmodel": plan.get("worldmodel"),
                "steps": [], "answer": "Plan 根據觀察判定目標已達成，沒有派送 Action。"}
    if mode == "real":
        if len(steps) != 1:
            raise AgentTaskError("Real 目前只允許單一步驟，不允許連續多步實機動作")
        _consume_approval(plan, approval_token)
    elif mode != "mock":
        raise AgentTaskError("mode 僅支援 mock/real")

    records = []
    for step in steps:
        result = execute_action(step, mode=mode, operator_authorized=(mode == "real"))
        success = result.get("result") is True
        if mode == "mock":
            success = success and result.get("simulated") is True
            status = "mock_completed" if success else "mock_failed"
        else:
            success = success and result.get("simulated") is False
            if not success:
                status = "real_failed"
            elif result.get("physical_completion_verified") is True and result.get("physical_completed") is True:
                status = "real_completed_verified"
            elif result.get("physical_completion_verified") is True and result.get("physical_completed") is False:
                status = "real_not_completed"
            else:
                status = "real_dispatched_unverified"
        record = deepcopy(step)
        record["status"] = status
        record["execution"] = deepcopy(result)
        records.append(record)
        if not success:
            raise AgentTaskError("Action Service 執行失敗；不執行後續動作")
    if mode == "mock":
        answer = f"Mock 模擬完成 {len(records)} 步；實機沒有移動，未驗證物理動作是否完成。"
    elif all(item["status"] == "real_completed_verified" for item in records):
        answer = "Action Service 已確認動作完成。"
    elif any(item["status"] == "real_not_completed" for item in records):
        answer = "Action Service 已確認動作尚未完成；不自動重試。"
    else:
        answer = "Action 已派送，但 Action Service 沒有回傳已驗證的完成結果；不自動續行。"
    return {
        "plan_id": plan["plan_id"],
        "mode": mode,
        "planner": plan.get("planner"),
        "worldmodel": plan.get("worldmodel"),
        "steps": records,
        "answer": answer,
    }





@contextmanager
def _real_process_lock():
    """Prevent overlapping operator sessions using this Agent gateway.

    Other components do not necessarily honor this file lock: it is NOT a
    substitute for controller-level exclusive ownership of the robot.
    """
    import fcntl
    with open("/tmp/robot_agent_real_action.lock", "a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AgentTaskError("另一個 Agent Real Action 程序正在執行") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def execute_prepared_plan(
    plan: dict[str, Any], *, mode: str = "mock", approval_token: str | None = None,
) -> dict[str, Any]:
    """Execute the exact previously previewed plan, WITHOUT re-calling an LLM."""
    if not _TASK_LOCK.acquire(blocking=False):
        raise AgentTaskError("目前已有 Agent Task 正在執行")
    try:
        if not isinstance(plan, dict) or not isinstance(plan.get("user_text"), str):
            raise AgentTaskError("缺少有效的 prepared plan")
        checked = _checked_plan(plan["user_text"], {"result": True, "data": plan})
        if mode == "real":
            with _real_process_lock():
                return _execute_checked(checked, mode=mode, approval_token=approval_token)
        return _execute_checked(checked, mode=mode, approval_token=approval_token)
    finally:
        _TASK_LOCK.release()


def run_task(
    user_text: str, *, provider: str = "local",
    planning_options: dict[str, Any] | None = None, mode: str = "mock",
) -> dict[str, Any]:
    """Backward-compatible Agent HTTP path; ALWAYS MOCK, even if forged mode."""
    if mode != "mock":
        raise AgentTaskError("Agent HTTP run_task 僅支援 Mock；實機須使用操作員準備／確認流程")
    if not isinstance(user_text, str) or not user_text.strip():
        raise AgentTaskError("user_text 必須是非空字串")
    if provider not in ("local", "openai"):
        raise AgentTaskError("provider 僅支援 local/openai")
    if not _TASK_LOCK.acquire(blocking=False):
        raise AgentTaskError("目前已有 Agent Task 正在執行")
    try:
        from services import plan_service
        envelope = plan_service.plan_task(
            user_text=user_text.strip(), provider=provider,
            planning_options=planning_options,
        )
        checked = _checked_plan(user_text, envelope)
        return _execute_checked(checked, mode="mock")
    finally:
        _TASK_LOCK.release()


def refresh_after_observe(report: dict[str, Any], *, provider: str = "local") -> dict[str, Any]:
    """Fresh WorldModel capture strictly AFTER a verified observe_world move.

    This does not dispatch robot motion. Do not reuse a pre-move scene.
    """
    if not isinstance(report, dict) or report.get("mode") != "real":
        raise AgentTaskError("只接受真實觀察動作的回傳")
    steps = report.get("steps")
    if (not isinstance(steps, list) or len(steps) != 1
            or steps[0].get("action_name") != "observe_world"
            or steps[0].get("status") != "real_completed_verified"
            or (steps[0].get("execution") or {}).get("physical_completion_verified") is not True):
        raise AgentTaskError("觀察動作未確認到位，拒絕取得後續 Plan 的場景")
    args = steps[0]["arguments"]
    arm_name = args["arm_name"]
    from services import worldmodel_service
    observation = worldmodel_service.describe_scene(
        camera_name=arm_name, vlm_backend=provider,
    )
    if not isinstance(observation, dict) or observation.get("result") is not True:
        raise AgentTaskError("移動後 WorldModel 觀察失敗：" + str(observation)[:250])
    data = observation.get("data")
    scene = data.get("scene") if isinstance(data, dict) else None
    if not isinstance(scene, dict):
        raise AgentTaskError("移動後 WorldModel 沒有 scene")
    return {"verified_arrival": True, "arm_name": arm_name,
            "direction": args["direction"], "camera_name": arm_name,
            "scene": deepcopy(scene)}
