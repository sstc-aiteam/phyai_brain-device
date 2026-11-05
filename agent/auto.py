#!/usr/bin/env python3
"""Agent autonomous single-job CLI. No operator prompts or manual movement.

Two modes:
  --home left/right: guarded return to configured default pose (not a Plan tool)
  --text ...: Agent -> safe information Service, or Plan -> Action / Observe -> Replan

Unverified motion routes STOP rather than relaxing joint limits. This CLI does
not implement collision checking and cannot certify a path as collision-free.
"""
from __future__ import annotations

import argparse
import json
from contextvars import ContextVar
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Legacy explicit --home has its own guard; Action safety is delegated to
# services.action_service for all registry-driven --text actions.
HOME_MAX_JOINT_DELTA_RAD = 1.80


# Request-local capture: Flask can reuse the SAME auto.py logic without
# redirecting process-global stdout (unsafe with threaded Flask requests).
_AUTO_EVENT_SINK: ContextVar[list | None] = ContextVar("agent_auto_event_sink", default=None)


def _print(label: str, obj) -> None:
    sink = _AUTO_EVENT_SINK.get()
    if sink is not None:
        sink.append({"phase": label,
                     "data": json.loads(json.dumps(obj, ensure_ascii=False, default=str))})
        return
    print(f"\n=== {label} ===", flush=True)
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str), flush=True)




def _set_nonactivating_left_gripper() -> None:
    """Prevent lazy driver creation from resetting/activating left gripper."""
    import config
    gripper = config.GRIPPERS.get("left", {})
    kwargs = gripper.get("kwargs")
    if gripper.get("driver") == "robotiq_eseries" and isinstance(kwargs, dict):
        kwargs["auto_activate"] = False
    # Gripper status polling can still send URScript to the UR controller.





def _strict_home_path(preview: dict) -> None:
    """Guard the dedicated --home command (not an Action Registry tool)."""
    delta = float(preview["max_joint_delta_rad"])
    limit = HOME_MAX_JOINT_DELTA_RAD
    if delta > limit:
        raise RuntimeError(
            f"Home 關節位移 {delta:.3f} rad > {limit:.2f} rad；"
            "尚未核對碰撞路徑，不允許透過一般 Home 入口移動。"
        )


def _run_home(arm_name: str, *, dry_run: bool, preflight_only: bool,
              ignore_delta_gate: bool = False) -> int:
    if dry_run:
        _print("HOME MOCK", {"arm_name": arm_name, "action": "move_arm_default",
                             "simulated": True, "physical_completed": False})
        return 0
    from agent.runtime.adapters.agent_action import (
        _assert_joint_delta_gate, _read_real_move_details, _postflight_check,
    )
    from agent.runtime.interfaces.task_runner import _real_process_lock
    from services import arm_service
    step = {"step_id": "auto_home", "action_name": "move_arm_default",
            "arguments": {"arm_name": arm_name}, "status": "planned"}

    # Protect preflight + dispatch under same process lock.
    with _real_process_lock():
        snapshot = _read_real_move_details(step)
        _print("AUTO HOME PREFLIGHT", snapshot)
        # Explicit temporary home-only override. This does not check the
        # swept trajectory; operator must independently validate the route.
        if ignore_delta_gate:
            _print("HOME JOINT DELTA GATE DISABLED", {
                "arm_name": arm_name,
                "max_joint_delta_rad": snapshot["max_joint_delta_rad"],
                "target_joints": snapshot["target_joints"],
                "trajectory_collision_checked": False,
                "warning": "只略過 Home 關節位移門檻；不代表路徑安全。",
            })
        else:
            _strict_home_path(snapshot)
            _assert_joint_delta_gate(snapshot)
        if preflight_only:
            return 0
        latest = _read_real_move_details(step)
        if not ignore_delta_gate:
            _strict_home_path(latest)
            _assert_joint_delta_gate(latest)
        if latest["target_joints"] != snapshot["target_joints"]:
            raise RuntimeError("執行前 Home 目標改變，拒絕移動")
        drift = max(abs(a - b) for a, b in zip(snapshot["current_joints"], latest["current_joints"]))
        if drift > 0.02:
            raise RuntimeError(f"執行前起始姿態已改變 {drift:.3f} rad，拒絕移動")
        result = arm_service.move_arm_default(
            arm_name=arm_name, speed=0.05, acceleration=0.05, wait=True,
        )
        _print("AUTO HOME RESULT", result)
        if not isinstance(result, dict) or result.get("result") is not True:
            raise RuntimeError("Home 派送後未回報成功；不得自動重試")
        postflight = _postflight_check(arm_name, latest["target_joints"])
        _print("AUTO HOME POSTFLIGHT", postflight)
        if postflight.get("verified") is not True or postflight.get("reached") is not True:
            raise RuntimeError("無法確認 Home 到位；不自動重試")
    return 0



# Explicit whitelist for the autonomous CLI's information-only dispatch.
# Gripper status is intentionally excluded: lazy Robotiq initialization may
# reset/activate hardware, which is not safe to call a zero-motion query.
_INFORMATION_CAPABILITIES = frozenset({
    "describe_scene",
    "get_object_state",
    "get_arm_status",
    "get_camera_status",
    "get_arm_pose",
    "get_arm_joints",
})


def _run_information_query(text: str, selected: dict, *, provider: str, camera: str,
                           dry_run: bool, preflight_only: bool) -> int:
    """Execute a selected registered information capability without Plan/Action.

    Do not trust the LLM-provided side_effect/execution_enabled fields. The
    registry and the existing capability executor enforce the permissions.
    """
    from agent.capability_registry import get_capability

    tool = selected.get("tool_name")
    if tool not in _INFORMATION_CAPABILITIES:
        raise RuntimeError(f"自動入口未開放的資訊 Capability：{tool}")
    spec = get_capability(tool)
    if not spec.execution_enabled or spec.side_effect != "none":
        raise RuntimeError(f"Capability 不允許資訊查詢：{tool}")
    arguments = selected.get("arguments")
    if not isinstance(arguments, dict):
        raise RuntimeError("Agent Capability arguments 必須為 dict")
    arguments = dict(arguments)
    # The CLI's --camera is the chosen visual source. Do not alter specific
    # camera_name arguments provided by the Agent, or scope device counts.
    if tool == "describe_scene" and "camera_name" not in arguments:
        arguments["camera_name"] = camera
    from agent.capability_selector import _validate_arguments
    arguments = _validate_arguments(tool, arguments)
    query = {**selected, "arguments": arguments}
    if dry_run or preflight_only:
        _print("INFORMATION PREVIEW", {
            "tool_name": tool, "arguments": arguments,
            "service_invoked": False, "robot_motion_dispatched": False,
        })
        return 0

    from time import perf_counter
    from agent.capability_executor import execute_capability
    from services.agent_service import _capability_result_data, _guard_equipment_query
    _guard_equipment_query(text, tool)
    started = perf_counter()
    execution = execute_capability(query)
    result = _capability_result_data(text, query, execution, started, provider=provider)
    # Keep the source response visible for diagnosis. This can include
    # images/scene objects, positions or joint telemetry, not just prose.
    _print("INFORMATION RESULT", {
        "tool_name": tool,
        "arguments": execution["arguments"],
        "answer": result.get("answer"),
        "service_data": execution["service_response"].get("data"),
        "robot_motion_dispatched": False,
    })
    return 0


def _run_text(text: str, *, provider: str, camera: str,
              dry_run: bool, preflight_only: bool) -> int:
    from agent.capability_selector import select_capability

    selected = select_capability(text, provider=provider)
    _print("AGENT", selected)
    tool = selected.get("tool_name")
    if tool != "plan_task":
        return _run_information_query(text, selected, provider=provider, camera=camera,
                                      dry_run=dry_run,
                                      preflight_only=preflight_only)
    if selected.get("arguments") != {}:
        raise RuntimeError("Agent plan_task 參數必須為空；拒絕動作")

    from agent.runtime.interfaces.task_runner import (
        prepare_task, execute_prepared_plan, refresh_after_observe,
        authorize_autonomous_plan,
    )

    options = {"camera_name": camera}
    plan = prepare_task(text, provider=provider, planning_options=options)
    _print("PLAN", {"plan_id": plan["plan_id"], "planner": plan.get("planner"),
                    "steps": plan["steps"],
                    "followup_after_observe": plan.get("followup_after_observe")})
    if plan.get("complete") is True and not plan["steps"]:
        print("任務已完成，不派送動作。", flush=True)
        return 0

    if dry_run:
        # Mock executes only the FIRST action; no physical post-observation state.
        report = execute_prepared_plan(plan, mode="mock")
        _print("MOCK", report)
        return 0

    from agent.runtime.adapters.agent_action import inspect_real_action_step

    def run_stage(stage: dict, *, preview_only: bool) -> dict | None:
        if len(stage["steps"]) != 1:
            raise RuntimeError("自動實機每回合只允許一個 Action")
        step = stage["steps"][0]
        action_name = step["action_name"]
        # Registry and matching Action Service callable are the contract.
        # This is NOT a motion-safety check; the Action Service owns that.
        preflight = inspect_real_action_step(step)
        _print("AUTO PREFLIGHT", preflight)
        if preflight.get("gateway_preflight_passed") is not True:
            raise RuntimeError("Action Service 介面檢查失敗：" + str(preflight.get("error")))
        if preview_only:
            print("Preflight 結束：沒有派送動作。", flush=True)
            return None
        # Issue a short-lived, one-use internal token; never fake human consent.
        # execute_prepared_plan and the action adapter validate the plan again.
        # One request-scoped switch: dry_run uses mock; Real Execute grants
        # autonomous dispatch permission only for this in-flight request.
        # Do not set os.environ here: Flask serves concurrent requests.
        from agent.runtime.interfaces.task_runner import _agent_auto_real_request
        with _agent_auto_real_request():
            token = authorize_autonomous_plan(stage)
            result = execute_prepared_plan(stage, mode="real", approval_token=token)
        _print("AUTO EXECUTION", result)
        records = result.get("steps") if isinstance(result, dict) else None
        if not isinstance(records, list) or len(records) != 1:
            raise RuntimeError("Action 執行報告缺少唯一 Step；不得宣告完成")
        status = records[0].get("status")
        if status == "real_not_completed":
            raise RuntimeError(f"{action_name} 已確認尚未達成目標；停止，不自動重試")
        if status != "real_completed_verified":
            raise RuntimeError(f"{action_name} 物理完成未經驗證（{status}）；停止後續動作")
        return result

    report = run_stage(plan, preview_only=preflight_only)
    if preflight_only or report is None:
        return 0
    first = plan["steps"][0]
    if first["action_name"] != "observe_world":
        return 0
    if report["steps"][0]["status"] != "real_completed_verified":
        raise RuntimeError("觀察姿態尚未驗證到位，不允許重新拍攝／繼續動作")

    context = refresh_after_observe(report, provider=provider)
    _print("FRESH WORLD MODEL", context)

    # The original Plan distinguishes a terminal observation request from a
    # multi-stage task. Never replan a completed look-only request, and never
    # guess intent from hardcoded Chinese keywords in this CLI.
    followup = plan.get("followup_after_observe")
    if followup is False:
        _print("OBSERVATION COMPLETE", {
            "result": True,
            "verified_arrival": context["verified_arrival"],
            "arm_name": context["arm_name"],
            "direction": context["direction"],
            "camera_name": context["camera_name"],
            "scene_summary": context["scene"].get("scene_summary"),
            "scene_elements": context["scene"].get("scene_elements", []),
            "followup_executed": False,
        })
        return 0
    if followup is not True:
        raise RuntimeError("Plan 缺少有效的 followup_after_observe，拒絕自動續行")

    updated = prepare_task(text, provider=provider,
                           planning_options={"camera_name": context["camera_name"]},
                           observation_context=context)
    _print("REPLAN", {"plan_id": updated["plan_id"], "steps": updated["steps"]})
    if updated.get("complete") is True and not updated["steps"]:
        print("新場景顯示任務完成，停止。", flush=True)
        return 0
    # Bounded two-stage execution; a repeated observe_world is rejected.
    run_stage(updated, preview_only=False)
    return 0


def run_auto_request(*, user_text: str, provider: str = "local",
                     camera_name: str = "left", dry_run: bool = False,
                     preflight_only: bool = False) -> dict:
    """Invoke the SAME CLI Agent/Plan/Action path in the current process.

    The existing _run_text() keeps request-scoped authorization and
    one-shot dispatch. All action-specific motion checks live in Action Service.
    No second Flask server and no process-global stdout redirection.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text 必須是非空字串")
    if provider not in ("local", "openai"):
        raise ValueError("provider 僅支援 local/openai")
    if camera_name not in ("left", "right", "middle"):
        raise ValueError("camera_name 僅支援 left/right/middle")
    if type(dry_run) is not bool or type(preflight_only) is not bool:
        raise ValueError("dry_run/preflight_only 必須是 boolean")
    if dry_run and preflight_only:
        raise ValueError("dry_run 和 preflight_only 不可同時啟用")

    events = []
    sink_token = _AUTO_EVENT_SINK.set(events)
    try:
        _set_nonactivating_left_gripper()
        try:
            rc = _run_text(user_text.strip(), provider=provider, camera=camera_name,
                           dry_run=dry_run, preflight_only=preflight_only)
            return {"result": rc == 0, "exit_code": rc, "events": events}
        except Exception as exc:
            # Preserve all events preceding the failure. Never retry a
            # dispatched physical action; the current pose may be uncertain.
            return {"result": False, "exit_code": 1, "events": events,
                    "error_type": type(exc).__name__, "message": str(exc)}
    finally:
        _AUTO_EVENT_SINK.reset(sink_token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 混合能力入口：資訊查詢／全自動動作（不需手動確認）")
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--home", choices=("left", "right"), help="自動回到 config.default_joints")
    task.add_argument("--text", help="自然語言：Agent -> 資訊 Service 或 Plan -> Action")
    parser.add_argument("--provider", choices=("local", "openai"), default="local")
    parser.add_argument("--camera", choices=("left", "right", "middle"), default="left")
    parser.add_argument("--home-ignore-delta-gate", action="store_true",
                        help="僅本次 --home 略過 0.2 rad 位移門檻；沒有路徑碰撞驗證")
    parser.add_argument("--dry-run", action="store_true", help="動作 Mock／資訊查詢預覽，不呼叫實機 Service")
    parser.add_argument("--preflight-only", action="store_true", help="資訊只預覽工具；動作檢查狀態／路徑，不派送動作")
    args = parser.parse_args(argv)
    if args.home_ignore_delta_gate and not args.home:
        parser.error("--home-ignore-delta-gate 只能與 --home 一起使用")
    if args.dry_run and args.preflight_only:
        parser.error("--dry-run 和 --preflight-only 不可同時使用")
    if args.home:
        return _run_home(args.home, dry_run=args.dry_run,
                         preflight_only=args.preflight_only,
                         ignore_delta_gate=args.home_ignore_delta_gate)
    _set_nonactivating_left_gripper()
    return _run_text(args.text, provider=args.provider, camera=args.camera,
                     dry_run=args.dry_run, preflight_only=args.preflight_only)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("中斷：如果已派送動作，須透過機器人的安全停止機制確認停止。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        print("禁止自動重試；若已派送動作，應檢查機器人實際狀態。", file=sys.stderr)
        raise SystemExit(1)
