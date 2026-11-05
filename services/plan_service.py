from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import RLock, Thread
from typing import Any
from uuid import uuid4

from agent.runtime.adapters.runtime_observation import (
    build_runtime_initial_world,
)
from agent.runtime.decision.interpreter import (
    RUNTIME_ALLOWED_PROVIDERS as _RUNTIME_ALLOWED_PROVIDERS,
    normalize_runtime_planning_options,
)
from agent.runtime.interfaces.manual_orchestrator import (
    prepare_manual_runtime,
)
from agent.runtime.interfaces.manual_session import (
    ManualRuntimeCancelled,
    _ManualRuntimeSession,
    _manual_short_error,
    _manual_world_summary,
)
from utils import response


MODULE = "plan"
_MANUAL_RUNTIME_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "runtime_data"
    / "manual"
)

# ============================================================
# Manual UI Runtime
# ============================================================


_MANUAL_RUNTIME_LOCK = RLock()
_MANUAL_RUNTIME_SESSION: _ManualRuntimeSession | None = None


def _manual_runtime_worker(
    session: _ManualRuntimeSession,
) -> None:
    try:
        session.update(
            status=
                "running",
            stage=
                "bootstrap",
            message=
                "正在建立 initial multi-region WorldState。",
        )
        session.record(
            "bootstrap_started"
        )

        world = build_runtime_initial_world()

        session.update(
            latest_world=
                world.snapshot(),
            stage=
                "task_interpretation",
            message=
                "Initial WorldState 完成，正在解析 Runtime task。",
        )
        session.record(
            "bootstrap_completed",
            data={
                "world":
                    _manual_world_summary(
                        world
                    ),
            },
        )

        setup = prepare_manual_runtime(
            session=session,
            world=world,
            runtime_data_dir=
                _MANUAL_RUNTIME_DATA_DIR,
        )

        task_spec = setup.task_spec
        coordinator = setup.engine

        session.update(
            stage=
                "runtime",
            message=
                "Manual closed-loop Runtime 已開始。",
        )
        session.record(
            "runtime_started"
        )

        run_result = coordinator.run_until_done(
            world=
                world,
            goals=
                task_spec.goals,
            coordination=
                task_spec.coordination,
            max_iterations=
                session.max_iterations,
        )

        result = {
            "success":
                run_result.success,
            "iterations":
                run_result.iterations,
            "final_world":
                run_result.world.snapshot(),
            "trace": [
                {
                    "iteration":
                        row.iteration,
                    "dispatch":
                        deepcopy(
                            row.dispatch
                        ),
                    "bound_dispatch":
                        deepcopy(
                            row.bound_dispatch
                        ),
                    "execution":
                        deepcopy(
                            row.execution
                        ),
                    "progress_feedback":
                        deepcopy(
                            row.progress_feedback
                        ),
                }
                for row
                in run_result.trace
            ],
        }

        session.update(
            status=(
                "complete"
                if run_result.success
                else "stopped"
            ),
            stage=(
                "complete"
                if run_result.success
                else "max_iterations"
            ),
            message=(
                "Runtime task 完成。"
                if run_result.success
                else "已到 max_iterations，Runtime 尚未完成 task。"
            ),
            instruction=
                None,
            pending_action=
                None,
            latest_world=
                run_result.world.snapshot(),
            result=
                result,
        )
        session.record(
            "runtime_finished",
            data={
                "success":
                    run_result.success,
                "iterations":
                    run_result.iterations,
            },
        )

    except ManualRuntimeCancelled as exc:
        session.update(
            status=
                "cancelled",
            stage=
                "cancelled",
            message=
                str(
                    exc
                ),
            instruction=
                None,
            pending_action=
                None,
        )
        session.record(
            "runtime_cancelled",
            message=
                str(
                    exc
                ),
        )

    except Exception as exc:
        full_message = str(
            exc
        )
        summary = _manual_short_error(
            exc
        )

        session.update(
            status=
                "failed",
            stage=
                "failed",
            message=
                summary,
            instruction=
                None,
            pending_action=
                None,
            error={
                "type":
                    type(
                        exc
                    ).__name__,
                "summary":
                    summary,
                "message":
                    full_message,
            },
        )
        session.record(
            "runtime_failed",
            message=
                summary,
            data={
                "error_type":
                    type(
                        exc
                    ).__name__,
                "full_message":
                    full_message,
                "latest_world":
                    _manual_world_summary(
                        session.latest_world
                    ),
            },
        )


def start_manual_runtime(
    user_text: str,
    *,
    planning_options: dict[str, Any] | None = None,
    openai_api_key: str | None = None,
    max_iterations: int = 10,
    force_device_id: str | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(
            user_text,
            str,
        )
        or not user_text.strip()
    ):
        raise ValueError(
            "user_text 必須是非空字串"
        )

    if (
        not isinstance(
            max_iterations,
            int,
        )
        or max_iterations < 1
    ):
        raise ValueError(
            "max_iterations 必須是 >= 1 的整數"
        )

    normalized = normalize_runtime_planning_options(
        planning_options
    )

    if openai_api_key is not None:
        if (
            not isinstance(
                openai_api_key,
                str,
            )
            or not openai_api_key.strip()
        ):
            raise ValueError(
                "openai_api_key 必須是非空字串或 null"
            )
        openai_api_key = openai_api_key.strip()

    if force_device_id is not None:
        force_device_id = str(
            force_device_id
        ).strip() or None

    global _MANUAL_RUNTIME_SESSION

    with _MANUAL_RUNTIME_LOCK:
        current = _MANUAL_RUNTIME_SESSION

        if (
            current is not None
            and current.status
            in {
                "starting",
                "running",
                "awaiting_confirmation",
            }
        ):
            raise RuntimeError(
                "已有 manual Runtime session 正在執行"
            )

        session = _ManualRuntimeSession(
            session_id=
                uuid4().hex,
            user_text=
                user_text.strip(),
            planning_options=
                normalized,
            max_iterations=
                max_iterations,
            openai_api_key=
                openai_api_key,
            force_device_id=
                force_device_id,
        )
        session.record(
            "session_created",
            data={
                "planning_options":
                    normalized,
                "openai_api_key_supplied":
                    bool(openai_api_key),
                "max_iterations":
                    max_iterations,
                "force_device_id":
                    force_device_id,
            },
        )

        thread = Thread(
            target=
                _manual_runtime_worker,
            args=(
                session,
            ),
            daemon=
                True,
            name=(
                "manual-runtime-"
                + session.session_id[
                    :8
                ]
            ),
        )

        session.thread = thread
        _MANUAL_RUNTIME_SESSION = session
        thread.start()

    return session.snapshot()


def get_manual_runtime_status(
    session_id: str | None = None,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if session is None:
        raise RuntimeError(
            "目前沒有 manual Runtime session"
        )

    if (
        isinstance(
            session_id,
            str,
        )
        and session_id
        and session_id
        != session.session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    return session.snapshot()


def confirm_manual_runtime_step(
    session_id: str,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if (
        session is None
        or session.session_id
        != session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    with session.lock:
        if session.status != "awaiting_confirmation":
            raise RuntimeError(
                "manual Runtime 目前不在等待確認"
            )

        step_number = session.pending_step
        session.status = "running"
        session.stage = "manual_action_confirmed"
        session.message = "已收到人工確認，Runtime 繼續。"

    session.record(
        "confirm_api_received",
        data={
            "step":
                step_number,
        },
    )
    session.continue_event.set()

    return session.snapshot()


def cancel_manual_runtime(
    session_id: str,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if (
        session is None
        or session.session_id
        != session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    with session.lock:
        if session.status in {
            "complete",
            "stopped",
            "failed",
            "cancelled",
        }:
            return session.snapshot()

        session.cancel_requested = True
        session.message = "正在取消 manual Runtime。"

    session.record(
        "cancel_requested"
    )
    session.continue_event.set()

    return session.snapshot()


def get_health() -> dict[str, Any]:
    return response.success(
        MODULE,
        "health",
        data={
            "service":
                "plan",
            "runtime_task_interpreter":
                True,
            "providers":
                sorted(
                    _RUNTIME_ALLOWED_PROVIDERS
                ),
            "executes_robot":
                True,
            "manual_runtime":
                True,
            "physical_manipulation":
                False,
            "task_interpreter_skill":
                "agent/skills/runtime_task_interpreter.md",
        },
    )


# ============================================================
# Agent Plan (independent of Manual Runtime)
# ============================================================

def plan_task(
    user_text: str,
    *,
    provider: str = "local",
    planning_options: dict[str, Any] | None = None,
    observation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan ONE next Action using PM's agent.tools registry; never dispatch.

    After a verified observe_world motion, caller supplies a NEW WorldModel
    observation in observation_context. That fresh image, not the old view,
    grounds the next planning step. Manual Runtime is entirely independent.
    """
    import json
    import math
    from agent.runtime.decision.prompt_loader import load_next_action_prompt
    from agent.tools import get_tool, get_tools_for_prompt
    from agent.providers.structured import chat_structured
    from services import worldmodel_service

    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text 必須是非空字串")
    user_text = user_text.strip()
    if provider not in ("local", "openai"):
        raise ValueError("Plan provider 只支援 local/openai")
    if planning_options is None:
        planning_options = {}
    if not isinstance(planning_options, dict) or set(planning_options) - {"camera_name", "model"}:
        raise ValueError("planning_options 只支援 camera_name/model")
    camera_name = planning_options.get("camera_name", "left")
    if camera_name not in ("left", "right", "middle"):
        raise ValueError("camera_name 只支援 left/right/middle")
    model = planning_options.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model 必須是非空字串")

    catalog = get_tools_for_prompt()
    if not isinstance(catalog, list) or not catalog:
        raise RuntimeError("agent.tools 沒有可用 Action Registry")
    names = [x.get("api_function") for x in catalog if isinstance(x, dict)]
    if len(names) != len(set(names)) or not all(isinstance(x, str) and x for x in names):
        raise RuntimeError("agent.tools 定義無效")

    # Strict structured response, while variable parameters stay inside a
    # JSON-encoded string and are checked against PM's actual tool schema.
    response_schema = {
        "type": "object", "properties": {
            "supported": {"type": "boolean"},
            "complete": {"type": "boolean"},
            "needs_worldmodel": {"type": "boolean"},
            "followup_after_observe": {"type": "boolean"},
            "reason": {"type": "string"},
            "action_name": {"type": "string", "enum": ["none", *names]},
            "arguments_json": {"type": "string"},
        },
        "required": ["supported", "complete", "needs_worldmodel", "followup_after_observe", "reason", "action_name", "arguments_json"],
        "additionalProperties": False,
    }

    def ask(*, scene=None, already_observed=False):
        rules = load_next_action_prompt(already_observed=already_observed)
        if not already_observed:
            rules += (
                "\n\nThe requested camera is already specified by camera_name. "
                "observation_direction is only relevant when executing observe_world, "
                "which physically changes the robot's camera pose. "
                "Do not require observation_direction or a prior observe_world "
                "to choose open_drawer, close_drawer, or another registered "
                "manipulation action. If the current scene is needed, use "
                "needs_worldmodel=true so it can be captured from the existing "
                "camera without moving the robot. "
                "Do not mark an otherwise supported manipulation request as "
                "unsupported solely because no observation_direction was given. "
                "Do not invent a target or claim a goal is complete without evidence."
            )
        if scene is not None and not already_observed:
            rules += (
                "\n\nA FRESH WorldModel scene has ALREADY been captured from the "
                "current camera pose without moving the robot. "
                "For a user-requested manipulation action, do not select observe_world "
                "just to get an initial scene or as a routine prerequisite. "
                "When the user explicitly requests a registered manipulation "
                "action, preserve that action intent even if the scene description "
                "labels the drawer as cabinet doors or omits the drawer. "
                "These labels are fallible and are not proof of target absence. "
                "Use the user-specified arm side for matching gripper and camera "
                "parameters when the tool requires same-side devices. "
                "Select observe_world only when a camera pose change was explicitly "
                "requested or is necessary for a concrete, independently justified "
                "visibility reason; never as an automatic prerequisite. "
                "Do not invent positions, object IDs, or physical completion. "
                "If the scene proves that the requested goal is already met, "
                "return complete=true and action_name=none. "
                "Never assume a task succeeded merely from its requested intent."
            )
        payload = {"user_text": user_text, "tools": catalog,
                   "camera_name": camera_name,
                   "observation_completed": already_observed}
        if already_observed:
            payload["observation_direction"] = observation_context["direction"]
        if scene is not None:
            payload["worldmodel"] = {"camera_name": camera_name, "scene": scene}
        message = chat_structured(
            stage="agent_plan_task", provider=provider, model=model,
            messages=[{"role": "system", "content": rules},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            response_schema=response_schema, temperature=0.0,
            max_tokens=1024, num_ctx=8192, timeout=120.0,
            owner="agent_plan_task",
        )
        if not isinstance(message, dict):
            raise RuntimeError("Plan LLM 回傳格式錯誤")
        content = message.get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Plan LLM 不是有效 JSON") from exc
        if not isinstance(parsed, dict) or set(parsed) != set(response_schema["required"]):
            raise RuntimeError("Plan LLM 結構化欄位錯誤")
        if any(type(parsed[k]) is not bool for k in ("supported", "complete", "needs_worldmodel", "followup_after_observe")):
            raise RuntimeError("Plan 布林欄位格式錯誤")
        if not isinstance(parsed["reason"], str) or parsed["action_name"] not in ["none", *names]:
            raise RuntimeError("Plan LLM 動作或原因格式錯誤")
        if not isinstance(parsed["arguments_json"], str):
            raise RuntimeError("arguments_json 必須是字串")
        return parsed

    def get_fresh_scene():
        response_value = worldmodel_service.describe_scene(camera_name=camera_name, vlm_backend=provider)
        if not isinstance(response_value, dict) or response_value.get("result") is not True:
            raise RuntimeError("WorldModel 觀察失敗，不能繼續視覺動作：" + str(response_value)[:250])
        data = response_value.get("data")
        scene_value = data.get("scene") if isinstance(data, dict) else None
        if not isinstance(scene_value, dict):
            raise RuntimeError("WorldModel 沒有有效的 scene")
        return scene_value

    observed = observation_context is not None
    if observed:
        if (not isinstance(observation_context, dict)
                or observation_context.get("arm_name") not in ("left", "right")
                or observation_context.get("direction") not in ("front", "left", "back", "right", "up", "down")
                or observation_context.get("camera_name") != camera_name
                or not isinstance(observation_context.get("scene"), dict)
                or observation_context.get("verified_arrival") is not True):
            raise ValueError("observation_context 必須是已確認到位後的新鮮觀察")
        scene = observation_context["scene"]
        worldmodel_data = {"used": True, "camera_name": camera_name,
                           "scene": scene, "source": "post_observe_world"}
        decision = ask(scene=scene, already_observed=True)
    else:
        decision = ask()
        scene = None
        worldmodel_data = {"used": False, "camera_name": None, "scene": None,
                           "reason": "當前動作規劃不需先取得畫面"}
        visual_actions = {"open_drawer", "close_drawer", "take_object", "follow_object"}
        passive_observation_required = (
            decision["needs_worldmodel"]
            or decision["action_name"] in visual_actions
            or (decision["action_name"] == "observe_world"
                and decision["followup_after_observe"])
            or (not decision["supported"]
                and decision["action_name"] == "none"
                and not decision["complete"])
        )
        if passive_observation_required:
            scene = get_fresh_scene()
            worldmodel_data = {"used": True, "camera_name": camera_name, "scene": scene,
                               "source": "initial_observation"}
            decision = ask(scene=scene)

    action_name = decision["action_name"]
    if decision["complete"]:
        # A complete state is returned as no-op, not a fabricated Action.
        if not worldmodel_data["used"]:
            raise RuntimeError("缺少當次觀察證據，不能宣稱視覺任務已完成")
        if action_name != "none":
            raise RuntimeError("已完成任務不應包含 Action")
        return response.success(MODULE, "plan_task", data={
            "plan_id": uuid4().hex, "user_text": user_text, "provider": provider,
            "mode": "mock", "physical_execution_allowed": False,
            "planner": "tools_registry_llm_v3", "steps": [],
            "reason": decision["reason"], "complete": True,
            "followup_after_observe": False,
            "worldmodel": worldmodel_data,
        })
    if not decision["supported"] or action_name == "none":
        raise ValueError("目前無法安全規劃此任務：" + decision["reason"])
    if observed and action_name == "observe_world":
        raise ValueError("觀察後仍需重新觀察：停止，避免無界限重試")
    if action_name != "observe_world" and decision["needs_worldmodel"] and not worldmodel_data["used"]:
        raise RuntimeError("動作需要 WorldModel，但沒有新鮮觀察")
    if action_name in ("open_drawer", "close_drawer", "take_object", "follow_object") and not worldmodel_data["used"]:
        raise RuntimeError("視覺動作缺少 WorldModel，拒絕規劃")
    try:
        arguments = json.loads(decision["arguments_json"])
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Plan arguments_json 格式錯誤") from exc
    tool = get_tool(action_name)
    if not isinstance(tool, dict) or not isinstance(arguments, dict):
        raise RuntimeError("選擇的 Action 不存在或參數不是 object")
    specs = tool.get("parameters")
    required = tool.get("required")
    if not isinstance(specs, dict) or not isinstance(required, list):
        raise RuntimeError("PM Action Registry 規格不完整")
    if set(arguments) - set(specs) or set(required) - set(arguments):
        raise RuntimeError("Plan Action 參數不符合 PM 白名單")
    for key, value in arguments.items():
        spec = specs[key]
        kind = spec.get("type")
        if kind == "string" and (not isinstance(value, str) or not value.strip()):
            raise RuntimeError(f"Action 參數 {key} 必須是字串")
        if kind == "number" and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)):
            raise RuntimeError(f"Action 參數 {key} 必須是有限數字")
        if kind == "number" and "minimum" in spec and value < spec["minimum"]:
            raise RuntimeError(f"Action 參數 {key} 小於最小值")
        if "enum" in spec and value not in spec["enum"]:
            raise RuntimeError(f"Action 參數 {key} 不在允許範圍")
    if arguments.get("arm_name") not in ("left", "right"):
        raise RuntimeError("僅支援已明確指定的左右手臂")
    if action_name in ("open_drawer", "close_drawer"):
        if arguments.get("camera_name") != arguments["arm_name"] or arguments.get("gripper_name") != arguments["arm_name"]:
            raise RuntimeError("抽屜操作必須同側相機與夾爪")
        if worldmodel_data["camera_name"] != arguments["camera_name"]:
            raise RuntimeError("觀察相機與動作相機不一致")
    if action_name == "observe_world" and observed:
        raise RuntimeError("同一回合禁止反覆移動觀察")

    return response.success(MODULE, "plan_task", data={
        "plan_id": uuid4().hex, "user_text": user_text, "provider": provider,
        "mode": "mock", "physical_execution_allowed": False,
        "planner": "tools_registry_llm_v3",
        "steps": [{"step_id": "S1", "action_name": action_name,
                   "arguments": arguments, "status": "planned"}],
        "reason": decision["reason"], "complete": False,
        "followup_after_observe": (decision["followup_after_observe"]
                                   if action_name == "observe_world" else False),
        "worldmodel": worldmodel_data,
    })
