"""
對外 AI Agent 服務層。
"""
import time
from agent.agent import (build_agent_command_data, route_command)
from time import perf_counter
# from agent.executor import execute_tools
from services import arm_service, perception_service, camera_service, vlm_narrator_service, plan_service
from utils import response


MODULE = "agent"


def _get_robot_status():
    """
    取得提供給 AI Agent 的精簡機器人狀態。
    """
    try:
        response = arm_service.get_arm_status()

        if not isinstance(response, dict):
            return {
                "status_available": False,
                "connected": False,
                "status_error": "arm service 回傳格式錯誤",
            }

        result = response.get("result")
        data = response.get("data")
        if result is not True or not isinstance(data, dict):
            message = response.get("message")

            if not isinstance(message, str) or not message.strip():
                message = "無法取得機器手臂狀態"

            return {
                "status_available": False,
                "connected": False,
                "status_error": message.strip(),
            }

        arms = data.get("arms")
        if not isinstance(arms, list):
            return {
                "status_available": False,
                "connected": False,
                "status_error": "arm service 缺少 arms 狀態",
            }

        right = next(
            (item for item in arms if item.get("arm_name") == "right"),
            None,
        )
        right_status = (right or {}).get("status") or {}
        connected = right_status.get("connected") is True

        return {
            "status_available": True,
            "connected": connected,
            "arm_name": "right",
            "driver": (right or {}).get("driver"),
            "pose": right_status.get("pose"),
            "joints": right_status.get("joints"),
        }

    except Exception as exc:
        return {
            "status_available": False,
            "connected": False,
            "status_error": (f"取得機器手臂狀態時發生錯誤：{type(exc).__name__}"),
        }
def _get_perception_context(
    camera_names=("left", "right"),
    model_name="object_detector",
    include_robot_xyz=True,
):
    """
    Agent 共用 perception context。

    現階段透過 perception_service.get_detections() 聚合多相機結果。
    未來 perception_service 提供 multi-camera API 時，只需修改此函式，
    其他 Agent / Plan 邏輯不用變。
    """

    raw_objects = []
    timestamps = {}
    camera_errors = {}

    for camera_name in camera_names:
        try:
            response = perception_service.get_detections(
                camera_name=camera_name,
                model_name=model_name,
                include_robot_xyz=include_robot_xyz,
            )

            # 第一次失敗時，嘗試啟動相機後再偵測一次
            if (
                not isinstance(response, dict)
                or response.get("status") != "success"
                or response.get("result") is not True
            ):
                start_response = camera_service.start_camera(
                    camera_name=camera_name,
                )

                if (
                    isinstance(start_response, dict)
                    and start_response.get("status") == "success"
                    and start_response.get("result") is True
                ):
                    time.sleep(0.3)

                    response = perception_service.get_detections(
                        camera_name=camera_name,
                        model_name=model_name,
                        include_robot_xyz=include_robot_xyz,
                    )

            if not isinstance(response, dict):
                camera_errors[camera_name] = (
                    "vision service 回傳格式錯誤"
                )
                continue

            data = response.get("data") or {}
            detections = data.get("detections")

            if (
                response.get("status") != "success"
                or response.get("result") is not True
                or not isinstance(detections, list)
            ):
                camera_errors[camera_name] = str(
                    response.get("message")
                    or "無法取得物件辨識結果"
                )
                continue

            timestamps[camera_name] = data.get("timestamp")

            for detection in detections:
                if not isinstance(detection, dict):
                    continue

                raw_objects.append({
                    **detection,
                    "camera_source": camera_name,
                })

        except Exception as exc:
            camera_errors[camera_name] = (
                f"{type(exc).__name__}: {exc}"
            )

    if not timestamps:
        return {
            "detection_available": False,
            "objects": [],
            "count": 0,
            "latest_time": None,
            "detection_error": "; ".join(
                f"{camera}: {message}"
                for camera, message in camera_errors.items()
            ),
            "camera_timestamps": {},
            "camera_errors": camera_errors,
        }

    # Reuse the centralized multi-camera identity assignment.
    objects = vlm_narrator_service.assign_object_ids(raw_objects)

    normalized_objects = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        normalized_objects.append({
            "object_id": obj.get("object_id"),
            "class_name": obj.get("class_name"),
            "camera_source": obj.get("camera_source"),
            "confidence": obj.get("confidence"),
            "robot_xyz": obj.get("robot_xyz"),
            "yaw_deg": obj.get("yaw_deg"),
            "condition": obj.get("condition"),
            "visual_description": obj.get("visual_description"),
            "state_tags": obj.get("state_tags"),
        })

    latest_time = next(
        (
            timestamps.get(camera_name)
            for camera_name in reversed(camera_names)
            if timestamps.get(camera_name) is not None
        ),
        None,
    )

    return {
        "detection_available": True,
        "objects": normalized_objects,
        "count": len(normalized_objects),
        "latest_time": latest_time,
        "camera_timestamps": timestamps,
        "camera_errors": camera_errors,
    }



#0待刪除
def action_command(user_text):
    """
    解析自然語言指令，建立工具執行序列。
    """
    action = "action_command"

    try:
        # result = _plan_result(user_text)
        user_text_result = result["user_text"]
        answer = result["answer"]
        tool_calls = result["tool_calls"]
        
        execution = None

        if tool_calls:
            # build_prompt 已完成白名單、必要參數、座標範圍與任務前置
            # 條件驗證；executor 會再驗證一次工具名稱，並在單一背景執行緒
            # 中依序呼叫 arm_service / task_service。若已有任務執行中，
            # execute_tools 會直接拒絕，避免兩組實體動作同時進行。
            execution = execute_tools(tool_calls)

        return response.success(
            MODULE,
            action,
            data={
                "user_text": user_text_result,
                "answer": answer,
                "tool_calls": tool_calls,
                "execution": execution,
            },
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            error_type=type(exc).__name__,
        )

def _command_perceive(user_text,camera_names, started,):
    perception = _get_perception_context(
        camera_names=camera_names,
    )

    if perception["detection_available"] is not True:
        raise RuntimeError(
            "無法取得物件辨識結果"
        )

    objects = perception["objects"]

    return build_agent_command_data(
        user_text=user_text,
        used_modules=["perceive"],
        answer=f"共辨識到 {len(objects)} 個物件。",
        objects=objects,
        timings_ms={
            "total": round(
                (perf_counter() - started) * 1000,
                2,
            ),
        },
    )

def _command_plan(user_text,planning_options, started,):

    perception_started = perf_counter()

    perception = _get_perception_context()

    perception_ms = (
        perf_counter() - perception_started
    ) * 1000

    result = (
        plan_service
        .build_one_shot_value(
            user_text,
            context={
                "detected_objects":
                    perception,
            },
            planning_options=
                planning_options,
        )
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            "plan module 回傳格式錯誤"
        )

    objects = perception.get("objects")
    if not isinstance(objects, list):
        objects = []

    steps = result.get("steps")
    if not isinstance(steps, list):
        steps = []

    timings = dict(
        result.get("timings_ms") or {}
    )

    timings["perception_context"] = round(
        perception_ms,
        2,
    )

    timings["total"] = round(
        (perf_counter() - started) * 1000,
        2,
    )

    return build_agent_command_data(
        user_text=user_text,
        used_modules=[
            "perceive",
            "plan",
        ],
        answer=result.get("answer") or "",
        objects=objects,
        steps=steps,
        trace_id=result.get("trace_id"),
        timings_ms=timings,
    )

def command(user_text, planning_options=None):
    """Agent 統一自然語言入口。"""

    action = "command"
    started = perf_counter()
    used_modules = []

    try:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text 必須是非空字串")

        user_text = user_text.strip()

        # Agent 只負責理解需求並選擇下一層 module
        route = route_command(user_text)

        module = route["module"]
        camera_names = tuple(
            route.get("camera_names") or []
        )

        # ----------------------------------------------------
        # perceive
        # ----------------------------------------------------
        if module == "perceive":
            used_modules = ["perceive"]

            if not camera_names:
                camera_names = ("left", "right")

            data = _command_perceive(
                user_text,
                camera_names,
                started,
            )

        # ----------------------------------------------------
        # plan
        # ----------------------------------------------------
        elif module == "plan":
            used_modules = [
                "perceive",
                "plan",
            ]

            data = _command_plan(
                user_text,
                planning_options,
                started,
            )

        # ----------------------------------------------------
        # world-model
        # ----------------------------------------------------
        elif module == "world-model":
            raise NotImplementedError(
                "world-model 尚未接入 Agent command"
            )

        # ----------------------------------------------------
        # action
        # ----------------------------------------------------
        elif module == "action":
            raise NotImplementedError(
                "action 尚未接入 Agent command"
            )

        # ----------------------------------------------------
        # calibrate
        # ----------------------------------------------------
        elif module == "calibrate":
            raise NotImplementedError(
                "calibrate 尚未接入 Agent command"
            )

        # ----------------------------------------------------
        # train
        # ----------------------------------------------------
        elif module == "train":
            raise NotImplementedError(
                "train 尚未接入 Agent command"
            )

        else:
            raise RuntimeError(
                f"不支援的 Agent module：{module}"
            )

        return response.success(
            MODULE,
            action,
            data=data,
        )

    except Exception as exc:
        data = build_agent_command_data(
            user_text=(
                user_text
                if isinstance(user_text, str)
                else ""
            ),
            used_modules=used_modules,
            answer=str(exc),
            timings_ms={
                "total": round(
                    (perf_counter() - started) * 1000,
                    2,
                ),
            },
        )

        return response.error(
            MODULE,
            action,
            error=exc,
            error_type=type(exc).__name__,
            result=False,
            data=data,
        )