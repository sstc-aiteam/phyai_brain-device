"""
對外 AI Agent 服務層。
"""

from agent.agent import build_prompt
from time import perf_counter
from agent.executor import execute_tools
from services.agent_planning_service import build_read_only_plan
from services.arm_service import get_arm_status
from services.vision_service import get_detections
from utils.object_identity import assign_request_object_ids
from utils.response import success, error


MODULE = "agent"


def _get_robot_status():
    """
    取得提供給 AI Agent 的精簡機器人狀態。
    """
    try:
        response = get_arm_status()

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

def _get_detected_objects():
    """
    取得提供給 AI Agent 的左右相機 structured detections。
    """
    try:
        raw_objects = []
        timestamps = {}
        camera_errors = {}
        for camera_name in ("left", "right"):
            response = get_detections(
                camera_name=camera_name,
                include_robot_xyz=True,
            )
            if not isinstance(response, dict):
                camera_errors[camera_name] = "vision service 回傳格式錯誤"
                continue
            data = response.get("data") or {}
            objects = data.get("detections")
            if response.get("status") != "success" or response.get("result") is not True \
                    or not isinstance(objects, list):
                camera_errors[camera_name] = str(
                    response.get("message") or "無法取得最新物件辨識結果"
                ).strip()
                continue
            timestamps[camera_name] = data.get("timestamp")
            raw_objects.extend({**obj, "camera_source": camera_name} for obj in objects)

        if not timestamps:
            return {
                "detection_available": False,
                "objects": [],
                "count": 0,
                "detection_error": "; ".join(
                    f"{camera}: {message}" for camera, message in camera_errors.items()
                ),
                "camera_errors": camera_errors,
            }

        normalized_objects = []
        objects = assign_request_object_ids(raw_objects)

        for obj in objects:
            if not isinstance(obj, dict):
                return {
                    "detection_available": False,
                    "objects": [],
                    "count": 0,
                    "detection_error": "vision service 的物件資料格式錯誤",
                    "latest_time": data.get("timestamp"),
                }

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

        return {
            "detection_available": True,
            "objects": normalized_objects,
            "count": len(normalized_objects),
            "latest_time": timestamps.get("left") or timestamps.get("right"),
            "camera_timestamps": timestamps,
            "camera_errors": camera_errors,
        }

    except Exception as exc:
        return {
            "detection_available": False,
            "objects": [],
            "count": 0,
            "detection_error": (f"取得最新物件辨識結果時發生錯誤：{type(exc).__name__}"),
        }

def _plan_result(user_text):
    robot_status = _get_robot_status()
    detected_objects = _get_detected_objects()
    result = build_prompt(
        user_text=user_text,
        robot_status=robot_status,
        detected_objects=detected_objects,
    )
    if not isinstance(result, dict):
        raise TypeError("Agent 回傳結果必須是 dict")
    required_fields = {"user_text", "answer", "tool_calls"}
    missing_fields = required_fields - result.keys()
    if missing_fields:
        raise ValueError("欄位缺少：" + "、".join(sorted(missing_fields)))
    if not isinstance(result["user_text"], str):
        raise TypeError("user_text 必須是字串")
    if not isinstance(result["answer"], str):
        raise TypeError("answer 必須是字串")
    if not isinstance(result["tool_calls"], list):
        raise TypeError("tool_calls 必須是 list")
    return result


def _normalize_planning_options(options):
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("planning_options 必須是 object")
    unknown = set(options) - {"stage1", "stage2", "openai_api_key"}
    if unknown:
        raise ValueError(f"planning_options 包含未知欄位：{sorted(unknown)}")
    normalized = {}
    for stage in ("stage1", "stage2"):
        value = options.get(stage) or {}
        if not isinstance(value, dict):
            raise ValueError(f"planning_options.{stage} 必須是 object")
        if set(value) - {"provider", "model"}:
            raise ValueError(f"planning_options.{stage} 包含未知欄位")
        provider = value.get("provider")
        model = value.get("model")
        if provider not in {None, "local", "openai"}:
            raise ValueError(f"planning_options.{stage}.provider 只支援 local/openai")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError(f"planning_options.{stage}.model 必須是非空字串")
        normalized[stage] = {
            "provider": provider,
            "model": model.strip() if isinstance(model, str) else None,
        }
    key = options.get("openai_api_key")
    if key is not None and (not isinstance(key, str) or not key.strip()):
        raise ValueError("planning_options.openai_api_key 必須是非空字串")
    if isinstance(key, str):
        normalized["openai_api_key"] = key.strip()
    return normalized


def plan_command(user_text, planning_options=None):
    """Build a resolved Abstract Plan without executing any tool."""
    action = "plan_command"
    started = perf_counter()
    try:
        options = _normalize_planning_options(planning_options)
        perception_started = perf_counter()
        detected_objects = _get_detected_objects()
        perception_ms = (perf_counter() - perception_started) * 1000
        result = build_read_only_plan(
            user_text,
            context={"detected_objects": detected_objects},
            llm_options=options,
        )
        timings = result.setdefault("timings_ms", {})
        timings["perception_context"] = round(perception_ms, 2)
        timings["request_total_backend"] = round(
            (perf_counter() - started) * 1000, 2
        )
        return success(
            MODULE,
            action,
            data=result,
        )
    except Exception as exc:
        return error(
            MODULE,
            action,
            error=exc,
            error_type=type(exc).__name__,
            timings_ms={
                "request_total_backend": round(
                    (perf_counter() - started) * 1000, 2
                )
            },
        )


def action_command(user_text):
    """
    解析自然語言指令，建立工具執行序列。
    """
    action = "action_command"

    try:
        result = _plan_result(user_text)
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

        return success(
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
        return error(
            MODULE,
            action,
            error=exc,
            error_type=type(exc).__name__,
        )
