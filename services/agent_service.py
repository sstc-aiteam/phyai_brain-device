"""
對外 AI Agent 服務層。
"""

from agent.agent import build_prompt
from agent.executor import execute_tools
from services.arm_service import get_arm_status
from services.vision_service import get_latest_detections
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

        connected = data.get("connected")
        if not isinstance(connected, bool):
            return {
                "status_available": False,
                "connected": False,
                "status_error": ("手臂無法連線"),
            }

        return {
            "status_available": True,
            "connected": connected,
            "pose": data.get("pose"),
            "joints": data.get("joints"),
        }

    except Exception as exc:
        return {
            "status_available": False,
            "connected": False,
            "status_error": (f"取得機器手臂狀態時發生錯誤：{type(exc).__name__}"),
        }

def _get_detected_objects():
    """
    取得提供給 AI Agent 的最新物件辨識結果。
    """
    try:
        response = get_latest_detections(include_robot_xyz=True)

        if not isinstance(response, dict):
            return {
                "detection_available": False,
                "objects": [],
                "count": 0,
                "detection_error": "vision service 回傳格式錯誤",
            }

        status = response.get("status")
        result = response.get("result")
        objects = response.get("objects")

        if (status != "success" or result is not True or not isinstance(objects, list)):
            message = response.get("message")

            if not isinstance(message, str) or not message.strip():
                message = "無法取得最新物件辨識結果"

            return {
                "detection_available": False,
                "objects": [],
                "count": 0,
                "detection_error": message.strip(),
                "latest_time": response.get("latest_time"),
            }

        normalized_objects = []
        objects = assign_request_object_ids(objects)

        for obj in objects:
            if not isinstance(obj, dict):
                return {
                    "detection_available": False,
                    "objects": [],
                    "count": 0,
                    "detection_error": "vision service 的物件資料格式錯誤",
                    "latest_time": response.get("latest_time"),
                }

            normalized_objects.append({
                "object_id": obj.get("object_id"),
                "class_name": obj.get("class_name"),
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
            "latest_time": response.get("latest_time"),
        }

    except Exception as exc:
        return {
            "detection_available": False,
            "objects": [],
            "count": 0,
            "detection_error": (f"取得最新物件辨識結果時發生錯誤：{type(exc).__name__}"),
        }

def action_command(user_text):
    """
    解析自然語言指令，建立工具執行序列。
    """
    action = "action_command"

    try:
        robot_status = _get_robot_status()
        detected_objects = _get_detected_objects()

        result = build_prompt(
            user_text=user_text,
            robot_status=robot_status,
            detected_objects=detected_objects,
        )

        if not isinstance(result, dict):
            raise TypeError("Agent 回傳結果必須是 dict")

        required_fields = {
            "user_text",
            "answer",
            "tool_calls",
        }

        missing_fields = required_fields - result.keys()

        if missing_fields:
            raise ValueError(
                "欄位缺少：" + "、".join(sorted(missing_fields))
            )

        user_text_result = result["user_text"]
        answer = result["answer"]
        tool_calls = result["tool_calls"]

        if not isinstance(user_text_result, str):
            raise TypeError("user_text 必須是字串")

        if not isinstance(answer, str):
            raise TypeError("answer 必須是字串")

        if not isinstance(tool_calls, list):
            raise TypeError("tool_calls 必須是 list")
        
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
