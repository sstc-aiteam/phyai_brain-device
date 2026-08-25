from flask import Blueprint, request

from services import gripper_service
from routes.utils import (
    get_json_body,
    json_response,
    request_error,
)


MODULE = "gripper"

gripper_bp = Blueprint(
    "gripper",
    __name__,
    url_prefix="/api/gripper",
)


# ============================================================
# GET STATUS
# ============================================================

@gripper_bp.get("/get_gripper_status")
def get_gripper_status():
    """
    取得夾爪狀態。

    Query:
        gripper_name:
            指定夾爪名稱，例如 left / right。
            若未指定，則取得所有夾爪狀態。

    Example:
        GET /api/gripper/get_gripper_status

        GET /api/gripper/get_gripper_status?gripper_name=left
    """

    action = "get_gripper_status"

    try:
        gripper_name = request.args.get(
            "gripper_name"
        )

        response = (
            gripper_service.get_gripper_status(
                gripper_name=gripper_name
            )
        )

        return json_response(
            response
        )

    except ValueError as exc:
        return request_error(
            MODULE,
            action,
            exc,
        )


# ============================================================
# MOVE TO POSITION
# ============================================================

@gripper_bp.post("/action_gripper_to_position")
def action_gripper_to_position():
    """
    將夾爪移動到指定位置。

    JSON:
    {
        "gripper_name": "left",
        "position": 128,
        "speed": 255,
        "force": 150,
        "wait": true,
        "timeout": 5.0
    }
    """

    action = "action_gripper_to_position"

    try:
        data = get_json_body()

        if "gripper_name" not in data:
            raise ValueError(
                "缺少必要參數：gripper_name"
            )

        if "position" not in data:
            raise ValueError(
                "缺少必要參數：position"
            )

        response = (
            gripper_service.move_gripper(
                gripper_name=
                    data["gripper_name"],
                position=
                    data["position"],
                speed=
                    data.get("speed"),
                force=
                    data.get("force"),
                wait=
                    data.get("wait"),
                timeout=
                    data.get("timeout"),
            )
        )

        return json_response(
            response
        )

    except ValueError as exc:
        return request_error(
            MODULE,
            action,
            exc,
        )


# ============================================================
# CLOSE
# ============================================================

@gripper_bp.post("/action_gripper_close")
def action_gripper_close():
    """
    關閉夾爪。

    JSON:
    {
        "gripper_name": "left",
        "speed": 255,
        "force": 150,
        "wait": true,
        "timeout": 5.0
    }
    """

    action = "action_gripper_close"

    try:
        data = get_json_body()

        if "gripper_name" not in data:
            raise ValueError(
                "缺少必要參數：gripper_name"
            )

        response = (
            gripper_service.close_gripper(
                gripper_name=
                    data["gripper_name"],
                speed=
                    data.get("speed"),
                force=
                    data.get("force"),
                wait=
                    data.get("wait"),
                timeout=
                    data.get("timeout"),
            )
        )

        return json_response(
            response
        )

    except ValueError as exc:
        return request_error(
            MODULE,
            action,
            exc,
        )


# ============================================================
# OPEN
# ============================================================

@gripper_bp.post("/action_gripper_open")
def action_gripper_open():
    """
    開啟夾爪。

    JSON:
    {
        "gripper_name": "left",
        "speed": 255,
        "force": 150,
        "wait": true,
        "timeout": 5.0
    }
    """

    action = "action_gripper_open"

    try:
        data = get_json_body()

        if "gripper_name" not in data:
            raise ValueError(
                "缺少必要參數：gripper_name"
            )

        response = (
            gripper_service.open_gripper(
                gripper_name=
                    data["gripper_name"],
                speed=
                    data.get("speed"),
                force=
                    data.get("force"),
                wait=
                    data.get("wait"),
                timeout=
                    data.get("timeout"),
            )
        )

        return json_response(
            response
        )

    except ValueError as exc:
        return request_error(
            MODULE,
            action,
            exc,
        )


# ============================================================
# STOP
# ============================================================

@gripper_bp.post("/stop")
def stop_gripper():
    """
    停止指定夾爪目前的動作。

    JSON:
    {
        "gripper_name": "left"
    }
    """

    action = "stop_gripper"

    try:
        data = get_json_body()

        if "gripper_name" not in data:
            raise ValueError(
                "缺少必要參數：gripper_name"
            )

        response = (
            gripper_service.stop_gripper(
                gripper_name=
                    data["gripper_name"]
            )
        )

        return json_response(
            response
        )

    except ValueError as exc:
        return request_error(
            MODULE,
            action,
            exc,
        )