from flask import (
    Blueprint,
    jsonify,
    request,
)

from services import gripper_service


gripper_bp = Blueprint(
    "gripper",
    __name__,
    url_prefix="/api/gripper",
)


# ============================================================
# Helpers
# ============================================================

def _get_json():
    return (
        request.get_json(
            silent=True
        )
        or {}
    )


# ============================================================
# STATUS
# ============================================================

@gripper_bp.route(
    "/get_gripper_status",
    methods=["GET"],
)
def get_gripper_status():
    return jsonify(
        gripper_service.get_gripper_status(
            gripper_name=
                request.args.get(
                    "gripper_name"
                ),
        )
    )


# ============================================================
# MOVE
# ============================================================

@gripper_bp.route(
    "/move_gripper",
    methods=["POST"],
)
def move_gripper():
    data = _get_json()

    return jsonify(
        gripper_service.move_gripper(
            gripper_name=
                data.get(
                    "gripper_name"
                ),
            position=
                data.get(
                    "position"
                ),
            speed=
                data.get(
                    "speed"
                ),
            force=
                data.get(
                    "force"
                ),
            wait=
                data.get(
                    "wait"
                ),
            timeout=
                data.get(
                    "timeout"
                ),
        )
    )


# ============================================================
# OPEN
# ============================================================

@gripper_bp.route(
    "/open_gripper",
    methods=["POST"],
)
def open_gripper():
    data = _get_json()

    return jsonify(
        gripper_service.open_gripper(
            gripper_name=
                data.get(
                    "gripper_name"
                ),
            speed=
                data.get(
                    "speed"
                ),
            force=
                data.get(
                    "force"
                ),
            wait=
                data.get(
                    "wait"
                ),
            timeout=
                data.get(
                    "timeout"
                ),
        )
    )


# ============================================================
# CLOSE
# ============================================================

@gripper_bp.route(
    "/close_gripper",
    methods=["POST"],
)
def close_gripper():
    data = _get_json()

    return jsonify(
        gripper_service.close_gripper(
            gripper_name=
                data.get(
                    "gripper_name"
                ),
            speed=
                data.get(
                    "speed"
                ),
            force=
                data.get(
                    "force"
                ),
            wait=
                data.get(
                    "wait"
                ),
            timeout=
                data.get(
                    "timeout"
                ),
        )
    )


# ============================================================
# STOP
# ============================================================

@gripper_bp.route(
    "/stop_gripper",
    methods=["POST"],
)
def stop_gripper():
    data = _get_json()

    return jsonify(
        gripper_service.stop_gripper(
            gripper_name=
                data.get(
                    "gripper_name"
                ),
        )
    )
