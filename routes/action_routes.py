from flask import Blueprint, request, jsonify

from services import action_service


action_bp = Blueprint(
    "action",
    __name__,
    url_prefix="/api/action",
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
# OPEN DRAWER
# ============================================================

@action_bp.route(
    "/open_drawer",
    methods=["POST"],
)
def open_drawer():
    data = _get_json()

    result = action_service.open_drawer(
        arm_name=data.get("arm_name"),
        gripper_name=data.get("gripper_name"),
        camera_name=data.get("camera_name"),
        dt=data.get("dt", 0.1),
        speed=data.get("speed", 0.1),
        acceleration=data.get(
            "acceleration",
            0.1,
        ),
        max_batches=data.get(
            "max_batches",
            100,
        ),
    )

    return jsonify(result)


# ============================================================
# TAKE OBJECT
# ============================================================

@action_bp.route(
    "/take_object",
    methods=["POST"],
)
def take_object():
    data = _get_json()

    result = action_service.take_object(
        arm_name=data.get("arm_name"),
        gripper_name=data.get("gripper_name"),
        x=data.get("x"),
        y=data.get("y"),
        z=data.get("z"),
        speed=data.get("speed"),
        acceleration=data.get(
            "acceleration"
        ),
    )

    return jsonify(result)


# ============================================================
# MOVE ARM DEFAULT
# ============================================================

@action_bp.route(
    "/move_arm_default",
    methods=["POST"],
)
def move_arm_default():
    data = _get_json()

    result = action_service.move_arm_default(
        arm_name=data.get("arm_name"),
        speed=data.get("speed"),
        acceleration=data.get(
            "acceleration"
        ),
    )

    return jsonify(result)