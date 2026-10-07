from flask import (
    Blueprint,
    jsonify,
    request,
)

from services import arm_service


arm_bp = Blueprint(
    "arm",
    __name__,
    url_prefix="/api/arm",
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
# GET STATUS
# ============================================================

@arm_bp.route(
    "/get_arm_status",
    methods=["GET"],
)
def get_arm_status():
    arm_name = (
        request.args.get(
            "arm_name"
        )
    )

    return jsonify(
        arm_service.get_arm_status(
            arm_name=arm_name
        )
    )


# ============================================================
# GET POSE
# ============================================================

@arm_bp.route(
    "/get_arm_pose",
    methods=["GET"],
)
def get_arm_pose():
    arm_name = (
        request.args.get(
            "arm_name"
        )
    )

    return jsonify(
        arm_service.get_arm_pose(
            arm_name=arm_name
        )
    )


# ============================================================
# GET JOINTS
# ============================================================

@arm_bp.route(
    "/get_arm_joints",
    methods=["GET"],
)
def get_arm_joints():
    arm_name = (
        request.args.get(
            "arm_name"
        )
    )

    return jsonify(
        arm_service.get_arm_joints(
            arm_name=arm_name
        )
    )


# ============================================================
# RECONNECT
# ============================================================

@arm_bp.route(
    "/reconnect_arm",
    methods=["POST"],
)
def reconnect_arm():
    data = _get_json()

    return jsonify(
        arm_service.reconnect_arm(
            arm_name=
                data.get(
                    "arm_name"
                ),
        )
    )


# ============================================================
# MOVE DEFAULT
# ============================================================

@arm_bp.route(
    "/move_arm_default",
    methods=["POST"],
)
def move_arm_default():
    data = _get_json()

    return jsonify(
        arm_service.move_arm_default(
            arm_name=
                data.get(
                    "arm_name"
                ),
            speed=
                data.get(
                    "speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            wait=
                data.get(
                    "wait",
                    True,
                ),
        )
    )


# ============================================================
# MOVE POSE
# ============================================================

@arm_bp.route(
    "/move_arm_pose",
    methods=["POST"],
)
def move_arm_pose():
    data = _get_json()

    return jsonify(
        arm_service.move_arm_pose(
            arm_name=
                data.get(
                    "arm_name"
                ),
            pose=
                data.get(
                    "pose"
                ),
            speed=
                data.get(
                    "speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            wait=
                data.get(
                    "wait",
                    True,
                ),
        )
    )


# ============================================================
# MOVE XYZ
# ============================================================

@arm_bp.route(
    "/move_arm_xyz",
    methods=["POST"],
)
def move_arm_xyz():
    data = _get_json()

    return jsonify(
        arm_service.move_arm_xyz(
            arm_name=
                data.get(
                    "arm_name"
                ),
            x=
                data.get(
                    "x"
                ),
            y=
                data.get(
                    "y"
                ),
            z=
                data.get(
                    "z"
                ),
            speed=
                data.get(
                    "speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            wait=
                data.get(
                    "wait",
                    True,
                ),
        )
    )


# ============================================================
# MOVE JOINTS
# ============================================================

@arm_bp.route(
    "/move_arm_joints",
    methods=["POST"],
)
def move_arm_joints():
    data = _get_json()

    return jsonify(
        arm_service.move_arm_joints(
            arm_name=
                data.get(
                    "arm_name"
                ),
            joints=
                data.get(
                    "joints"
                ),
            speed=
                data.get(
                    "speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            wait=
                data.get(
                    "wait",
                    True,
                ),
        )
    )


# ============================================================
# MOVE JOINT TRAJECTORY
# ============================================================
@arm_bp.route(
    "/move_arm_pose_trajectory",
    methods=["POST"],
)
def move_arm_pose_trajectory():
    data = _get_json()

    return jsonify(
        arm_service.move_arm_pose_trajectory(
            arm_name=
                data.get(
                    "arm_name"
                ),
            pose_trajectory=
                data.get(
                    "pose_trajectory"
                ),
            dt=
                data.get(
                    "dt"
                ),
            speed=
                data.get(
                    "speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            wait=
                data.get(
                    "wait",
                    True,
                ),
            move_to_start=
                data.get(
                    "move_to_start",
                    True,
                ),
            move_to_start_speed=
                data.get(
                    "move_to_start_speed"
                ),
            move_to_start_acceleration=
                data.get(
                    "move_to_start_acceleration"
                ),
        )
    )
    
@arm_bp.route(
    "/move_arm_joint_trajectory",
    methods=["POST"],
)
def move_arm_joint_trajectory():
    data = _get_json()

    return jsonify(
        arm_service.move_arm_joint_trajectory(
            arm_name=
                data.get(
                    "arm_name"
                ),
            joint_trajectory=
                data.get(
                    "joint_trajectory"
                ),
            dt=
                data.get(
                    "dt"
                ),
            speed=
                data.get(
                    "speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            wait=
                data.get(
                    "wait",
                    True,
                ),
            move_to_start=
                data.get(
                    "move_to_start",
                    True,
                ),
            move_to_start_speed=
                data.get(
                    "move_to_start_speed"
                ),
            move_to_start_acceleration=
                data.get(
                    "move_to_start_acceleration"
                ),
        )
    )


# ============================================================
# MOVE STEP
# ============================================================

@arm_bp.route(
    "/move_arm_step",
    methods=["POST"],
)
def move_arm_step():
    data = _get_json()

    return jsonify(
        arm_service.move_arm_step(
            arm_name=
                data.get(
                    "arm_name"
                ),
            direction=
                data.get(
                    "direction"
                ),
            distance=
                data.get(
                    "distance"
                ),
            speed=
                data.get(
                    "speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            wait=
                data.get(
                    "wait",
                    True,
                ),
        )
    )


# ============================================================
# MOVE ROTATE
# ============================================================

@arm_bp.route(
    "/move_arm_rotate",
    methods=["POST"],
)
def move_arm_rotate():
    data = _get_json()

    return jsonify(
        arm_service.move_arm_rotate(
            arm_name=
                data.get(
                    "arm_name"
                ),
            direction=
                data.get(
                    "direction"
                ),
            angle=
                data.get(
                    "angle"
                ),
            speed=
                data.get(
                    "speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            wait=
                data.get(
                    "wait",
                    True,
                ),
        )
    )


# ============================================================
# START JOG
# ============================================================
@arm_bp.route(
    "/start_arm_jog",
    methods=["POST"],
)
def start_arm_jog():
    data = _get_json()

    return jsonify(
        arm_service.start_arm_jog(
            arm_name=
                data.get(
                    "arm_name"
                ),
            direction=
                data.get(
                    "direction"
                ),
            linear_speed=
                data.get(
                    "linear_speed"
                ),
            angular_speed=
                data.get(
                    "angular_speed"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
            timeout=
                data.get(
                    "timeout"
                ),
        )
    )


# ============================================================
# STOP JOG
# ============================================================

@arm_bp.route(
    "/stop_arm_jog",
    methods=["POST"],
)
def stop_arm_jog():
    data = _get_json()

    return jsonify(
        arm_service.stop_arm_jog(
            arm_name=
                data.get(
                    "arm_name"
                ),
        )
    )

# ============================================================
# START FREEDRIVE
# ============================================================

@arm_bp.route(
    "/start_arm_freedrive",
    methods=["POST"],
)
def start_arm_freedrive():
    data = _get_json()

    return jsonify(
        arm_service.start_arm_freedrive(
            arm_name=
                data.get(
                    "arm_name"
                ),
        )
    )

@arm_bp.route(
    "/stop_arm_freedrive",
    methods=["POST"],
)
def stop_arm_freedrive():
    data = _get_json()

    return jsonify(
        arm_service.stop_arm_freedrive(
            arm_name=
                data.get(
                    "arm_name"
                ),
        )
    )

# ============================================================
# STOP ARM
# ============================================================

@arm_bp.route(
    "/stop_arm",
    methods=["POST"],
)
def stop_arm():
    data = _get_json()

    return jsonify(
        arm_service.stop_arm(
            arm_name=
                data.get(
                    "arm_name"
                ),
            acceleration=
                data.get(
                    "acceleration"
                ),
        )
    )