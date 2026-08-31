from flask import (
    Blueprint,
    jsonify,
    request,
)

from services import recording as record_service


record_bp = Blueprint(
    "record",
    __name__,
    url_prefix="/api/record",
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
# START RECORDING
# ============================================================

@record_bp.route(
    "/start_robot_recording",
    methods=["POST"],
)
def start_robot_recording():
    data = _get_json()

    return jsonify(
        record_service.start_robot_recording(
            arm_name=
                data.get(
                    "arm_name"
                ),

            gripper_name=
                data.get(
                    "gripper_name"
                ),

            output_path=
                data.get(
                    "output_path"
                ),

            interval=
                data.get(
                    "interval",
                    0.1,
                ),

            freedrive=
                data.get(
                    "freedrive",
                    False,
                ),

            record_video=
                data.get(
                    "record_video",
                    False,
                ),

            task=
                data.get(
                    "task",
                    "robot demonstration",
                ),

            initial_gripper_position=
                data.get(
                    "initial_gripper_position",
                    0,
                ),

            dataset_format=data.get("dataset_format", "lerobot_v3"),
        )
    )


# ============================================================
# STOP RECORDING
# ============================================================

@record_bp.route(
    "/stop_robot_recording",
    methods=["POST"],
)
def stop_robot_recording():
    data = _get_json()

    return jsonify(
        record_service.stop_robot_recording(
            arm_name=data.get("arm_name"),
            gripper_name=data.get("gripper_name"),
            dataset_format=data.get("dataset_format"),
        )
    )


# ============================================================
# GET RECORDING STATUS
# ============================================================

@record_bp.route(
    "/get_robot_recording_status",
    methods=["GET"],
)
def get_robot_recording_status():
    return jsonify(
        record_service
        .get_robot_recording_status(
            dataset_format=request.args.get("dataset_format")
        )
    )


# ============================================================
# GET TASK REGISTRY
# ============================================================

@record_bp.route(
    "/get_task_registry",
    methods=["GET"],
)
def get_task_registry():
    return jsonify(
        record_service.get_task_registry()
    )


# ============================================================
# GET REPLAY CATALOG
# ============================================================

@record_bp.route(
    "/get_replay_catalog",
    methods=["GET"],
)
def get_replay_catalog():
    return jsonify(
        record_service.get_replay_catalog()
    )


# ============================================================
# MOVE RECORDING GRIPPER
# ============================================================

@record_bp.route(
    "/move_recording_gripper",
    methods=["POST"],
)
def move_recording_gripper():
    data = _get_json()

    return jsonify(
        record_service.move_recording_gripper(
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
                    "wait",
                    False,
                ),

            timeout=
                data.get(
                    "timeout"
                ),
        )
    )


# ============================================================
# OPEN RECORDING GRIPPER
# ============================================================

@record_bp.route(
    "/open_recording_gripper",
    methods=["POST"],
)
def open_recording_gripper():
    data = _get_json()

    return jsonify(
        record_service.open_recording_gripper(
            gripper_name=data.get("gripper_name"),
            speed=data.get("speed"),
            force=data.get("force"),
            wait=data.get("wait", False),
            timeout=data.get("timeout"),
        )
    )


# ============================================================
# CLOSE RECORDING GRIPPER
# ============================================================
@record_bp.route(
    "/close_recording_gripper",
    methods=["POST"],
)
def close_recording_gripper():
    data = _get_json()

    return jsonify(
        record_service.close_recording_gripper(
            gripper_name=data.get("gripper_name"),
            speed=data.get("speed"),
            force=data.get("force"),
            wait=data.get("wait", False),
            timeout=data.get("timeout"),
        )
    )


# ============================================================
# START PLAYBACK
# ============================================================

@record_bp.route(
    "/start_robot_playback",
    methods=["POST"],
)
def start_robot_playback():
    data = _get_json()

    return jsonify(
        record_service.start_robot_playback(
            input_path=
                data.get(
                    "input_path"
                ),

            arm_name=
                data.get(
                    "arm_name"
                ),

            gripper_name=
                data.get(
                    "gripper_name"
                ),

            episode_index=
                data.get(
                    "episode_index",
                    0,
                ),

            speed=
                data.get(
                    "speed"
                ),

            acceleration=
                data.get(
                    "acceleration"
                ),

            lookahead_time=
                data.get(
                    "lookahead_time",
                    0.1,
                ),

            gain=
                data.get(
                    "gain",
                    300,
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

            dataset_format=data.get("dataset_format"),
        )
    )


# ============================================================
# STOP PLAYBACK
# ============================================================
@record_bp.route(
    "/stop_robot_playback",
    methods=["POST"],
)
def stop_robot_playback():
    data = _get_json()

    return jsonify(
        record_service.stop_robot_playback(
            arm_name=data.get("arm_name"),
            gripper_name=data.get("gripper_name"),
        )
    )

# ============================================================
# GET PLAYBACK STATUS
# ============================================================

@record_bp.route(
    "/get_robot_playback_status",
    methods=["GET"],
)
def get_robot_playback_status():

    return jsonify(
        record_service
        .get_robot_playback_status()
    )