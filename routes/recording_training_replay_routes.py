"""HTTP endpoints for recording, replay, and training."""

from flask import Blueprint, jsonify, request

from services import recording_training_replay_service as subsystem_service


# ============================================================
# RECORDING
# ============================================================

record_bp = Blueprint("record", __name__, url_prefix="/api/record")


# 取得請求中的 JSON 資料。
def _json():
    return request.get_json(silent=True) or {}


# 接收並啟動機器人錄製請求。
@record_bp.post("/start_robot_recording")
def start_robot_recording():
    data = _json()
    return jsonify(subsystem_service.start_recording(
        arm_name=data.get("arm_name"), gripper_name=data.get("gripper_name"),
        output_path=data.get("output_path"), interval=data.get("interval", 0.1),
        freedrive=data.get("freedrive", True), record_video=data.get("record_video", True),
        task=data.get("task", "robot demonstration"),
        initial_gripper_position=data.get("initial_gripper_position", 0),
        dataset_mode=data.get("dataset_mode", "multi_task"),
        dataset_format=data.get("dataset_format", "lerobot_v3"),
        dataset_name=data.get("dataset_name"),
    ))


# 停止錄製並儲存目前 episode。
@record_bp.post("/stop_robot_recording")
def stop_robot_recording():
    data = _json()
    return jsonify(subsystem_service.stop_recording(
        arm_name=data.get("arm_name"), gripper_name=data.get("gripper_name"),
        dataset_format=data.get("dataset_format"),
    ))


# 回傳目前的錄製狀態。
@record_bp.get("/get_robot_recording_status")
def get_robot_recording_status():
    return jsonify(subsystem_service.get_recording_status(
        dataset_format=request.args.get("dataset_format")
    ))


# 回傳已註冊的錄製任務清單。
@record_bp.get("/get_task_registry")
def get_task_registry():
    return jsonify(subsystem_service.get_task_registry())


@record_bp.get("/startup_status")
def recording_startup_status():
    return jsonify(subsystem_service.get_startup_status())


# 新增一筆錄製任務。
@record_bp.post("/register_task")
def register_task():
    return jsonify(subsystem_service.register_task(_json().get("task")))


# 解析參數並執行指定的夾爪操作。
def _gripper_call(action):
    data = _json()
    return jsonify(action(
        gripper_name=data.get("gripper_name"), speed=data.get("speed"),
        force=data.get("force"), wait=data.get("wait", False),
        timeout=data.get("timeout"),
    ))


# 將錄製用夾爪移動至指定位置。
@record_bp.post("/move_recording_gripper")
def move_recording_gripper():
    data = _json()
    return jsonify(subsystem_service.move_recording_gripper(
        position=data.get("position"), speed=data.get("speed"), force=data.get("force"),
        wait=data.get("wait", False), timeout=data.get("timeout"),
    ))


# 開啟錄製用夾爪。
@record_bp.post("/open_recording_gripper")
def open_recording_gripper():
    return _gripper_call(subsystem_service.open_recording_gripper)


# 關閉錄製用夾爪。
@record_bp.post("/close_recording_gripper")
def close_recording_gripper():
    return _gripper_call(subsystem_service.close_recording_gripper)


# ============================================================
# REPLAY
# ============================================================

replay_bp = Blueprint("replay", __name__, url_prefix="/api/record")


# 回傳可供 Replay 的資料集清單。
@replay_bp.get("/get_replay_catalog")
def get_replay_catalog():
    return jsonify(subsystem_service.get_replay_catalog())


# 啟動指定 episode 的機器人回放。
@replay_bp.post("/start_robot_playback")
def start_robot_playback():
    data = _json()

    return jsonify(
        subsystem_service.start_robot_playback(
            input_path=data.get("input_path"),
            arm_name=data.get("arm_name"),
            gripper_name=data.get("gripper_name"),
            episode_index=data.get("episode_index", 0),
            speed=data.get("speed"),
            acceleration=data.get("acceleration"),
            lookahead_time=data.get("lookahead_time", 0.1),
            gain=data.get("gain", 300),
            move_to_start=data.get("move_to_start", True),
            move_to_start_speed=data.get("move_to_start_speed"),
            move_to_start_acceleration=data.get("move_to_start_acceleration"),
            dataset_format=data.get("dataset_format", "lerobot_v3"),
        )
    )



# 停止目前進行中的機器人回放。
@replay_bp.post("/stop_robot_playback")
def stop_robot_playback():
    data = request.get_json(silent=True) or {}
    return jsonify(subsystem_service.stop_robot_playback(
        arm_name=data.get("arm_name"), gripper_name=data.get("gripper_name"),
    ))


# 回傳目前的機器人回放狀態。
@replay_bp.get("/get_robot_playback_status")
def get_robot_playback_status():
    return jsonify(subsystem_service.get_robot_playback_status())


# ============================================================
# TRAINING
# ============================================================

training_bp = Blueprint("training", __name__, url_prefix="/api/training")


def _training_token():
    return request.headers.get("Authorization", "").removeprefix("Bearer ")


def _training_call(action, **kwargs):
    try:
        return jsonify(action(token=_training_token(), **kwargs))
    except PermissionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 401
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except LookupError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404


@training_bp.post("/auth/register")
def training_register():
    data = _json()
    return _training_call(subsystem_service.register_training_user,
                          username=data.get("username"), password=data.get("password"))


@training_bp.post("/auth/login")
def training_login():
    data = _json()
    return _training_call(subsystem_service.login_training_user,
                          username=data.get("username"), password=data.get("password"))


@training_bp.post("/auth/logout")
def training_logout():
    return _training_call(subsystem_service.training_logout)


@training_bp.get("/jobs")
def training_jobs():
    return _training_call(subsystem_service.get_training_jobs, query=request.args.get("q", ""))


@training_bp.get("/jobs/<job_id>")
def training_job(job_id):
    return _training_call(subsystem_service.get_training_jobs, job_id=job_id)


def upload_training_dataset():
    data = _json()
    return _training_call(subsystem_service.start_dataset_upload,
                          dataset_path=data.get("dataset_path"), model=data.get("model"))


def get_dataset_upload_status():
    return _training_call(subsystem_service.get_dataset_upload_status)


training_bp.add_url_rule("/datasets", view_func=upload_training_dataset, methods=["POST"])
training_bp.add_url_rule("/datasets/status", view_func=get_dataset_upload_status, methods=["GET"])


BLUEPRINTS = (
    record_bp,
    replay_bp,
    training_bp,
)

__all__ = ["BLUEPRINTS"]
