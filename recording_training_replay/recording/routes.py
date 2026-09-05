"""HTTP endpoints for collecting robot demonstrations only."""

from flask import Blueprint, jsonify, request

from services import recording_training_replay_service as subsystem_service

record_bp = Blueprint("record", __name__, url_prefix="/api/record")


# 取得請求中的 JSON 資料。
def _json():
    return request.get_json(silent=True) or {}


# 接收並啟動機器人錄製請求。
@record_bp.post("/start_robot_recording")
def start_robot_recording():
    data = _json()
    return jsonify(subsystem_service.start_recording_service(
        arm_name=data.get("arm_name"), gripper_name=data.get("gripper_name"),
        output_path=data.get("output_path"), interval=data.get("interval", 0.1),
        freedrive=data.get("freedrive", False), record_video=data.get("record_video", False),
        task=data.get("task", "robot demonstration"),
        initial_gripper_position=data.get("initial_gripper_position", 0),
        dataset_mode=data.get("dataset_mode", "multi_task"),
        dataset_format=data.get("dataset_format", "lerobot_v3"),
        training_target=data.get("training_target"),
        dataset_name=data.get("dataset_name"),
    ))


# 停止錄製並儲存目前 episode。
@record_bp.post("/stop_robot_recording")
def stop_robot_recording():
    data = _json()
    return jsonify(subsystem_service.stop_recording_service(
        arm_name=data.get("arm_name"), gripper_name=data.get("gripper_name"),
        dataset_format=data.get("dataset_format"),
    ))


# 回傳目前的錄製狀態。
@record_bp.get("/get_robot_recording_status")
def get_robot_recording_status():
    return jsonify(subsystem_service.get_recording_status_service(
        dataset_format=request.args.get("dataset_format")
    ))


# 回傳已註冊的錄製任務清單。
@record_bp.get("/get_task_registry")
def get_task_registry():
    return jsonify(subsystem_service.get_task_registry())


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
