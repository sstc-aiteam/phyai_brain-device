"""HTTP endpoints for browsing and replaying recorded trajectories."""

from flask import Blueprint, jsonify, request

from services import recording_training_replay_service as subsystem_service

replay_bp = Blueprint("replay", __name__, url_prefix="/api/record")


# 回傳可供 Replay 的資料集清單。
@replay_bp.get("/get_replay_catalog")
def get_replay_catalog():
    return jsonify(subsystem_service.get_replay_catalog())


# 啟動指定 episode 的機器人回放。
@replay_bp.post("/start_robot_playback")
def start_robot_playback():
    data = request.get_json(silent=True) or {}
    return jsonify(subsystem_service.start_replay_service(
        input_path=data.get("input_path"), arm_name=data.get("arm_name"),
        gripper_name=data.get("gripper_name"), episode_index=data.get("episode_index", 0),
        speed=data.get("speed"), acceleration=data.get("acceleration"),
        lookahead_time=data.get("lookahead_time", 0.1), gain=data.get("gain", 300),
        move_to_start=data.get("move_to_start", True),
        move_to_start_speed=data.get("move_to_start_speed"),
        move_to_start_acceleration=data.get("move_to_start_acceleration"),
        dataset_format=data.get("dataset_format"),
    ))


# 停止目前進行中的機器人回放。
@replay_bp.post("/stop_robot_playback")
def stop_robot_playback():
    data = request.get_json(silent=True) or {}
    return jsonify(subsystem_service.stop_replay_service(
        arm_name=data.get("arm_name"), gripper_name=data.get("gripper_name"),
    ))


# 回傳目前的機器人回放狀態。
@replay_bp.get("/get_robot_playback_status")
def get_robot_playback_status():
    return jsonify(subsystem_service.get_replay_status_service())
