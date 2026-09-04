"""HTTP endpoints for submitting datasets to the training backend."""

from flask import Blueprint, jsonify, request

from services import recording_training_replay_service as subsystem_service

training_bp = Blueprint("training", __name__, url_prefix="/api/training")
legacy_training_bp = Blueprint("legacy_training", __name__, url_prefix="/api/record")


# 上傳資料集並建立訓練工作。
def upload_training_dataset():
    data = request.get_json(silent=True) or {}
    return jsonify(subsystem_service.start_dataset_upload(
        dataset_path=data.get("dataset_path"), server_url=data.get("server_url"),
        api_token=data.get("api_token"), metadata=data.get("metadata"),
        timeout_seconds=data.get("timeout_seconds"),
    ))


# 回傳訓練資料集的上傳狀態。
def get_dataset_upload_status():
    return jsonify(subsystem_service.get_dataset_upload_status())


training_bp.add_url_rule("/datasets", view_func=upload_training_dataset, methods=["POST"])
training_bp.add_url_rule("/datasets/status", view_func=get_dataset_upload_status, methods=["GET"])
legacy_training_bp.add_url_rule(
    "/upload_training_dataset", view_func=upload_training_dataset, methods=["POST"]
)
legacy_training_bp.add_url_rule(
    "/get_dataset_upload_status", view_func=get_dataset_upload_status, methods=["GET"]
)
