"""LeRobot v2 recording adapter for the shared recording service."""

from recording_training_replay.recording import lerobot_common as _common
from recording_training_replay.recording import groot_n15_dataset as _dataset


MODULE = "record_lerobotv2"
DATASET_FORMAT = "lerobot_v2"
CODEBASE_VERSION = _dataset.CODEBASE_VERSION

# 建立 GR00T 相容的 LeRobot v2 writer。
def create_writer(dataset_path, fps, camera_frames, robot_type):
    return _dataset.create_writer(
        dataset_path, fps, camera_frames, robot_type
    )


# 將錄製樣本編碼成 LeRobot v2 frame。
def encode_frame(sample, next_sample, task, done=False):
    return _dataset.encode_frame(sample, next_sample, task, done=done)


# 完成 v2 writer 的資料寫入。
def finalize_writer(writer, dataset_path):
    _common.finalize_writer(writer, dataset_path)


# 補上 LeRobot v2 回應的格式資訊。
def _rewrite_response(response):
    response["module"] = MODULE
    data = response.get("data")
    if isinstance(data, dict):
        data["format"] = DATASET_FORMAT
        data["codebase_version"] = CODEBASE_VERSION
        data["action_spaces"] = {"action": "tcp_local_delta"}
    return response


# 以 LeRobot v2 格式啟動錄製。
def start_robot_recording(*args, **kwargs):
    response = _common.start_format_recording(DATASET_FORMAT, *args, **kwargs)
    return _rewrite_response(response)


# 停止 v2 錄製並整理回應內容。
def stop_robot_recording(*args, **kwargs):
    return _rewrite_response(_common.stop_recording(*args, **kwargs))


# 取得 LeRobot v2 的錄製狀態。
def get_robot_recording_status():
    return _rewrite_response(_common.get_recording_status(DATASET_FORMAT))
