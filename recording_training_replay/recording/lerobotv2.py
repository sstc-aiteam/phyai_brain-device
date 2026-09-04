"""LeRobot v2 recording adapter for the shared recording service."""

from recording_training_replay.recording.lerobot_common import *  # noqa: F401,F403
from recording_training_replay.recording import lerobot_common as _common
from recording_training_replay.recording import groot_n15_dataset as _dataset


MODULE = "record_lerobotv2"
DATASET_FORMAT = "lerobot_v2"
CODEBASE_VERSION = _dataset.CODEBASE_VERSION

def create_writer(dataset_path, fps, camera_frames, robot_type):
    return _dataset._open_v2_dataset(
        dataset_path, fps, camera_frames, robot_type
    )


def encode_frame(sample, next_sample, task, done=False):
    return _dataset._v2_frame(sample, next_sample, task, done=done)


def finalize_writer(writer, dataset_path):
    _common.finalize_writer(writer, dataset_path)


def _rewrite_response(response):
    response["module"] = MODULE
    data = response.get("data")
    if isinstance(data, dict):
        data["format"] = DATASET_FORMAT
        data["codebase_version"] = CODEBASE_VERSION
        data["action_spaces"] = {"action": "tcp_local_delta"}
    return response


def start_robot_recording(*args, **kwargs):
    response = _common.start_format_recording(DATASET_FORMAT, *args, **kwargs)
    return _rewrite_response(response)


def stop_robot_recording(*args, **kwargs):
    return _rewrite_response(_common.stop_recording(*args, **kwargs))


def get_robot_recording_status():
    return _rewrite_response(_common.get_recording_status(DATASET_FORMAT))


def __getattr__(name):
    """Expose the GR00T dataset helpers through the v2 adapter."""
    try:
        return getattr(_dataset, name)
    except AttributeError:
        return getattr(_common, name)
