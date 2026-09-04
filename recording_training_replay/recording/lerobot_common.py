"""Shared adapter operations for LeRobot v2 and v3.

The complete recording, replay, and training service remains in
``services.recording_training_replay_service``.  This module only centralizes
the glue shared by the two LeRobot format adapters so that v2 and v3 do not
duplicate writer finalization, recording dispatch, and recording-status code.
"""

from services import recording_training_replay_service as _service


# Dataset implementations use these canonical service constants.
DEFAULT_DATASET_DIR = _service.DEFAULT_DATASET_DIR
DEFAULT_ARM_JOINT_NAMES = _service.DEFAULT_ARM_JOINT_NAMES


def open_lerobot_v3_dataset(dataset_path, fps, camera_frames, robot_type):
    """Create or resume the official LeRobot v3 dataset writer."""
    return _service._open_lerobot_dataset(
        dataset_path, fps, camera_frames, robot_type
    )


def encode_lerobot_v3_frame(sample, next_sample, task, done=False):
    """Convert captured samples into one LeRobot v3 frame."""
    return _service._sample_to_lerobot_frame(
        sample, next_sample, task, done=done
    )


def finalize_writer(writer, dataset_path=None):
    """Finalize a v2 or v3 dataset writer."""
    del dataset_path
    writer.finalize()


def start_format_recording(dataset_format, *args, **kwargs):
    """Start the service recorder with an explicit dataset format."""
    kwargs["dataset_format"] = dataset_format
    return _service.start_recording(*args, **kwargs)


def stop_recording(*args, **kwargs):
    """Stop the single recording session owned by the service."""
    return _service.stop_recording(*args, **kwargs)


def get_recording_status(dataset_format=None):
    """Read service recording status for an optional dataset format."""
    return _service.get_recording_status(dataset_format)


# Recording and dataset helpers shared by the format adapters.
get_task_registry = _service.get_task_registry
move_recording_gripper = _service.move_recording_gripper
open_recording_gripper = _service.open_recording_gripper
close_recording_gripper = _service.close_recording_gripper
_normalize_dataset_mode = _service._normalize_dataset_mode
_dataset_slug = _service._dataset_slug
_task_slug = _service._task_slug
_read_dataset_info = _service._read_dataset_info
_tcp_local_delta = _service._tcp_local_delta
