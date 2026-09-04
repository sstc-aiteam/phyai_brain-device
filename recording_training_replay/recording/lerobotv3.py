"""LeRobot v3 adapter for the unified recording service."""

from services.recording_training_replay_service import *  # noqa: F401,F403
from services import recording_training_replay_service as _service


def create_writer(dataset_path, fps, camera_frames, robot_type):
    return _service._open_lerobot_dataset(
        dataset_path, fps, camera_frames, robot_type
    )


def encode_frame(sample, next_sample, task, done=False):
    return _service._sample_to_lerobot_frame(
        sample, next_sample, task, done=done
    )


def start_robot_recording(*args, **kwargs):
    kwargs["dataset_format"] = "lerobot_v3"
    return _service.start_recording(*args, **kwargs)


def start_robot_playback(*args, **kwargs):
    return _service.start_robot_playback(
        *args,
        **kwargs,
        _episode_loader=_service._load_lerobot_episode,
    )


def __getattr__(name):
    return getattr(_service, name)
