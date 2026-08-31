"""LeRobot v3 backend.

The shared recording/session implementation lives in :mod:`recording.core`.
This module exposes the v3 backend without being imported by the v2 backend.
"""

from recording.core import *  # noqa: F401,F403
from recording import core as _core


def start_robot_recording(*args, **kwargs):
    return _core.start_robot_recording(
        *args,
        **kwargs,
        _dataset_opener=_core._open_lerobot_dataset,
        _frame_mapper=_core._sample_to_lerobot_frame,
    )


def start_robot_playback(*args, **kwargs):
    return _core.start_robot_playback(
        *args,
        **kwargs,
        _episode_loader=_core._load_lerobot_episode,
    )


def __getattr__(name):
    return getattr(_core, name)
