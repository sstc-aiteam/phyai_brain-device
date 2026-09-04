"""LeRobot v3 dataset-format adapter for the shared LeRobot service."""

from recording_training_replay.recording.lerobot_common import *  # noqa: F401,F403
from recording_training_replay.recording import lerobot_common as _common


DATA_PATH = "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet"
VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/"
    "episode_{file_index:06d}.mp4"
)


def create_writer(dataset_path, fps, camera_frames, robot_type):
    dataset = _common.open_lerobot_v3_dataset(
        dataset_path, fps, camera_frames, robot_type
    )
    if dataset.meta.total_episodes > 0 and dataset.meta.data_path != DATA_PATH:
        raise ValueError(
            "既有 LeRobot v3 dataset 不是 episode 檔名格式；"
            "請刪除測試資料後重新錄製"
        )

    dataset.meta.info.data_path = DATA_PATH
    if dataset.meta.video_keys:
        dataset.meta.info.video_path = VIDEO_PATH

    from lerobot.datasets.io_utils import write_info
    write_info(dataset.meta.info, dataset.meta.root)
    return dataset


def encode_frame(sample, next_sample, task, done=False):
    return _common.encode_lerobot_v3_frame(
        sample, next_sample, task, done=done
    )


def finalize_writer(writer, dataset_path):
    _common.finalize_writer(writer, dataset_path)


def start_robot_recording(*args, **kwargs):
    return _common.start_format_recording("lerobot_v3", *args, **kwargs)


def stop_robot_recording(*args, **kwargs):
    return _common.stop_recording(*args, **kwargs)


def get_robot_recording_status():
    return _common.get_recording_status("lerobot_v3")
