"""LeRobot v3 dataset-format adapter for the shared LeRobot service."""

from recording_training_replay.recording import lerobot_common as _common
from services import recording_training_replay_service as _service


DATA_PATH = "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet"
VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/"
    "episode_{file_index:06d}.mp4"
)


# 建立或恢復 LeRobot v3 dataset writer。
def create_writer(dataset_path, fps, camera_frames, robot_type):
    dataset = _service.open_lerobot_dataset(
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


# 將錄製樣本編碼成 LeRobot v3 frame。
def encode_frame(sample, next_sample, task, done=False):
    return _service.sample_to_lerobot_frame(
        sample, next_sample, task, done=done
    )


# 完成 v3 writer 的資料寫入。
def finalize_writer(writer, dataset_path):
    _common.finalize_writer(writer, dataset_path)


# 以 LeRobot v3 格式啟動錄製。
def start_robot_recording(*args, **kwargs):
    return _common.start_format_recording("lerobot_v3", *args, **kwargs)


# 停止目前的 LeRobot v3 錄製。
def stop_robot_recording(*args, **kwargs):
    return _common.stop_recording(*args, **kwargs)


# 取得 LeRobot v3 的錄製狀態。
def get_robot_recording_status():
    return _common.get_recording_status("lerobot_v3")
