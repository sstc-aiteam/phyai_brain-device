"""Dataset episode loaders used by the replay service."""

import os

import numpy as np

from services import recording_training_replay_service as _service


def load_lerobot_v2_episode(input_path, episode_index=0):
    """Load a joint trajectory from a LeRobot v2 parquet episode."""
    import pyarrow.parquet as pq

    dataset_path = _service._normalize_input_path(input_path)
    try:
        episode_index = int(episode_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("episode_index 必須是整數") from exc
    if episode_index < 0:
        raise ValueError("episode_index 不可小於 0")

    info = _service._read_dataset_info(dataset_path) or {}
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        raise ValueError("選取的資料集不是 LeRobot v2")
    fps = int(info.get("fps", 0))
    if fps <= 0:
        raise ValueError("LeRobot v2 dataset 的 fps 無效")

    chunk_size = max(1, int(info.get("chunks_size", 1000)))
    episode_chunk = episode_index // chunk_size
    data_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    episode_path = os.path.join(
        dataset_path,
        data_template.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        ),
    )
    if not os.path.isfile(episode_path):
        raise ValueError(f"找不到 LeRobot v2 episode {episode_index}")

    table = pq.read_table(episode_path)
    if "observation.joints" not in table.column_names:
        raise ValueError(
            "此 v2 episode 未保存 observation.joints，無法安全 Replay；"
            "請使用修正後的版本重新錄製。"
        )
    rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
    if not rows:
        raise ValueError(f"LeRobot v2 episode {episode_index} 沒有 frame")
    joint_and_gripper_trajectory = [
        list(map(float, row["observation.joints"])) for row in rows
    ]
    if any(len(values) != 7 for values in joint_and_gripper_trajectory):
        raise ValueError("observation.joints 必須包含 6 個關節角與 gripper")
    joint_trajectory = [values[:6] for values in joint_and_gripper_trajectory]

    gripper_events = []
    previous_position = None
    for row, values in zip(rows, joint_and_gripper_trajectory):
        position = max(0.0, min(1.0, values[6]))
        if previous_position is None or position != previous_position:
            gripper_events.append({
                "t": float(row["timestamp"]),
                "position": position,
            })
            previous_position = position
    return dataset_path, 1.0 / fps, joint_trajectory, gripper_events


def load_lerobot_v3_episode(input_path, episode_index=0):
    """Load a joint trajectory from a LeRobot v3 parquet episode."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow unavailable，無法讀取 Replay parquet") from exc

    dataset_path = _service._normalize_input_path(input_path)
    try:
        episode_index = int(episode_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("episode_index 必須是整數") from exc
    if episode_index < 0:
        raise ValueError("episode_index 不可小於 0")

    info = _service._read_dataset_info(dataset_path) or {}
    if info.get("codebase_version") != "v3.0":
        raise ValueError("選取的資料集不是 LeRobot v3")
    fps = float(info.get("fps", 0))
    if fps <= 0:
        raise ValueError("LeRobot v3 dataset 的 fps 無效")

    episode = next(
        (
            item
            for item in _service._parquet_episode_details(dataset_path, info)
            if item["episode_index"] == episode_index
        ),
        None,
    )
    if episode is None:
        raise ValueError(f"找不到 LeRobot v3 episode {episode_index}")

    table = pq.read_table(episode["parquet_path"])
    joint_key = "observation.state"
    if joint_key not in table.column_names:
        raise ValueError("v3 episode 沒有 observation.state，無法 Replay")
    if "frame_index" not in table.column_names:
        raise ValueError("v3 episode 沒有 frame_index，無法 Replay")
    rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
    if not rows:
        raise ValueError(f"episode {episode_index} 沒有 frame")

    values = [list(map(float, row[joint_key])) for row in rows]
    if any(len(item) != 7 for item in values):
        raise ValueError("observation.state 必須包含 6 個關節角與 gripper")
    if not all(np.isfinite(value) for item in values for value in item):
        raise ValueError("observation.state 包含 NaN 或 Infinity")

    trajectory = [item[:6] for item in values]
    events = []
    previous = None
    for frame_number, (row, item) in enumerate(zip(rows, values)):
        position = max(0.0, min(1.0, item[6]))
        if previous is None or position != previous:
            events.append({
                "frame_index": frame_number,
                "t": float(row.get("timestamp", frame_number / fps)),
                "position": position,
            })
            previous = position
    return dataset_path, 1.0 / fps, trajectory, events
