"""Load original recording samples without dataset conversion dependencies."""

import csv
import json
import math
from pathlib import Path

from recording_training_replay.replay.dataset import raw_episode_details


POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")


def read_frames(path):
    """Normalize both expanded CSV and older array-valued CSV for replay."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expanded = "state_x" in (reader.fieldnames or [])
        rows = []
        for index, row in enumerate(reader):
            if expanded:
                rows.append({"timestamp": float(row["timestamp"]),
                             "tcp_pose": [float(row[f"state_{axis}"]) for axis in POSE_AXES],
                             "joints": [float(row[f"joint_{i}"]) for i in range(1, 7)],
                             "gripper": float(row.get("gripper_command", row.get("state_gripper"))),
                             "frame_index": int(row.get("frame_index", index)),
                             "images": {}})
            else:
                rows.append({"timestamp": float(row["timestamp"]),
                             "tcp_pose": json.loads(row["tcp_pose"]), "joints": json.loads(row["joints"]),
                             "gripper": float(row["gripper"]), "frame_index": int(row["frame_index"]),
                             "images": json.loads(row["images"])})
        return rows


class JointTrajectory(list):
    """Joint measurements plus original relative sample times for playback."""
    def __init__(self):
        super().__init__()
        self.sample_times = []


def load_raw_episode(input_path, episode_index=0):
    dataset_path = str(Path(input_path).resolve())
    try:
        episode_index = int(episode_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("episode_index 必須是整數") from exc
    if episode_index < 0:
        raise ValueError("episode_index 不可小於 0")
    episode = next((item for item in raw_episode_details(dataset_path)
                    if item["episode_index"] == episode_index), None)
    if episode is None:
        raise ValueError(f"找不到已完成的 raw episode {episode_index}")
    metadata = episode["metadata"]
    interval = float(metadata["interval"])
    csv_path = Path(episode["episode_path"]) / "frames.csv"
    if csv_path.is_file():
        rows = read_frames(csv_path)
    else:
        with (Path(episode["episode_path"]) / "samples.jsonl").open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) != int(metadata["sample_count"]):
        raise ValueError("raw episode sample_count 與資料不一致")
    trajectory, events = JointTrajectory(), []
    previous = None
    start_timestamp = None
    last_timestamp = None
    for frame_number, row in enumerate(rows):
        if row.get("frame_index") != frame_number:
            raise ValueError("raw episode frame_index 必須從 0 連續遞增")
        joints = list(map(float, row["joints"]))
        position = float(row["gripper"])
        timestamp = float(row["timestamp"])
        if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
            raise ValueError("raw joints 必須包含 6 個有限關節角")
        if not math.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("raw gripper 必須介於 0 和 1")
        invalid_order = last_timestamp is not None and (timestamp <= last_timestamp if metadata.get("timestamp_order") == "strictly_increasing" else timestamp < last_timestamp)
        if not math.isfinite(timestamp) or invalid_order:
            raise ValueError("raw timestamp 必須是有限且非遞減的數值")
        if start_timestamp is None:
            start_timestamp = timestamp
        last_timestamp = timestamp
        trajectory.append(joints)
        trajectory.sample_times.append(timestamp - start_timestamp)
        if previous is None or position != previous:
            events.append({"frame_index": frame_number, "t": timestamp - start_timestamp,
                           "position": position})
            previous = position
    return dataset_path, interval, trajectory, events
