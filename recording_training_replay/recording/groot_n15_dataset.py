"""GR00T N1.5 training-dataset writer using the LeRobot v2 layout.

This module owns the GR00T modality schema and LeRobot v2 serialization
required by the GR00T-compatible recording adapter.
"""

import json
import os
from types import SimpleNamespace

import cv2
import numpy as np

from services import recording_training_replay_service as _service
CODEBASE_VERSION = "v2.0"
SUPPORTED_ROBOT_TYPES = {"ur5", "ur7e"}

STATE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
ACTION_NAMES = [
    "delta_x", "delta_y", "delta_z",
    "delta_rx", "delta_ry", "delta_rz", "gripper",
]
LANGUAGE_COLUMN = "annotation.language.language_instruction"
DEFAULT_DATASET_DIR = os.path.abspath(os.environ.get(
    "LEROBOT_DATASET_ROOT",
    _service.DEFAULT_DATASET_DIR,
))
SCHEMA_DIR = os.path.join(DEFAULT_DATASET_DIR, "schema")
SCHEMA_PATH = os.path.join(SCHEMA_DIR, "modality.json")

# 建立 GR00T N1.5 使用的 modality schema。
def modality_config(camera_name, robot_type=None):
    """Return the GR00T N1.5 modality payload for a UR5/UR7e wrist camera."""
    if robot_type is not None:
        robot_type = str(robot_type).strip().lower()
        if robot_type not in SUPPORTED_ROBOT_TYPES:
            raise ValueError(
                "robot_type 必須是 " + " 或 ".join(sorted(SUPPORTED_ROBOT_TYPES))
            )
    if not isinstance(camera_name, str) or not camera_name.strip():
        raise ValueError("camera_name 必須是非空字串")
    camera_name = camera_name.strip()
    return {
        "state": {
            "eef_position": {"start": 0, "end": 3},
            "eef_rotation": {
                "start": 3, "end": 6, "rotation_type": "axis_angle",
            },
            "gripper_position": {"start": 6, "end": 7},
        },
        "action": {
            "eef_position_delta": {
                "start": 0, "end": 3, "absolute": False,
            },
            "eef_rotation_delta": {
                "start": 3,
                "end": 6,
                "absolute": False,
                "rotation_type": "axis_angle",
            },
            "gripper_position": {
                "start": 6, "end": 7, "absolute": True,
            },
        },
        "video": {
            "left_image": {
                "original_key": f"observation.images.{camera_name}",
            },
        },
        "annotation": {"language.language_instruction": {}},
    }


# 使用目前與下一個錄製 frame，產生一筆包含狀態與動作的 GR00T 訓練資料。
def sample_to_lerobot_v2_row(
    sample,
    next_sample,
    *,
    timestamp,
    task_index,
    episode_index,
    frame_index,
    index,
    done=False,
):
    """Map recorder samples to the numeric columns required by GR00T N1.5."""
    state = np.asarray(
        sample["tcp_pose"] + [sample["gripper"]], dtype=np.float32
    )
    delta = _service.tcp_local_delta(sample["tcp_pose"], next_sample["tcp_pose"])
    action = np.concatenate([
        delta,
        np.asarray([next_sample["gripper"]], dtype=np.float32),
    ])
    if state.shape != (7,) or action.shape != (7,):
        raise ValueError("GR00T N1.5 state/action 必須各包含 7 個值")
    return {
        "observation.state": state,
        "action": action,
        "timestamp": float(timestamp),
        LANGUAGE_COLUMN: int(task_index),
        "task_index": int(task_index),
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "index": int(index),
        "next.reward": np.float32(0.0),
        "next.done": bool(done),
    }


# 讀取 JSON Lines 檔案中的所有資料列。
def _jsonl_read(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


# 以安全替換方式寫入 JSON Lines 檔案。
def _jsonl_write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


# 以安全替換方式寫入 JSON 檔案。
def _json_write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    os.replace(temporary, path)


# 檢查目錄內是否包含任何檔案。
def _directory_has_files(path):
    return os.path.isdir(path) and any(
        filenames for _, _, filenames in os.walk(path)
    )


# 建立或驗證機器人共用的 GR00T schema。
def _ensure_shared_schema(camera_name, robot_type):
    """Create the canonical schema once and reject incompatible recordings."""
    expected = modality_config(camera_name, robot_type)
    schema_path = os.path.join(SCHEMA_DIR, robot_type, "modality.json")
    if os.path.isfile(schema_path):
        with open(schema_path, "r", encoding="utf-8") as source:
            existing = json.load(source)
        if existing != expected:
            raise ValueError(
                f"相機或 modality 與 {robot_type} 共用 schema 不一致: {schema_path}"
            )
    else:
        _json_write(schema_path, expected)
    return expected


# 根據錄製模式產生穩定的 v2 dataset 路徑。
def _resolve_dataset_path(
    task, arm_name, robot_type, dataset_mode="multi_task", dataset_name=None,
):
    """Return one stable v2 dataset according to mode, task, and arm."""
    robot_type = str(robot_type).strip().lower()
    if robot_type not in SUPPORTED_ROBOT_TYPES:
        raise ValueError("robot_type 必須是 " + " 或 ".join(sorted(SUPPORTED_ROBOT_TYPES)))
    mode = _service.normalize_dataset_mode(dataset_mode)
    folder_name = dataset_name if mode == "multi_task" else task
    if not str(folder_name or "").strip():
        raise ValueError("multi_task 必須設定 dataset_name")
    slug = (
        _service.dataset_slug(folder_name)
        if mode == "multi_task"
        else _service.task_slug(str(folder_name))
    )
    folder = os.path.join(mode, slug)
    return os.path.join(DEFAULT_DATASET_DIR, "lerobot_v2", folder)


# 產生相容舊呼叫方式的 multi-task 路徑。
def _resolve_multitask_path(task, robot_type, arm_name=None):
    """Backward-compatible helper for callers expecting multi-task mode."""
    arm_name = arm_name or ("left" if robot_type == "ur7e" else "right")
    return _resolve_dataset_path(
        task, arm_name, robot_type, "multi_task", dataset_name=task,
    )


class LeRobotV2DatasetWriter:
    """Small append-only LeRobot v2 writer matching the API used by v3 recorder."""

    # 初始化並驗證 LeRobot v2 dataset writer。
    def __init__(self, root, fps, camera_frames, robot_type):
        self.root = os.path.abspath(root)
        self.fps = int(fps)
        if self.fps != 10:
            raise ValueError("GR00T multi-task dataset 的錄製頻率固定為 10 Hz")
        self.camera_frames = camera_frames
        self.robot_type = robot_type
        if robot_type not in SUPPORTED_ROBOT_TYPES:
            raise ValueError("此 multi-task dataset 僅接受 UR5 或 UR7e 錄製資料")
        if len(camera_frames) != 1:
            raise ValueError("GR00T multi-task schema 必須且只能包含一台相機")
        camera_name = next(iter(camera_frames))
        image = camera_frames[camera_name]
        if image.shape[:2] != (480, 640):
            raise ValueError("GR00T multi-task dataset 的影像解析度固定為 640x480")
        schema = _ensure_shared_schema(camera_name, robot_type)
        self.info_path = os.path.join(self.root, "meta", "info.json")
        existing = _service.read_dataset_info(self.root)
        if existing:
            if existing.get("codebase_version") not in {"v2.0", "v2.1"}:
                raise ValueError("output_path 已包含非 LeRobot v2 資料")
            if int(existing.get("fps", self.fps)) != self.fps:
                raise ValueError("同一 LeRobot v2 資料集的 fps 必須一致")
            if existing.get("robot_type") != robot_type:
                raise ValueError("同一資料集不可混用 UR5 與 UR7e")
            self.info = existing
        else:
            if _directory_has_files(self.root):
                raise ValueError("output_path 已存在且不是 LeRobot v2 資料集")
            self.info = self._new_info()
        os.makedirs(os.path.join(self.root, "meta"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "data", "chunk-000"), exist_ok=True)
        os.makedirs(
            os.path.join(
                self.root, "videos", "chunk-000",
                f"observation.images.{camera_name}",
            ),
            exist_ok=True,
        )
        _json_write(os.path.join(self.root, "meta", "modality.json"), schema)
        self.info.setdefault("features", {}).setdefault(
            "observation.joints",
            {
                "dtype": "float32",
                "shape": [7],
                "names": list(_service.DEFAULT_ARM_JOINT_NAMES) + ["gripper"],
            },
        )
        self.meta = SimpleNamespace(total_episodes=int(self.info.get("total_episodes", 0)))
        self.rows = []
        self.task = None
        self.video_writers = {}

    # 建立新資料集的 info.json 內容。
    def _new_info(self):
        features = {
            "observation.state": {"dtype": "float32", "shape": [7], "names": STATE_NAMES},
            "observation.joints": {
                "dtype": "float32", "shape": [7],
                "names": list(_service.DEFAULT_ARM_JOINT_NAMES) + ["gripper"],
            },
            "action": {"dtype": "float32", "shape": [7], "names": ACTION_NAMES},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            LANGUAGE_COLUMN: {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "next.reward": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
        }
        for name, image in self.camera_frames.items():
            height, width = image.shape[:2]
            features[f"observation.images.{name}"] = {
                "dtype": "video", "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "fps": self.fps,
                "video_info": {"video.fps": self.fps},
            }
        return {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }

    # 取得既有 task index 或建立新的 index。
    def _task_index(self, task):
        path = os.path.join(self.root, "meta", "tasks.jsonl")
        tasks = _jsonl_read(path)
        for row in tasks:
            if str(row.get("task", "")).casefold() == str(task).casefold():
                return int(row["task_index"])
        task_index = max(
            (int(row.get("task_index", -1)) for row in tasks),
            default=-1,
        ) + 1
        tasks.append({"task_index": task_index, "task": task})
        _jsonl_write(path, tasks)
        return task_index

    # 取得或建立目前 episode 的相機影片 writer。
    def _video_writer(self, camera_name, image):
        writer = self.video_writers.get(camera_name)
        if writer is not None:
            return writer
        episode = self.meta.total_episodes
        path = os.path.join(
            self.root, "videos", "chunk-000", f"observation.images.{camera_name}",
            f"episode_{episode:06d}.mp4",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        height, width = image.shape[:2]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"無法建立 LeRobot v2 video: {path}")
        self.video_writers[camera_name] = writer
        return writer

    # 將一個錄製 frame 加入目前 episode。
    def add_frame(self, frame):
        task = str(frame.pop("task"))
        self.task = task
        frame_index = len(self.rows)
        episode_index = self.meta.total_episodes
        task_index = self._task_index(task)
        global_index = int(self.info.get("total_frames", 0)) + frame_index
        sample = frame.pop("_sample")
        next_sample = frame.pop("_next_sample")
        row = sample_to_lerobot_v2_row(
            sample, next_sample,
            timestamp=frame_index / self.fps,
            task_index=task_index,
            episode_index=episode_index,
            frame_index=frame_index,
            index=global_index,
            done=bool(frame.pop("_done")),
        )
        row["observation.joints"] = np.concatenate([
            np.asarray(frame.pop("_joints"), dtype=np.float32),
            np.asarray([sample["gripper"]], dtype=np.float32),
        ])
        if row["observation.joints"].shape != (7,):
            raise ValueError("observation.joints 必須包含 6 個關節角與 gripper")
        for camera_name, image in frame.pop("_images").items():
            self._video_writer(camera_name, image).write(image)
        self.rows.append(row)

    # 將目前 episode 寫入 Parquet、影片與 metadata。
    def save_episode(self):
        if not self.rows:
            raise RuntimeError("LeRobot v2 episode 沒有 frame")
        import pyarrow as pa
        import pyarrow.parquet as pq

        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        episode_index = self.meta.total_episodes
        path = os.path.join(
            self.root, "data", "chunk-000", f"episode_{episode_index:06d}.parquet"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        columns = {
            "observation.state": pa.array(
                [row["observation.state"].tolist() for row in self.rows],
                type=pa.list_(pa.float32(), 7),
            ),
            "action": pa.array(
                [row["action"].tolist() for row in self.rows],
                type=pa.list_(pa.float32(), 7),
            ),
            "observation.joints": pa.array(
                [row["observation.joints"].tolist() for row in self.rows],
                type=pa.list_(pa.float32(), 7),
            ),
            "timestamp": pa.array(
                [row["timestamp"] for row in self.rows], type=pa.float32()
            ),
        }
        for key in [
            LANGUAGE_COLUMN, "task_index", "episode_index", "frame_index", "index",
        ]:
            columns[key] = pa.array([row[key] for row in self.rows], type=pa.int64())
        columns["next.reward"] = pa.array(
            [row["next.reward"] for row in self.rows], type=pa.float32()
        )
        columns["next.done"] = pa.array(
            [row["next.done"] for row in self.rows], type=pa.bool_()
        )
        pq.write_table(pa.table(columns), path)
        episodes_path = os.path.join(self.root, "meta", "episodes.jsonl")
        episodes = _jsonl_read(episodes_path)
        episodes.append({
            "episode_index": episode_index,
            "tasks": [self.task],
            "length": len(self.rows),
        })
        _jsonl_write(episodes_path, episodes)
        self.info["total_episodes"] = episode_index + 1
        self.info["total_frames"] = int(self.info.get("total_frames", 0)) + len(self.rows)
        self.info["total_tasks"] = len(_jsonl_read(os.path.join(self.root, "meta", "tasks.jsonl")))
        self.info["total_videos"] = self.info["total_episodes"] * len(self.camera_frames)
        _json_write(self.info_path, self.info)
        _json_write(
            os.path.join(self.root, "meta", "modality.json"),
            modality_config(next(iter(self.camera_frames)), self.robot_type),
        )
        self.meta.total_episodes += 1
        self.rows = []

    # 關閉仍在使用的影片 writer。
    def finalize(self):
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()


# 建立 GR00T N1.5 訓練資料集 writer。
def create_writer(dataset_path, fps, camera_frames, robot_type):
    return LeRobotV2DatasetWriter(dataset_path, fps, camera_frames, robot_type)


# 封裝錄製樣本供 v2 writer 後續處理。
def encode_frame(sample, next_sample, task, done=False):
    return {
        "task": task,
        "_sample": sample,
        "_next_sample": next_sample,
        "_done": done,
        "_images": sample["images"],
        "_joints": sample["joints"],
    }
