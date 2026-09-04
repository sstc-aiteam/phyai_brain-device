"""LeRobot v2 recorder and GR00T N1.5 schema adapter.

The v2 writer stays separate from the production LeRobot v3 dataset writer so
selecting v2 cannot silently create v3 data.
"""

import json
import os
import threading
from types import SimpleNamespace

import cv2
import numpy as np

from services import recording_training_replay_service as _service
from utils.response import error


MODULE = "record_lerobotv2"
DATASET_FORMAT = "lerobot_v2"
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

_adapter_lock = threading.RLock()


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
    delta = _service._tcp_local_delta(sample["tcp_pose"], next_sample["tcp_pose"])
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


def _jsonl_read(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _jsonl_write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _json_write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    os.replace(temporary, path)


def _directory_has_files(path):
    return os.path.isdir(path) and any(
        filenames for _, _, filenames in os.walk(path)
    )


def _is_recoverable_incomplete_dataset(root, camera_name, expected_schema):
    """Recognize files left when recording stopped before its first save.

    ``info.json`` is intentionally written only after ``save_episode``.  A
    process interruption before that point can therefore leave the task,
    modality and episode-0 video behind.  Those files are safe to reuse: the
    next episode-0 video writer truncates the orphan video and the task list is
    append-only.  Anything else remains an error so an unrelated directory is
    never silently adopted as a dataset.
    """
    if not os.path.isdir(root):
        return False
    allowed_files = {
        os.path.join("meta", "tasks.jsonl"),
        os.path.join("meta", "modality.json"),
        os.path.join(
            "videos", "chunk-000", f"observation.images.{camera_name}",
            "episode_000000.mp4",
        ),
    }
    found_files = set()
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            found_files.add(os.path.relpath(os.path.join(directory, filename), root))
    if not found_files or not found_files.issubset(allowed_files):
        return False

    modality_path = os.path.join(root, "meta", "modality.json")
    if os.path.isfile(modality_path):
        try:
            with open(modality_path, "r", encoding="utf-8") as source:
                if json.load(source) != expected_schema:
                    return False
        except (OSError, ValueError, TypeError):
            return False
    tasks_path = os.path.join(root, "meta", "tasks.jsonl")
    if os.path.isfile(tasks_path):
        try:
            tasks = _jsonl_read(tasks_path)
            if any(
                not isinstance(row, dict)
                or not isinstance(row.get("task"), str)
                or not isinstance(row.get("task_index"), int)
                for row in tasks
            ):
                return False
        except (OSError, ValueError, TypeError):
            return False
    return True


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


def _resolve_dataset_path(
    task, arm_name, robot_type, dataset_mode="multi_task", dataset_name=None,
):
    """Return one stable v2 dataset according to mode, task, and arm."""
    robot_type = str(robot_type).strip().lower()
    if robot_type not in SUPPORTED_ROBOT_TYPES:
        raise ValueError("robot_type 必須是 " + " 或 ".join(sorted(SUPPORTED_ROBOT_TYPES)))
    mode = _service._normalize_dataset_mode(dataset_mode)
    folder_name = dataset_name if mode == "multi_task" else task
    if not str(folder_name or "").strip():
        raise ValueError("multi_task 必須設定 dataset_name")
    slug = (
        _service._dataset_slug(folder_name)
        if mode == "multi_task"
        else _service._task_slug(str(folder_name))
    )
    folder = os.path.join(mode, slug)
    return os.path.join(DEFAULT_DATASET_DIR, "lerobot_v2", folder)


def _resolve_multitask_path(task, robot_type, arm_name=None):
    """Backward-compatible helper for callers expecting multi-task mode."""
    arm_name = arm_name or ("left" if robot_type == "ur7e" else "right")
    return _resolve_dataset_path(
        task, arm_name, robot_type, "multi_task", dataset_name=task,
    )


class LeRobotV2DatasetWriter:
    """Small append-only LeRobot v2 writer matching the API used by v3 recorder."""

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
        existing = _service._read_dataset_info(self.root)
        if existing:
            if existing.get("codebase_version") not in {"v2.0", "v2.1"}:
                raise ValueError("output_path 已包含非 LeRobot v2 資料")
            if int(existing.get("fps", self.fps)) != self.fps:
                raise ValueError("同一 LeRobot v2 資料集的 fps 必須一致")
            if existing.get("robot_type") != robot_type:
                raise ValueError("同一資料集不可混用 UR5 與 UR7e")
            self.info = existing
        else:
            if _directory_has_files(self.root) and not _is_recoverable_incomplete_dataset(
                self.root, camera_name, schema
            ):
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

    def finalize(self):
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()


def _open_v2_dataset(dataset_path, fps, camera_frames, robot_type):
    return LeRobotV2DatasetWriter(dataset_path, fps, camera_frames, robot_type)


def _v2_frame(sample, next_sample, task, done=False):
    return {
        "task": task,
        "_sample": sample,
        "_next_sample": next_sample,
        "_done": done,
        "_images": sample["images"],
        "_joints": sample["joints"],
    }


def create_writer(dataset_path, fps, camera_frames, robot_type):
    return _open_v2_dataset(dataset_path, fps, camera_frames, robot_type)


def encode_frame(sample, next_sample, task, done=False):
    return _v2_frame(sample, next_sample, task, done=done)


def _rewrite_response(response):
    response["module"] = MODULE
    data = response.get("data")
    if isinstance(data, dict):
        data["format"] = DATASET_FORMAT
        data["codebase_version"] = CODEBASE_VERSION
        data["action_spaces"] = {"action": "tcp_local_delta"}
    return response


def start_robot_recording(*args, **kwargs):
    kwargs["dataset_format"] = "lerobot_v2"
    return _rewrite_response(_service.start_recording(*args, **kwargs))


def stop_robot_recording(*args, **kwargs):
    with _adapter_lock:
        return _rewrite_response(_service.stop_recording(*args, **kwargs))


def get_robot_recording_status():
    return _rewrite_response(_service.get_recording_status("lerobot_v2"))


def _load_lerobot_v2_episode(input_path, episode_index=0):
    """Load the joint trajectory directly from a LeRobot v2 parquet episode."""
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
        position = int(round(max(0.0, min(1.0, values[6])) * 255))
        if previous_position is None or position != previous_position:
            gripper_events.append({"t": float(row["timestamp"]), "position": position})
            previous_position = position
    return dataset_path, 1.0 / fps, joint_trajectory, gripper_events


def start_robot_playback(*args, **kwargs):
    with _adapter_lock:
        response = _service.start_robot_playback(
            *args, **kwargs, _episode_loader=_load_lerobot_v2_episode
        )
        return _rewrite_response(response)


def stop_robot_playback(*args, **kwargs):
    with _adapter_lock:
        return _rewrite_response(_service.stop_robot_playback(*args, **kwargs))


def get_robot_playback_status():
    with _adapter_lock:
        return _rewrite_response(_service.get_robot_playback_status())


# These operations do not define the recording serialization format.
get_task_registry = _service.get_task_registry
move_recording_gripper = _service.move_recording_gripper
open_recording_gripper = _service.open_recording_gripper
close_recording_gripper = _service.close_recording_gripper
