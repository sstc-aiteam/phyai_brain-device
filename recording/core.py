import json
import logging
import os
import re
import threading
import time
from glob import glob
from datetime import datetime, timezone

import numpy as np

from services import arm_service, camera_service, gripper_service
from utils.response import success, error


MODULE = "record_lerobotv3"
logger = logging.getLogger(__name__)


DEFAULT_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


DEFAULT_RECORD_INTERVAL = 0.1
DEFAULT_DATASET_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "lerobot_datasets",
    )
)
TASK_REGISTRY_PATH = os.path.join(DEFAULT_DATASET_DIR, "task_registry.json")

LEROBOT_STATE_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
]

LEROBOT_TCP_POSE_NAMES = [
    "x", "y", "z", "rx", "ry", "rz", "gripper",
]

LEROBOT_TCP_ACTION_NAMES = [
    "tcp_local_delta_x", "tcp_local_delta_y", "tcp_local_delta_z",
    "tcp_local_delta_rx", "tcp_local_delta_ry", "tcp_local_delta_rz",
    "gripper",
]

LEROBOT_TCP_BASE_ACTION_NAMES = [
    "tcp_base_delta_x", "tcp_base_delta_y", "tcp_base_delta_z",
    "tcp_base_delta_rx", "tcp_base_delta_ry", "tcp_base_delta_rz",
    "gripper",
]

LEROBOT_JOINT_ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
]

LEROBOT_ACTION_SPACES = {
    "action_tcp": "tcp_local_delta",
    "action_tcp_base": "tcp_base_delta",
    "action_joints": "joint_absolute",
}


# ============================================================
# Recording State
# ============================================================

_record_lock = threading.RLock()
_record_stop_event = threading.Event()

_record_thread = None

_record_gripper_events = []
_record_dataset = None
_record_pending_sample = None
_record_saved_frame_count = 0
_record_gripper_position = 0

_record_start_time = None
_record_output_path = None

_record_interval = DEFAULT_RECORD_INTERVAL

_record_arm_name = None
_record_gripper_name = None

_record_freedrive = False
_record_camera_names = []
_record_task = None
_record_task_id = None
_record_frame_mapper = None

_task_registry_lock = threading.RLock()


# ============================================================
# Playback State
# ============================================================

_playback_lock = threading.RLock()
_playback_stop_event = threading.Event()

_playback_thread = None

_playback_input_path = None
_playback_arm_name = None
_playback_gripper_name = None

_playback_start_time = None
_playback_phase = "idle"
_playback_error = None
_playback_completed = False

# Robotiq e-Series commands are complete URScript programs injected through
# port 30002.  The TCP send returns before the controller has necessarily
# started and finished that program, so immediately re-uploading RTDE can race
# with a delayed Robotiq program and leave ServoJ without its control script.
ROBOTIQ_ESERIES_SCRIPT_TIMEOUT_SECONDS = 5.0


# ============================================================
# Common Helpers
# ============================================================

def _settle_robotiq_eseries_script(arm_name):
    """Synchronize with the actual port-30002 program, without fixed delay."""
    response = arm_service.wait_for_external_script_completion(
        arm_name=arm_name,
        timeout=ROBOTIQ_ESERIES_SCRIPT_TIMEOUT_SECONDS,
        cancel_event=_playback_stop_event,
    )
    if _playback_stop_event.is_set():
        return False
    _require_success(
        response,
        "arm_service.wait_for_external_script_completion",
    )
    return True

def _project_root():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
        )
    )


def _ensure_directory(
    path,
):
    directory = os.path.dirname(
        path
    )

    if (
        directory
        and not os.path.isdir(
            directory
        )
    ):
        os.makedirs(
            directory,
            exist_ok=True,
        )


def _relative_path(
    path,
):
    if path is None:
        return None

    return os.path.relpath(
        path,
        _project_root(),
    )


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _empty_task_registry():
    return {
        "version": 2,
        "dataset_root": DEFAULT_DATASET_DIR,
        "naming_rule": {
            "multi_task": "lerobot_vN/multitask_<arm-name>",
            "single_task": "lerobot_vN/single_task/<task-slug>_<arm-name>",
        },
        "episode_label_rule": "episode_NNNNNN_<arm-name>_YYYYMMDD_HHMMSS",
        "tasks": [],
    }


def _load_task_registry():
    if not os.path.isfile(TASK_REGISTRY_PATH):
        return _empty_task_registry()
    with open(TASK_REGISTRY_PATH, "r", encoding="utf-8") as source:
        registry = json.load(source)
    if not isinstance(registry, dict) or not isinstance(registry.get("tasks"), list):
        raise ValueError(f"Invalid task registry: {TASK_REGISTRY_PATH}")
    registry["version"] = max(2, int(registry.get("version", 1)))
    registry["dataset_root"] = DEFAULT_DATASET_DIR
    registry["naming_rule"] = {
        "multi_task": "lerobot_vN/multitask_<arm-name>",
        "single_task": "lerobot_vN/single_task/<task-slug>_<arm-name>",
    }
    registry["episode_label_rule"] = "episode_NNNNNN_<arm-name>_YYYYMMDD_HHMMSS"
    return registry


def _save_task_registry(registry):
    os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)
    temporary_path = f"{TASK_REGISTRY_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as destination:
        json.dump(registry, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    os.replace(temporary_path, TASK_REGISTRY_PATH)


def _task_slug(task):
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    return slug[:48] or "task"


def _multitask_dataset_path(arm_name):
    normalized_arm = str(arm_name).strip().lower()
    if not normalized_arm:
        raise ValueError("arm_name 不可為空")
    return os.path.join(
        DEFAULT_DATASET_DIR, "lerobot_v3", f"multitask_{normalized_arm}"
    )


def _normalize_dataset_mode(dataset_mode):
    normalized = str(dataset_mode or "multi_task").strip().lower().replace("-", "_")
    if normalized not in {"multi_task", "single_task"}:
        raise ValueError("dataset_mode 必須是 multi_task 或 single_task")
    return normalized


def _recording_dataset_path(task, arm_name, dataset_mode="multi_task"):
    normalized_arm = str(arm_name).strip().lower()
    mode = _normalize_dataset_mode(dataset_mode)
    if mode == "multi_task":
        folder = f"multitask_{normalized_arm}"
    else:
        folder = os.path.join("single_task", f"{_task_slug(str(task))}_{normalized_arm}")
    return os.path.join(DEFAULT_DATASET_DIR, "lerobot_v3", folder)


def _resolve_task_dataset(task, arm_name, output_path=None, dataset_mode="multi_task"):
    normalized_task = " ".join(task.split())
    normalized_arm = str(arm_name).strip().lower()
    requested_path = (
        _normalize_output_path(output_path)
        if output_path is not None
        else _recording_dataset_path(normalized_task, normalized_arm, dataset_mode)
    )
    with _task_registry_lock:
        registry = _load_task_registry()
        entry = next(
            (
                item for item in registry["tasks"]
                if str(item.get("task", "")).casefold() == normalized_task.casefold()
                and str(item.get("arm_name", "")).lower() == normalized_arm
                and (
                    os.path.abspath(item.get("dataset_path", ""))
                    == os.path.abspath(requested_path)
                )
            ),
            None,
        )
        if entry is None:
            same_task_ids = [
                str(item.get("task_id"))
                for item in registry["tasks"]
                if str(item.get("task", "")).casefold() == normalized_task.casefold()
                and re.fullmatch(r"task_\d+", str(item.get("task_id", "")))
            ]
            used_numbers = []
            for item in registry["tasks"]:
                match = re.fullmatch(r"task_(\d+)", str(item.get("task_id", "")))
                if match:
                    used_numbers.append(int(match.group(1)))
            task_id = (
                same_task_ids[0]
                if same_task_ids
                else f"task_{max(used_numbers, default=0) + 1:04d}"
            )
            folder_name = os.path.relpath(requested_path, DEFAULT_DATASET_DIR)
            dataset_path = requested_path
            now = _utc_now()
            entry = {
                "task_id": task_id,
                "task": normalized_task,
                "arm_name": normalized_arm,
                "dataset_folder": folder_name,
                "dataset_path": dataset_path,
                "dataset_mode": _normalize_dataset_mode(dataset_mode),
                "episode_count": 0,
                "episodes": [],
                "created_at": now,
                "updated_at": now,
            }
            registry["tasks"].append(entry)
            _save_task_registry(registry)
        return dict(entry), registry


def _episode_label(episode_index, arm_name, recorded_at):
    timestamp = datetime.fromisoformat(recorded_at).strftime("%Y%m%d_%H%M%S")
    return f"episode_{int(episode_index):06d}_{arm_name}_{timestamp}"


def _write_episode_labels(dataset_path, episodes):
    metadata_directory = os.path.join(dataset_path, "meta")
    os.makedirs(metadata_directory, exist_ok=True)
    path = os.path.join(metadata_directory, "episode_labels.json")
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as destination:
        json.dump({"version": 1, "episodes": episodes}, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    os.replace(temporary_path, path)
    return path


def _update_task_episode_count(task_id, arm_name, episode_count, episode_index=None):
    with _task_registry_lock:
        registry = _load_task_registry()
        for entry in registry["tasks"]:
            if (
                entry.get("task_id") == task_id
                and entry.get("arm_name") == arm_name
            ):
                recorded_at = datetime.now().astimezone().isoformat()
                entry["updated_at"] = recorded_at
                episodes = entry.setdefault("episodes", [])
                if episode_index is not None and not any(
                    int(item.get("episode_index", -1)) == int(episode_index)
                    for item in episodes
                ):
                    episodes.append({
                        "episode_index": int(episode_index),
                        "arm_name": arm_name,
                        "recorded_at": recorded_at,
                        "label": _episode_label(episode_index, arm_name, recorded_at),
                        "task": entry.get("task"),
                    })
                entry["episode_count"] = len(episodes)
                _save_task_registry(registry)
                dataset_episodes = []
                for task_entry in registry["tasks"]:
                    if os.path.abspath(task_entry.get("dataset_path", "")) == os.path.abspath(entry["dataset_path"]):
                        dataset_episodes.extend(task_entry.get("episodes", []))
                dataset_episodes.sort(key=lambda item: int(item.get("episode_index", -1)))
                _write_episode_labels(entry["dataset_path"], dataset_episodes)
                return dict(entry)
    raise ValueError(f"Unknown task dataset: task_id={task_id}, arm_name={arm_name}")


def get_task_registry():
    action = "get_task_registry"
    try:
        with _task_registry_lock:
            registry = _load_task_registry()
            if _reconcile_v3_task_registry(registry):
                _save_task_registry(registry)
        return success(
            MODULE,
            action,
            result=True,
            data={
                **registry,
                "registry_path": TASK_REGISTRY_PATH,
                "registry_relative_path": _relative_path(TASK_REGISTRY_PATH),
            },
        )
    except Exception as exc:
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def get_replay_catalog():
    action = "get_replay_catalog"
    try:
        with _task_registry_lock:
            registry = _load_task_registry()
            if _reconcile_v3_task_registry(registry):
                _save_task_registry(registry)
        registered = [
            {**entry, "legacy": False}
            for entry in registry.get("tasks", [])
        ] + [
            {**entry, "legacy": True}
            for entry in registry.get("legacy_datasets", [])
        ]
        filesystem_datasets = _filesystem_dataset_paths()
        live_arms = _recording_arm_inventory()
        path_entry_counts = {}
        for entry in registered:
            path = os.path.abspath(str(entry.get("dataset_path", "")))
            if path:
                path_entry_counts[path] = path_entry_counts.get(path, 0) + 1
        datasets = []
        catalog_keys = set()
        for entry in registered:
            registered_path = entry.get("dataset_path")
            canonical_path = (
                os.path.realpath(registered_path)
                if isinstance(registered_path, str) and registered_path
                else None
            )
            dataset_path = filesystem_datasets.get(canonical_path)
            info = _read_dataset_info(dataset_path) if dataset_path else None
            if info is None:
                continue
            total_episodes = int(info.get("total_episodes", 0))
            total_tasks = int(info.get("total_tasks", 0))
            codebase_version = info.get("codebase_version")
            is_v2 = codebase_version in {"v2.0", "v2.1"}
            raw_episode_details = entry.get("episodes", [])
            valid_episode_details = [
                item for item in raw_episode_details
                if 0 <= int(item.get("episode_index", -1)) < total_episodes
            ]
            native_task_episodes = []
            if not is_v2:
                native_episode_tasks = _read_v3_episode_tasks(dataset_path)
                expected_task = str(entry.get("task", "")).casefold()
                native_task_episodes = sorted(
                    episode_index
                    for episode_index, tasks in native_episode_tasks.items()
                    if 0 <= episode_index < total_episodes
                    and any(str(task).casefold() == expected_task for task in tasks)
                )
            # v2 datasets are one-task-per-directory, so the dataset metadata
            # is authoritative even when an older registry contains stale or
            # non-contiguous episode indices. Legacy/single-entry datasets can
            # likewise expose every real episode safely.
            dataset_path_key = os.path.abspath(dataset_path)
            if native_task_episodes:
                playable_episodes = native_task_episodes
            elif (
                (is_v2 and total_tasks <= 1)
                or entry.get("legacy")
                or (not raw_episode_details and path_entry_counts.get(dataset_path_key) == 1)
            ):
                playable_episodes = list(range(total_episodes))
            else:
                playable_episodes = [
                    int(item["episode_index"])
                    for item in valid_episode_details
                ]
            if not playable_episodes:
                continue
            catalog_key = (
                os.path.abspath(dataset_path),
                str(entry.get("task", "")).casefold(),
                str(entry.get("arm_name", "")).casefold(),
            )
            if catalog_key in catalog_keys:
                continue
            catalog_keys.add(catalog_key)
            robot_type = info.get("robot_type")
            compatible_arms = [
                item["arm_name"] for item in live_arms
                if item.get("driver") == robot_type
            ]
            datasets.append({
                **entry,
                "episode_details": valid_episode_details,
                "dataset_path": dataset_path,
                "relative_path": _relative_path(dataset_path),
                "robot_type": robot_type,
                "compatible_arms": compatible_arms,
                "fps": info.get("fps"),
                "codebase_version": codebase_version,
                "format": (
                    "lerobot_v2"
                    if is_v2
                    else "lerobot_v3"
                ),
                "total_episodes": len(playable_episodes),
                "dataset_total_episodes": total_episodes,
                "total_frames": info.get("total_frames"),
                "episodes": playable_episodes,
                "action_features": [
                    key for key in info.get("features", {})
                    if key == "action" or key.startswith("action_")
                ],
                "video_keys": [
                    key for key, feature in info.get("features", {}).items()
                    if feature.get("dtype") == "video"
                ],
            })
        return success(
            MODULE,
            action,
            result=True,
            data={"datasets": datasets},
        )
    except Exception as exc:
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def _normalize_output_path(
    output_path,
):
    if output_path is None:
        raise ValueError("output_path 必須由 task registry 解析")

    if (
        not isinstance(
            output_path,
            str,
        )
        or not output_path.strip()
    ):
        raise ValueError(
            "output_path 必須是非空字串"
        )

    output_path = (
        output_path.strip()
    )

    if output_path.lower().endswith(".json"):
        raise ValueError(
            "output_path 必須是 LeRobot dataset 資料夾，不可使用 .json"
        )

    if os.path.isabs(
        output_path
    ):
        return output_path

    return os.path.join(DEFAULT_DATASET_DIR, output_path)


def _prepare_video_cameras(arm_info, record_video):
    if not isinstance(record_video, bool):
        raise ValueError("record_video 必須是 bool")
    if not record_video:
        return {}

    arm_name = arm_info["arm_name"]
    camera_names = [item["camera_name"] for item in arm_info.get("cameras", [])]
    if not camera_names:
        raise ValueError(
            f"arm '{arm_name}' 沒有對應相機"
        )

    frames = {}
    for camera_name in camera_names:
        response = camera_service.start_camera(camera_name)
        _require_success(response, "camera_service.start_camera")
        frame = camera_service.get_frame(camera_name)
        color_image = frame.get("color_image")
        if color_image is None:
            raise RuntimeError(f"camera '{camera_name}' 沒有 color frame")
        frames[camera_name] = color_image
    return frames


def _stop_video_cameras(camera_names):
    stopped = []
    failures = []
    for camera_name in camera_names:
        response = camera_service.stop_camera(camera_name)
        try:
            _require_success(response, "camera_service.stop_camera")
            stopped.append(camera_name)
        except Exception as exc:
            failures.append(f"{camera_name}: {exc}")
    if failures:
        raise RuntimeError("停止錄製相機失敗：" + "; ".join(failures))
    return stopped


def _read_dataset_info(dataset_path):
    info_path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.isfile(info_path):
        return None
    with open(info_path, "r", encoding="utf-8") as source:
        return json.load(source)


def _filesystem_dataset_paths():
    """Return canonical dataset directories that physically exist under the root."""
    datasets = {}
    if not os.path.isdir(DEFAULT_DATASET_DIR):
        return datasets
    for directory, child_directories, filenames in os.walk(DEFAULT_DATASET_DIR):
        if os.path.basename(directory) == "meta" and "info.json" in filenames:
            dataset_path = os.path.dirname(directory)
            datasets[os.path.realpath(dataset_path)] = dataset_path
            child_directories[:] = []
    return datasets


def _read_v3_episode_tasks(dataset_path):
    """Read the authoritative episode/task mapping written by LeRobot v3."""
    episode_tasks = {}
    pattern = os.path.join(dataset_path, "meta", "episodes", "**", "*.parquet")
    for path in sorted(glob(pattern, recursive=True)):
        try:
            import pandas as pd
            frame = pd.read_parquet(path, columns=["episode_index", "tasks"])
        except (ImportError, OSError, ValueError, KeyError):
            continue
        for row in frame.to_dict("records"):
            episode_index = int(row["episode_index"])
            tasks = row.get("tasks")
            if tasks is None:
                normalized_tasks = []
            elif isinstance(tasks, str):
                normalized_tasks = [tasks]
            else:
                normalized_tasks = [str(task) for task in list(tasks)]
            episode_tasks[episode_index] = normalized_tasks
    return episode_tasks


def _reconcile_v3_task_registry(registry):
    """Repair registry entries from LeRobot's authoritative v3 metadata."""
    changed = False
    v3_root = os.path.join(DEFAULT_DATASET_DIR, "lerobot_v3")
    if not os.path.isdir(v3_root):
        return changed
    live_arms = _recording_arm_inventory()
    for dataset_path in sorted(_filesystem_dataset_paths().values()):
        if os.path.commonpath([v3_root, dataset_path]) != v3_root:
            continue
        folder = os.path.basename(dataset_path)
        info = _read_dataset_info(dataset_path)
        if not info or info.get("codebase_version") != "v3.0":
            continue
        robot_type = info.get("robot_type")
        compatible_arms = [
            item["arm_name"] for item in live_arms
            if item.get("driver") == robot_type
        ]
        arm_name = (
            folder.removeprefix("multitask_")
            if folder.startswith("multitask_")
            else (compatible_arms[0] if len(compatible_arms) == 1 else "")
        )
        if not any(item["arm_name"] == arm_name for item in live_arms):
            continue
        native_episode_tasks = _read_v3_episode_tasks(dataset_path)
        task_episodes = {}
        for episode_index, tasks in native_episode_tasks.items():
            for task in tasks:
                task_episodes.setdefault(str(task), []).append(int(episode_index))
        for task, episode_indices in task_episodes.items():
            entry = next((
                item for item in registry["tasks"]
                if str(item.get("task", "")).casefold() == task.casefold()
                and str(item.get("arm_name", "")).casefold() == arm_name.casefold()
                and os.path.abspath(item.get("dataset_path", "")) == os.path.abspath(dataset_path)
            ), None)
            if entry is None:
                same_task = next((
                    item for item in registry["tasks"]
                    if str(item.get("task", "")).casefold() == task.casefold()
                ), None)
                used_numbers = [
                    int(match.group(1))
                    for item in registry["tasks"]
                    if (match := re.fullmatch(r"task_(\d+)", str(item.get("task_id", ""))))
                ]
                now = _utc_now()
                entry = {
                    "task_id": same_task.get("task_id") if same_task else f"task_{max(used_numbers, default=0) + 1:04d}",
                    "task": task,
                    "arm_name": arm_name,
                    "dataset_folder": os.path.relpath(dataset_path, DEFAULT_DATASET_DIR),
                    "dataset_path": dataset_path,
                    "dataset_mode": (
                        "single_task"
                        if f"{os.sep}single_task{os.sep}" in dataset_path
                        else "multi_task"
                    ),
                    "episode_count": 0,
                    "episodes": [],
                    "created_at": now,
                    "updated_at": now,
                }
                registry["tasks"].append(entry)
                changed = True
            existing_by_index = {
                int(item.get("episode_index", -1)): item
                for item in entry.get("episodes", [])
            }
            repaired_episodes = []
            for episode_index in sorted(set(episode_indices)):
                item = existing_by_index.get(episode_index)
                if item is None:
                    recorded_at = datetime.now().astimezone().isoformat()
                    item = {
                        "episode_index": episode_index,
                        "arm_name": arm_name,
                        "recorded_at": recorded_at,
                        "label": _episode_label(episode_index, arm_name, recorded_at),
                        "task": task,
                    }
                repaired_episodes.append(item)
            if entry.get("episodes") != repaired_episodes or entry.get("episode_count") != len(repaired_episodes):
                entry["episodes"] = repaired_episodes
                entry["episode_count"] = len(repaired_episodes)
                entry["updated_at"] = datetime.now().astimezone().isoformat()
                changed = True
    return changed


def _dataset_repo_id(dataset_path):
    name = os.path.basename(os.path.normpath(dataset_path)) or "ur7e_dataset"
    safe_name = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in name
    )
    return f"local/{safe_name}"


def _lerobot_features(camera_frames):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_STATE_NAMES,
        },
        "observation.tcp_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_TCP_POSE_NAMES,
        },
        "action_tcp": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_TCP_ACTION_NAMES,
        },
        "action_tcp_base": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_TCP_BASE_ACTION_NAMES,
        },
        "action_joints": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_JOINT_ACTION_NAMES,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }
    for camera_name, image in camera_frames.items():
        height, width = image.shape[:2]
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _open_lerobot_dataset(dataset_path, fps, camera_frames, robot_type):
    os.environ.setdefault(
        "HF_DATASETS_CACHE",
        os.path.join(_project_root(), ".cache", "huggingface", "datasets"),
    )
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot dataset dependencies unavailable; "
            "run pip install -r requirements.txt"
        ) from exc

    info = _read_dataset_info(dataset_path)
    record_video = bool(camera_frames)
    repo_id = _dataset_repo_id(dataset_path)
    common = {
        "repo_id": repo_id,
        "root": dataset_path,
        "streaming_encoding": record_video,
        "encoder_queue_maxsize": 30,
        "video_backend": "pyav",
    }

    if info is not None:
        if info.get("codebase_version") != "v3.0":
            raise ValueError(
                "output_path 已包含非 LeRobot v3 資料；請使用新的資料集名稱"
            )
        if int(info.get("fps", fps)) != fps:
            raise ValueError("同一 LeRobot 資料集的 fps 必須一致")
        if info.get("robot_type") != robot_type:
            raise ValueError(
                "同一 LeRobot 資料集不可混用不同手臂類型："
                f"既有為 {info.get('robot_type')}，本次為 {robot_type}"
            )
        existing_features = info.get("features", {})
        existing_tcp_action_names = existing_features.get("action_tcp", {}).get("names")
        existing_tcp_base_action_names = existing_features.get(
            "action_tcp_base", {}
        ).get("names")
        existing_joint_action_names = existing_features.get("action_joints", {}).get("names")
        if (
            existing_tcp_action_names != LEROBOT_TCP_ACTION_NAMES
            or existing_tcp_base_action_names != LEROBOT_TCP_BASE_ACTION_NAMES
            or existing_joint_action_names != LEROBOT_JOINT_ACTION_NAMES
        ):
            raise ValueError(
                "既有資料集的 action schema 與目前版本不相容；"
                "請使用新的 output_path"
            )
        video_keys = [
            key for key, feature in info.get("features", {}).items()
            if feature.get("dtype") == "video"
        ]
        if bool(video_keys) != record_video:
            raise ValueError(
                "同一 LeRobot 資料集不可混用 record_video=true/false"
            )
        return LeRobotDataset.resume(**common)

    if os.path.exists(dataset_path):
        if os.listdir(dataset_path):
            raise ValueError("output_path 已存在且不是 LeRobot v3 資料集")
        os.rmdir(dataset_path)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    return LeRobotDataset.create(
        fps=fps,
        features=_lerobot_features(camera_frames),
        robot_type=robot_type,
        use_videos=record_video,
        **common,
    )


def _tcp_local_delta(current_pose, next_pose):
    import cv2

    current = np.asarray(current_pose, dtype=np.float64)
    following = np.asarray(next_pose, dtype=np.float64)
    current_rotation, _ = cv2.Rodrigues(current[3:6])
    next_rotation, _ = cv2.Rodrigues(following[3:6])

    # Express both translation and rotation in the current TCP/tool frame.
    # Deployment must map local translation back with R_current @ delta_local.
    base_translation = following[:3] - current[:3]
    local_translation = current_rotation.T @ base_translation
    relative_rotation = current_rotation.T @ next_rotation
    rotation_delta, _ = cv2.Rodrigues(relative_rotation)
    return np.concatenate([
        local_translation,
        rotation_delta.reshape(3),
    ]).astype(np.float32)


def _tcp_base_delta(current_pose, next_pose):
    import cv2

    current = np.asarray(current_pose, dtype=np.float64)
    following = np.asarray(next_pose, dtype=np.float64)
    current_rotation, _ = cv2.Rodrigues(current[3:6])
    next_rotation, _ = cv2.Rodrigues(following[3:6])

    # Express both translation and rotation in the robot base frame.
    base_translation = following[:3] - current[:3]
    relative_rotation = next_rotation @ current_rotation.T
    rotation_delta, _ = cv2.Rodrigues(relative_rotation)
    return np.concatenate([
        base_translation,
        rotation_delta.reshape(3),
    ]).astype(np.float32)


def _sample_to_lerobot_frame(sample, next_sample, task, done=False):
    state = np.asarray(sample["joints"] + [sample["gripper"]], dtype=np.float32)
    tcp_pose = np.asarray(sample["tcp_pose"] + [sample["gripper"]], dtype=np.float32)
    delta = _tcp_local_delta(sample["tcp_pose"], next_sample["tcp_pose"])
    action_tcp = np.concatenate([
        delta,
        np.asarray([next_sample["gripper"]], dtype=np.float32),
    ])
    base_delta = _tcp_base_delta(sample["tcp_pose"], next_sample["tcp_pose"])
    action_tcp_base = np.concatenate([
        base_delta,
        np.asarray([next_sample["gripper"]], dtype=np.float32),
    ])
    action_joints = np.asarray(
        next_sample["joints"] + [next_sample["gripper"]],
        dtype=np.float32,
    )
    frame = {
        "observation.state": state,
        "observation.tcp_pose": tcp_pose,
        "action_tcp": action_tcp,
        "action_tcp_base": action_tcp_base,
        "action_joints": action_joints,
        "next.done": np.atleast_1d(np.bool_(done)),
        "task": task,
    }
    for camera_name, image in sample["images"].items():
        # Camera drivers expose BGR; LeRobot policies expect RGB.
        frame[f"observation.images.{camera_name}"] = image[:, :, ::-1].copy()
    return frame


def _normalize_input_path(
    input_path,
):
    if (
        not isinstance(
            input_path,
            str,
        )
        or not input_path.strip()
    ):
        raise ValueError(
            "input_path 必須是非空字串"
        )

    input_path = (
        input_path.strip()
    )

    full_path = input_path if os.path.isabs(input_path) else os.path.join(
        DEFAULT_DATASET_DIR, input_path
    )

    if not os.path.isdir(full_path):
        raise FileNotFoundError(f"LeRobot dataset not found: {full_path}")

    if not os.path.isfile(os.path.join(full_path, "meta", "info.json")):
        raise ValueError(f"Invalid LeRobot v3 dataset: {full_path}")

    return full_path


def _require_success(
    response,
    action_name,
):
    if not isinstance(
        response,
        dict,
    ):
        raise RuntimeError(
            f"{action_name} 回傳格式錯誤"
        )

    if (
        response.get(
            "status"
        )
        != "success"
    ):
        raise RuntimeError(
            f"{action_name} failed: "
            f"{response.get('message', 'unknown error')}"
        )

    if (
        response.get(
            "result",
            True,
        )
        is False
    ):
        raise RuntimeError(
            f"{action_name} result=False"
        )

    return response


def _recording_arm_inventory():
    arm_response = arm_service.get_arm_status()
    gripper_response = gripper_service.get_gripper_status()
    _require_success(arm_response, "arm_service.get_arm_status")
    _require_success(gripper_response, "gripper_service.get_gripper_status")
    live_grippers = {
        item.get("gripper_name"): item
        for item in (gripper_response.get("data") or {}).get("grippers", [])
        if item.get("connected", True) is not False and not item.get("error")
    }
    return [
        {
            **arm,
            "grippers": (
                [{**live_grippers[arm["arm_name"]], "arm_name": arm["arm_name"]}]
                if arm["arm_name"] in live_grippers else []
            ),
            # A stopped camera may report disconnected. Resolve it by the
            # arm naming convention, then start it before requesting a frame.
            "cameras": [{
                "camera_name": arm["arm_name"],
                "arm_name": arm["arm_name"],
            }],
        }
        for arm in (arm_response.get("data") or {}).get("arms", [])
    ]


def _recording_arm_info(arm_name):
    normalized = str(arm_name).strip().lower()
    arm_response = arm_service.get_arm_status(normalized)
    gripper_response = gripper_service.get_gripper_status(normalized)
    _require_success(arm_response, "arm_service.get_arm_status")
    _require_success(gripper_response, "gripper_service.get_gripper_status")
    arm = next(iter((arm_response.get("data") or {}).get("arms", [])), None)
    grippers = [
        item for item in (gripper_response.get("data") or {}).get("grippers", [])
        if item.get("connected", True) is not False and not item.get("error")
    ]
    info = (
        {
            **arm,
            "grippers": grippers,
            "cameras": [{"camera_name": normalized, "arm_name": normalized}],
        }
        if arm is not None else None
    )
    if info is None:
        raise ValueError(f"arm '{normalized}' 未連線或不存在")
    return info


def _recording_gripper_info(arm_info, gripper_name):
    normalized = str(gripper_name).strip().lower()
    info = next((
        item for item in arm_info.get("grippers", [])
        if item.get("gripper_name") == normalized
    ), None)
    if info is None:
        raise ValueError(
            f"gripper '{normalized}' 未連線、不存在或不屬於 arm "
            f"'{arm_info.get('arm_name')}'"
        )
    return info


# ============================================================
# Arm / Gripper Sampling
# ============================================================

def _get_arm_sample(
    arm_name,
):
    response = (
        arm_service
        .get_arm_joints(
            arm_name
        )
    )

    _require_success(
        response,
        "arm_service.get_arm_joints",
    )

    data = (
        response.get(
            "data"
        )
        or {}
    )

    arms = (
        data.get(
            "arms"
        )
        or []
    )

    if not arms:
        raise RuntimeError(
            f"arm '{arm_name}' "
            f"joints unavailable"
        )

    arm_data = (
        arms[0]
    )

    joints = (
        arm_data.get(
            "joints"
        )
    )

    if joints is None:
        raise RuntimeError(
            f"arm '{arm_name}' "
            f"joints unavailable"
        )

    return {
        "arm_name":
            arm_data.get(
                "arm_name",
                arm_name,
            ),

        "driver":
            arm_data.get(
                "driver"
            ),

        "joint_names":
            list(
                DEFAULT_ARM_JOINT_NAMES
            ),

        "joints":
            [
                float(
                    value
                )
                for value
                in joints
            ],
    }


def _get_arm_pose_sample(arm_name):
    response = arm_service.get_arm_pose(arm_name)
    _require_success(response, "arm_service.get_arm_pose")
    arms = (response.get("data") or {}).get("arms") or []
    if not arms or arms[0].get("pose") is None:
        raise RuntimeError(f"arm '{arm_name}' TCP pose unavailable")
    pose = [float(value) for value in arms[0]["pose"]]
    if len(pose) != 6:
        raise RuntimeError(f"arm '{arm_name}' TCP pose 必須包含 6 個值")
    return pose


def _get_gripper_sample(
    gripper_name,
):
    status = (
        gripper_service
        .get_gripper_status_value(
            gripper_name
        )
    )

    if not isinstance(
        status,
        dict,
    ):
        raise RuntimeError(
            f"gripper '{gripper_name}' "
            f"status invalid"
        )

    if not status.get(
        "connected",
        False,
    ):
        raise RuntimeError(
            f"gripper '{gripper_name}' "
            f"not connected"
        )

    return {
        "gripper_name":
            gripper_name,

        "activated":
            bool(
                status.get(
                    "activated",
                    False,
                )
            ),

        "position":
            status.get(
                "position"
            ),

        "requested_position":
            status.get(
                "requested_position"
            ),

        "speed":
            status.get(
                "speed"
            ),

        "force":
            status.get(
                "force"
            ),

        "object_detected":
            bool(
                status.get(
                    "object_detected",
                    False,
                )
            ),

        "object_status":
            status.get(
                "object_status"
            ),

        "fault_code":
            status.get(
                "fault_code"
            ),
    }


def _get_record_sample(
    arm_name,
    gripper_name,
    include_gripper=True,
):
    arm = (
        _get_arm_sample(
            arm_name
        )
    )

    gripper = None

    if include_gripper:
        gripper = (
            _get_gripper_sample(
                gripper_name
            )
        )

    return {
        "arm":
            arm,

        "tcp_pose":
            _get_arm_pose_sample(
                arm_name
            ),

        "gripper":
            gripper,
    }


# ============================================================
# Recording Helpers
# ============================================================

def _is_recording():
    global _record_thread

    return (
        _record_thread
        is not None
        and _record_thread.is_alive()
    )


def _reset_record_state():
    global _record_thread
    global _record_start_time
    global _record_output_path
    global _record_interval
    global _record_gripper_events
    global _record_arm_name
    global _record_gripper_name
    global _record_freedrive
    global _record_camera_names
    global _record_task
    global _record_task_id
    global _record_dataset
    global _record_pending_sample
    global _record_saved_frame_count
    global _record_gripper_position
    global _record_frame_mapper

    with _record_lock:
        _record_thread = None
        _record_start_time = None
        _record_output_path = None
        _record_interval = DEFAULT_RECORD_INTERVAL
        _record_gripper_events = []
        _record_arm_name = None
        _record_gripper_name = None
        _record_freedrive = False
        _record_camera_names = []
        _record_task = None
        _record_task_id = None
        _record_dataset = None
        _record_pending_sample = None
        _record_saved_frame_count = 0
        _record_gripper_position = 0
        _record_frame_mapper = None
    _record_stop_event.clear()


def _get_record_elapsed():
    with _record_lock:

        if not _is_recording():
            raise RuntimeError(
                "目前沒有正在進行的記錄"
            )

        if _record_start_time is None:
            raise RuntimeError(
                "recording start time unavailable"
            )

        return (
            time.perf_counter()
            - _record_start_time
        )


def _append_gripper_event(
    action,
    position,
    speed=None,
    force=None,
    event_time=None,
):
    global _record_gripper_events
    global _record_gripper_position

    with _record_lock:

        if not _is_recording():
            raise RuntimeError(
                "目前沒有正在進行的記錄"
            )

        if _record_start_time is None:
            raise RuntimeError(
                "recording start time unavailable"
            )

        if event_time is None:
            event_time = (
                time.perf_counter()
                - _record_start_time
            )

        event = {
            "t":
                float(
                    event_time
                ),

            "action":
                str(
                    action
                ),

            "position":
                int(
                    position
                ),

            "speed":
                (
                    int(speed)
                    if speed is not None
                    else None
                ),

            "force":
                (
                    int(force)
                    if force is not None
                    else None
                ),
        }

        _record_gripper_events.append(
            event
        )

        _record_gripper_position = max(
            0.0,
            min(1.0, float(position) / 255.0),
        )

        return event

def _restore_recording_freedrive(
    arm_name,
    freedrive,
):
    """
    Gripper command 可能透過 URScript
    中斷目前 RTDE Freedrive。

    如果 recording 原本有開啟 freedrive，
    gripper command 完成後重新進入 freedrive。
    """

    if not freedrive:
        return False

    # 等待 gripper URScript command
    # 完成 controller 端切換。
    time.sleep(
        0.2
    )

    response = (
        arm_service
        .start_arm_freedrive(
            arm_name
        )
    )

    _require_success(
        response,
        "arm_service.start_arm_freedrive",
    )

    return True

# ============================================================
# Recording Loop
# ============================================================

def _record_loop():
    global _record_pending_sample
    global _record_saved_frame_count

    next_sample_time = time.perf_counter()
    while not _record_stop_event.is_set():
        with _record_lock:
            start_time = _record_start_time
            interval = _record_interval
            arm_name = _record_arm_name
            gripper_name = _record_gripper_name
            freedrive = _record_freedrive
            camera_names = list(_record_camera_names)
            dataset = _record_dataset
            task = _record_task

        if start_time is None or arm_name is None or gripper_name is None or dataset is None:
            break

        now = time.perf_counter()
        if now < next_sample_time and _record_stop_event.wait(next_sample_time - now):
            break

        try:
            sample = _get_record_sample(
                arm_name=arm_name,
                gripper_name=gripper_name,
                include_gripper=not freedrive,
            )
            gripper_data = sample["gripper"]
            if isinstance(gripper_data, dict):
                position = gripper_data.get("requested_position")
                if position is None:
                    position = gripper_data.get("position")
                gripper_position = (
                    max(0.0, min(1.0, float(position) / 255.0))
                    if position is not None
                    else _record_gripper_position
                )
            else:
                gripper_position = _record_gripper_position

            images = {}
            for camera_name in camera_names:
                frame_data = camera_service.get_frame(camera_name)
                color_image = frame_data.get("color_image")
                if color_image is None:
                    raise RuntimeError(f"camera '{camera_name}' color frame unavailable")
                images[camera_name] = color_image

            current_sample = {
                "joints": list(sample["arm"]["joints"]),
                "tcp_pose": list(sample["tcp_pose"]),
                "gripper": float(gripper_position),
                "images": images,
            }

            with _record_lock:
                previous_sample = _record_pending_sample
                if previous_sample is not None:
                    dataset.add_frame(
                        _record_frame_mapper(
                            previous_sample, current_sample, task, done=False
                        )
                    )
                    _record_saved_frame_count += 1
                _record_pending_sample = current_sample

        except Exception:
            logger.exception("failed to sample robot state during recording")

        next_sample_time += interval
        current = time.perf_counter()
        if next_sample_time < current:
            next_sample_time = current + interval


# ============================================================
# START RECORDING
# ============================================================

def start_robot_recording(
    arm_name,
    gripper_name=None,
    output_path=None,
    interval=DEFAULT_RECORD_INTERVAL,
    freedrive=False,
    record_video=False,
    task="robot demonstration",
    initial_gripper_position=0,
    dataset_mode="multi_task",
    _dataset_opener=None,
    _frame_mapper=None,
    _arm_info=None,
):
    startup_started_at = time.perf_counter()
    action = (
        "start_robot_recording"
    )

    freedrive_started = False
    dataset = None

    try:
        global _record_thread
        global _record_gripper_events
        global _record_start_time
        global _record_output_path
        global _record_interval
        global _record_arm_name
        global _record_gripper_name
        global _record_freedrive
        global _record_dataset
        global _record_pending_sample
        global _record_saved_frame_count
        global _record_gripper_position
        global _record_camera_names
        global _record_task
        global _record_task_id
        global _record_frame_mapper

        with _record_lock:

            if _is_recording():
                raise RuntimeError(
                    "recording already in progress"
                )

            if _is_playing():
                raise RuntimeError(
                    "playback is currently running"
                )

            # ==================================================
            # interval
            # ==================================================

            if interval is None:
                interval = (
                    DEFAULT_RECORD_INTERVAL
                )

            try:
                interval = (
                    float(
                        interval
                    )
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    "interval 必須是數值"
                ) from exc

            if interval <= 0:
                raise ValueError(
                    "interval 必須大於 0"
                )

            # ==================================================
            # freedrive
            # ==================================================

            if not isinstance(
                freedrive,
                bool,
            ):
                raise ValueError(
                    "freedrive 必須是 bool"
                )

            if not isinstance(task, str) or not task.strip():
                raise ValueError("task 必須是非空字串")
            task = task.strip()

            try:
                initial_gripper_position = int(initial_gripper_position)
            except (TypeError, ValueError) as exc:
                raise ValueError("initial_gripper_position 必須是 0~255 的整數") from exc
            if not 0 <= initial_gripper_position <= 255:
                raise ValueError("initial_gripper_position 必須介於 0~255")

            # ==================================================
            # names
            # ==================================================

            if arm_name is None:
                raise ValueError(
                    "arm_name 不可為空"
                )

            arm_name = (
                str(
                    arm_name
                )
                .strip()
                .lower()
            )

            if gripper_name is None:
                gripper_name = arm_name
            else:
                gripper_name = str(gripper_name).strip().lower()

            if not arm_name:
                raise ValueError(
                    "arm_name 不可為空"
                )

            if not gripper_name:
                raise ValueError(
                    "gripper_name 不可為空"
                )

            arm_info = _arm_info or _recording_arm_info(arm_name)
            _recording_gripper_info(arm_info, gripper_name)

            camera_frames = _prepare_video_cameras(arm_info, record_video)
            camera_names = list(camera_frames)

            # ==================================================
            # Hardware Validation
            # ==================================================

            # Freedrive 時只確認 arm。
            #
            # 不先 polling e-Series gripper，
            # 避免 30002 URScript 干擾 freedrive。
            _get_arm_sample(
                arm_name
            )
            _get_arm_pose_sample(
                arm_name
            )

            if not freedrive:
                _get_gripper_sample(
                    gripper_name
                )

            # ==================================================
            # Start Freedrive
            # ==================================================

            if freedrive:

                response = (
                    arm_service
                    .start_arm_freedrive(
                        arm_name
                    )
                )

                _require_success(
                    response,
                    "arm_service."
                    "start_arm_freedrive",
                )

                freedrive_started = (
                    True
                )

            # ==================================================
            # Output
            # ==================================================

            task_entry, _ = _resolve_task_dataset(
                task, arm_name, output_path, dataset_mode=dataset_mode
            )
            task_id = task_entry["task_id"]
            full_output_path = task_entry["dataset_path"]

            fps = max(1, int(round(1.0 / interval)))
            dataset_opener = _dataset_opener or _open_lerobot_dataset
            frame_mapper = _frame_mapper or _sample_to_lerobot_frame
            dataset = dataset_opener(
                full_output_path,
                fps,
                camera_frames,
                arm_info["driver"],
            )
            episode_index = int(dataset.meta.total_episodes)

            # ==================================================
            # State
            # ==================================================

            _record_output_path = (
                full_output_path
            )

            _record_interval = (
                interval
            )

            _record_arm_name = (
                arm_name
            )

            _record_gripper_name = (
                gripper_name
            )

            _record_freedrive = (
                freedrive
            )

            _record_gripper_events = []

            _record_camera_names = list(camera_names)

            _record_dataset = dataset

            _record_pending_sample = None

            _record_saved_frame_count = 0

            _record_gripper_position = initial_gripper_position / 255.0

            _record_task = task

            _record_task_id = task_id
            _record_frame_mapper = frame_mapper

            _record_start_time = (
                time.perf_counter()
            )

            _record_stop_event.clear()

            # ==================================================
            # Thread
            # ==================================================

            _record_thread = (
                threading.Thread(
                    target=
                        _record_loop,

                    name=
                        "robot-recording",

                    daemon=
                        True,
                )
            )

            _record_thread.start()

            return success(
                MODULE,
                action,
                result=True,
                data={
                    "arm_name":
                        arm_name,

                    "gripper_name":
                        gripper_name,

                    "output_path":
                        _relative_path(
                            full_output_path
                        ),

                    "absolute_path":
                        full_output_path,

                    "interval":
                        interval,

                    "freedrive":
                        freedrive,

                    "recording":
                        True,

                    "gripper_sampling":
                        not freedrive,

                    "gripper_event_recording":
                        True,

                    "gripper_event_count":
                        0,

                    "format":
                        "lerobot_v3",

                    "task":
                        task,

                    "task_id":
                        task_id,

                    "task_registry_path":
                        TASK_REGISTRY_PATH,

                    "record_video":
                        record_video,

                    "camera_names":
                        camera_names,

                    "episode_index":
                        episode_index,

                    "video_storage":
                        "lerobot_streaming" if record_video else None,

                    "codebase_version":
                        "v3.0",

                    "action_spaces":
                        dict(LEROBOT_ACTION_SPACES),

                    "initial_gripper_position":
                        initial_gripper_position,

                    "startup_seconds":
                        time.perf_counter() - startup_started_at,
                },
            )

    except Exception as exc:

        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                logger.exception("failed to finalize LeRobot dataset after start failure")

        # Freedrive 已成功但後續初始化失敗時 rollback。
        if freedrive_started:

            try:
                arm_service.stop_arm_freedrive(
                    arm_name
                )

            except Exception:

                logger.exception(
                    "failed to rollback freedrive"
                )

        logger.exception(
            "start_robot_recording failed"
        )

        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(
                    exc
                ).__name__,
        )


# ============================================================
# Recording Gripper Commands
# ============================================================

def move_recording_gripper(
    gripper_name,
    position,
    speed=None,
    force=None,
    wait=False,
    timeout=None,
):
    action = (
        "move_recording_gripper"
    )

    try:

        with _record_lock:

            if not _is_recording():
                raise RuntimeError(
                    "目前沒有正在進行的記錄"
                )

            arm_name = (
                _record_arm_name
            )

            gripper_name = (
                _record_gripper_name
            )

            freedrive = (
                _record_freedrive
            )

        if position is None:
            raise ValueError(
                "position 不可為空"
            )

        position = (
            int(
                position
            )
        )

        # ====================================================
        # Record Command Time
        # ====================================================

        event_time = (
            _get_record_elapsed()
        )

        # ====================================================
        # Execute Gripper
        # ====================================================

        response = (
            gripper_service
            .move_gripper(
                gripper_name=
                    gripper_name,

                position=
                    position,

                speed=
                    speed,

                force=
                    force,

                wait=
                    wait,

                timeout=
                    timeout,
            )
        )

        _require_success(
            response,
            "gripper_service.move_gripper",
        )

        # ====================================================
        # Record Event
        # ====================================================

        event = (
            _append_gripper_event(
                action=
                    "move",

                position=
                    position,

                speed=
                    speed,

                force=
                    force,

                event_time=
                    event_time,
            )
        )

        # ====================================================
        # Restore Freedrive
        # ====================================================

        freedrive_restored = (
            _restore_recording_freedrive(
                arm_name=
                    arm_name,

                freedrive=
                    freedrive,
            )
        )

        return success(
            MODULE,
            action,
            result=True,
            data={
                "arm_name":
                    arm_name,

                "gripper_name":
                    gripper_name,

                "position":
                    position,

                "recorded":
                    True,

                "freedrive_restored":
                    freedrive_restored,

                "event":
                    event,
            },
        )

    except Exception as exc:

        logger.exception(
            "move_recording_gripper failed"
        )

        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(
                    exc
                ).__name__,
        )


def open_recording_gripper(
    gripper_name,
    speed=None,
    force=None,
    wait=False,
    timeout=None,
):
    action = (
        "open_recording_gripper"
    )

    try:

        if gripper_name is None:
            raise ValueError(
                "gripper_name 不可為空"
            )

        gripper_name = (
            str(gripper_name)
            .strip()
            .lower()
        )

        if not gripper_name:
            raise ValueError(
                "gripper_name 不可為空"
            )

        with _record_lock:

            if not _is_recording():
                raise RuntimeError(
                    "目前沒有正在進行的記錄"
                )

            if gripper_name != _record_gripper_name:
                raise ValueError(
                    f"gripper_name 不符合目前 recording："
                    f"requested={gripper_name}, "
                    f"recording={_record_gripper_name}"
                )

            arm_name = _record_arm_name
            freedrive = _record_freedrive

        # ====================================================
        # Record Command Time
        # ====================================================

        event_time = (
            _get_record_elapsed()
        )

        # ====================================================
        # Execute Gripper
        # ====================================================

        response = (
            gripper_service
            .open_gripper(
                gripper_name=
                    gripper_name,

                speed=
                    speed,

                force=
                    force,

                wait=
                    wait,

                timeout=
                    timeout,
            )
        )

        _require_success(
            response,
            "gripper_service.open_gripper",
        )

        # ====================================================
        # Record Event
        # ====================================================

        event = (
            _append_gripper_event(
                action=
                    "open",

                position=
                    0,

                speed=
                    speed,

                force=
                    force,

                event_time=
                    event_time,
            )
        )

        # ====================================================
        # Restore Freedrive
        # ====================================================

        freedrive_restored = (
            _restore_recording_freedrive(
                arm_name=
                    arm_name,

                freedrive=
                    freedrive,
            )
        )

        response = success(
            MODULE,
            action,
            result=True,
            data={
                "arm_name":
                    arm_name,

                "gripper_name":
                    gripper_name,

                "position":
                    0,

                "recorded":
                    True,

                "freedrive_restored":
                    freedrive_restored,

                "event":
                    event,
            },
        )
        return response

    except Exception as exc:

        logger.exception(
            "open_recording_gripper failed"
        )

        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(
                    exc
                ).__name__,
        )

def close_recording_gripper(
    gripper_name,
    speed=None,
    force=None,
    wait=False,
    timeout=None,
):
    action = (
        "close_recording_gripper"
    )

    try:

        with _record_lock:

            if not _is_recording():
                raise RuntimeError(
                    "目前沒有正在進行的記錄"
                )

            arm_name = (
                _record_arm_name
            )

            gripper_name = (
                _record_gripper_name
            )

            freedrive = (
                _record_freedrive
            )

        # ====================================================
        # Record Command Time
        # ====================================================

        event_time = (
            _get_record_elapsed()
        )

        # ====================================================
        # Execute Gripper
        # ====================================================

        response = (
            gripper_service
            .close_gripper(
                gripper_name=
                    gripper_name,

                speed=
                    speed,

                force=
                    force,

                wait=
                    wait,

                timeout=
                    timeout,
            )
        )

        _require_success(
            response,
            "gripper_service.close_gripper",
        )

        # ====================================================
        # Record Event
        # ====================================================

        event = (
            _append_gripper_event(
                action=
                    "close",

                position=
                    255,

                speed=
                    speed,

                force=
                    force,

                event_time=
                    event_time,
            )
        )

        # ====================================================
        # Restore Freedrive
        # ====================================================

        freedrive_restored = (
            _restore_recording_freedrive(
                arm_name=
                    arm_name,

                freedrive=
                    freedrive,
            )
        )

        return success(
            MODULE,
            action,
            result=True,
            data={
                "arm_name":
                    arm_name,

                "gripper_name":
                    gripper_name,

                "position":
                    255,

                "recorded":
                    True,

                "freedrive_restored":
                    freedrive_restored,

                "event":
                    event,
            },
        )

    except Exception as exc:

        logger.exception(
            "close_recording_gripper failed"
        )

        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(
                    exc
                ).__name__,
        )


# ============================================================
# STOP RECORDING
# ============================================================
def stop_robot_recording(arm_name, gripper_name=None):
    action = "stop_robot_recording"
    should_cleanup = False
    stop_started_at = time.perf_counter()
    stop_warnings = []

    try:
        global _record_thread
        global _record_start_time
        global _record_output_path
        global _record_interval
        global _record_gripper_events
        global _record_arm_name
        global _record_gripper_name
        global _record_freedrive
        global _record_camera_names
        global _record_task
        global _record_task_id
        global _record_dataset
        global _record_pending_sample
        global _record_saved_frame_count
        global _record_gripper_position

        if arm_name is None:
            raise ValueError("arm_name 不可為空")
        arm_name = str(arm_name).strip().lower()
        gripper_name = (
            arm_name if gripper_name is None
            else str(gripper_name).strip().lower()
        )

        with _record_lock:
            if not _is_recording():
                raise RuntimeError("目前沒有正在進行的記錄")
            if arm_name != _record_arm_name or gripper_name != _record_gripper_name:
                raise ValueError("arm_name 或 gripper_name 不符合目前 recording")
            thread = _record_thread
            output_path = _record_output_path
            freedrive = _record_freedrive
            start_time = _record_start_time
            camera_names = list(_record_camera_names)
            task_id = _record_task_id

        should_cleanup = True
        _record_stop_event.set()
        wall_recording_seconds = max(0.0, stop_started_at - start_time)
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("錄製執行緒無法在 10 秒內停止")
        thread_stop_seconds = time.perf_counter() - stop_started_at

        stopped_camera_names = _stop_video_cameras(camera_names)

        if freedrive:
            response = arm_service.stop_arm_freedrive(arm_name)
            if (
                response.get("status") != "success"
                or response.get("result", True) is False
            ):
                first_error = response.get("message", "unknown error")
                reconnect_response = arm_service.reconnect_arm(arm_name=arm_name)
                if (
                    reconnect_response.get("status") == "success"
                    and reconnect_response.get("result", True) is not False
                ):
                    response = arm_service.stop_arm_freedrive(arm_name)
                if (
                    response.get("status") != "success"
                    or response.get("result", True) is False
                ):
                    safety_response = arm_service.stop_arm(arm_name=arm_name)
                    warning = (
                        "freedrive 停止命令失敗，但錄製資料仍繼續保存："
                        f"{first_error}; retry={response.get('message', 'unknown error')}; "
                        f"safety_stop={safety_response.get('status', 'unknown')}"
                    )
                    logger.warning(warning)
                    stop_warnings.append(warning)

        with _record_lock:
            dataset = _record_dataset
            pending_sample = _record_pending_sample
            episode_index = int(dataset.meta.total_episodes) if dataset else None
            if dataset is None or pending_sample is None:
                raise RuntimeError("沒有足夠的 sample 可建立 LeRobot episode")
            # The final observation has no future command; use a zero TCP delta
            # and preserve its current gripper state.
            dataset.add_frame(
                _record_frame_mapper(
                    pending_sample, pending_sample, _record_task, done=True
                )
            )
            _record_saved_frame_count += 1
            sample_count = _record_saved_frame_count

        save_started_at = time.perf_counter()
        try:
            dataset.save_episode()
        finally:
            dataset.finalize()
        save_seconds = time.perf_counter() - save_started_at

        info = _read_dataset_info(output_path) or {}
        fps = max(1, int(info.get("fps", round(1.0 / _record_interval))))
        # LeRobot timestamps are frame-based (0, 1/fps, ...).  Derive the
        # episode duration from the saved timeline.  The previous wall-clock
        # calculation happened after camera shutdown, video finalization and
        # parquet writes, so a 5-second demonstration could be reported as
        # 18+ seconds even though its dataset timestamps were correct.
        duration_seconds = max(0.0, (sample_count - 1) / float(fps))
        task_entry = _update_task_episode_count(
            task_id,
            arm_name,
            info.get("total_episodes", 0),
            episode_index=episode_index,
        )
        video_keys = [
            key for key, feature in info.get("features", {}).items()
            if feature.get("dtype") == "video"
        ]

        response = success(
            MODULE,
            action,
            result=True,
            data={
                "arm_name": arm_name,
                "gripper_name": gripper_name,
                "output_path": _relative_path(output_path),
                "absolute_path": output_path,
                "task_id": task_id,
                "task": task_entry["task"],
                "task_registry_path": TASK_REGISTRY_PATH,
                "episode_label": next(
                    (
                        item.get("label")
                        for item in task_entry.get("episodes", [])
                        if item.get("episode_index") == episode_index
                    ),
                    None,
                ),
                "sample_count": sample_count,
                "duration_seconds": duration_seconds,
                "wall_recording_seconds": wall_recording_seconds,
                "freedrive": freedrive,
                "format": "lerobot_v3",
                "codebase_version": info.get("codebase_version"),
                "episode_index": episode_index,
                "fps": info.get("fps"),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "video_keys": video_keys,
                "stopped_camera_names": stopped_camera_names,
                "cameras_stopped": len(stopped_camera_names) == len(camera_names),
                "stop_timings": {
                    "thread_stop_seconds": thread_stop_seconds,
                    "save_finalize_seconds": save_seconds,
                    "total_seconds": time.perf_counter() - stop_started_at,
                },
                "warnings": stop_warnings,
                "action_spaces": dict(LEROBOT_ACTION_SPACES),
            },
        )
        _reset_record_state()
        return response

    except Exception as exc:
        logger.exception("stop_robot_recording failed")
        if should_cleanup:
            _reset_record_state()
        return error(
            MODULE,
            action,
            error=exc,
            error_type=type(exc).__name__,
        )


# ============================================================
# GET RECORDING STATUS
# ============================================================

def get_robot_recording_status():
    action = (
        "get_robot_recording_status"
    )

    try:

        with _record_lock:

            recording = (
                _is_recording()
            )

            elapsed = (
                time.perf_counter()
                - _record_start_time

                if (
                    recording
                    and _record_start_time
                    is not None
                )

                else 0.0
            )

            return success(
                MODULE,
                action,
                result=True,
                data={
                    "recording":
                        recording,

                    "arm_name":
                        _record_arm_name,

                    "gripper_name":
                        _record_gripper_name,

                    "output_path":
                        _relative_path(
                            _record_output_path
                        ),

                    "interval":
                        _record_interval,

                    "freedrive":
                        _record_freedrive,

                    "gripper_sampling":
                        (
                            not _record_freedrive
                            if recording
                            else False
                        ),

                    "gripper_event_recording":
                        True,

                    "sample_count":
                        _record_saved_frame_count
                        + (1 if _record_pending_sample is not None else 0),

                    "gripper_event_count":
                        len(
                            _record_gripper_events
                        ),

                    "elapsed_seconds":
                        elapsed,

                    "format":
                        "lerobot_v3",

                    "codebase_version":
                        "v3.0",

                    "episode_index":
                        (
                            int(_record_dataset.meta.total_episodes)
                            if _record_dataset is not None
                            else None
                        ),

                    "action_spaces":
                        dict(LEROBOT_ACTION_SPACES),

                    "task":
                        _record_task,

                    "task_id":
                        _record_task_id,

                    "task_registry_path":
                        TASK_REGISTRY_PATH,

                    "record_video":
                        bool(_record_camera_names),

                    "camera_names":
                        list(_record_camera_names),

                    "video_storage":
                        "lerobot_streaming" if _record_camera_names else None,

                    "video_frame_count":
                        (
                            _record_saved_frame_count
                            if _record_camera_names
                            else 0
                        ),
                },
            )

    except Exception as exc:

        logger.exception(
            "get_robot_recording_status failed"
        )

        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(
                    exc
                ).__name__,
        )


# ============================================================
# LeRobot v3 Episode Load
# ============================================================

def _load_lerobot_episode(input_path, episode_index=0):
    os.environ.setdefault(
        "HF_DATASETS_CACHE",
        os.path.join(_project_root(), ".cache", "huggingface", "datasets"),
    )
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot dataset dependencies unavailable") from exc

    dataset_path = _normalize_input_path(input_path)
    try:
        episode_index = int(episode_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("episode_index 必須是整數") from exc
    if episode_index < 0:
        raise ValueError("episode_index 不可小於 0")

    dataset = LeRobotDataset(
        repo_id=_dataset_repo_id(dataset_path),
        root=dataset_path,
        episodes=[episode_index],
        video_backend="pyav",
    )
    if len(dataset) == 0:
        raise ValueError(f"LeRobot episode {episode_index} 沒有 frame")

    raw = dataset.hf_dataset.select_columns([
        "observation.state", "timestamp", "frame_index"
    ])
    rows = sorted(raw, key=lambda row: int(row["frame_index"]))
    states = [list(map(float, row["observation.state"])) for row in rows]
    if any(len(state) != 7 for state in states):
        raise ValueError("observation.state 必須包含 6 joints + gripper")

    joint_trajectory = [state[:6] for state in states]
    gripper_events = []
    previous_position = None
    for row, state in zip(rows, states):
        position = int(round(max(0.0, min(1.0, state[6])) * 255))
        if previous_position is None or position != previous_position:
            gripper_events.append({"t": float(row["timestamp"]), "position": position})
            previous_position = position

    return dataset_path, 1.0 / dataset.fps, joint_trajectory, gripper_events


# ============================================================
# Playback
# ============================================================

def _is_playing():
    global _playback_thread

    return (
        _playback_thread
        is not None
        and _playback_thread.is_alive()
    )


def _play_gripper_events(
    gripper_name,
    events,
    start_time,
):
    for event in events:

        if _playback_stop_event.is_set():
            return

        target_time = (
            start_time
            + float(
                event[
                    "t"
                ]
            )
        )

        while True:

            if _playback_stop_event.is_set():
                return

            remaining = (
                target_time
                - time.perf_counter()
            )

            if remaining <= 0:
                break

            _playback_stop_event.wait(
                min(
                    remaining,
                    0.02,
                )
            )

        response = (
            gripper_service
            .move_gripper(
                gripper_name=
                    gripper_name,

                position=
                    event[
                        "position"
                    ],

                speed=
                    event.get(
                        "speed"
                    ),

                force=
                    event.get(
                        "force"
                    ),

                wait=
                    False,
            )
        )

        _require_success(
            response,
            "gripper_service.move_gripper",
        )


def _playback_worker(
    input_path,
    episode_index,
    arm_name,
    gripper_name,
    speed,
    acceleration,
    lookahead_time,
    gain,
    move_to_start,
    move_to_start_speed,
    move_to_start_acceleration,
    episode_loader,
):
    global _playback_thread
    global _playback_input_path
    global _playback_arm_name
    global _playback_gripper_name
    global _playback_start_time
    global _playback_phase
    global _playback_error
    global _playback_completed

    try:

        with _playback_lock:
            _playback_phase = "loading"

        _, recorded_interval, joint_trajectory, gripper_events = (
            episode_loader(input_path, episode_index)
        )

        if not joint_trajectory:

            raise RuntimeError(
                "沒有可播放的 arm trajectory"
            )

        if _playback_stop_event.is_set():
            return

        # t=0 的夾爪狀態是 episode 初始條件，必須在手臂 ServoJ
        # 開始前設定。特別是 robotiq_eseries 會透過 port 30002 傳送
        # URScript；若與 ServoJ 同時執行，該 script 會取代 RTDE
        # control script，導致後續 servoJ 連續回傳 False。
        initial_gripper_events = [
            event for event in gripper_events
            if float(event.get("t", 0.0)) <= 0.0
        ]
        gripper_events = [
            event for event in gripper_events
            if float(event.get("t", 0.0)) > 0.0
        ]
        if initial_gripper_events:
            initial_event = initial_gripper_events[-1]
            response = gripper_service.move_gripper(
                gripper_name=gripper_name,
                position=initial_event["position"],
                speed=initial_event.get("speed"),
                force=initial_event.get("force"),
                wait=False,
            )
            _require_success(
                response,
                "gripper_service.move_gripper(initial)",
            )
            if _recording_gripper_info(
                _recording_arm_info(arm_name), gripper_name
            ).get("driver") == "robotiq_eseries":
                if not _settle_robotiq_eseries_script(arm_name):
                    return
                response = arm_service.reconnect_arm(arm_name=arm_name)
                _require_success(
                    response,
                    "arm_service.reconnect_arm(after initial gripper)",
                )

        if _playback_stop_event.is_set():
            return

        # ======================================================
        # Move Arm To Start
        # ======================================================
        #
        # 先移動到 trajectory 第一點，再開始 timeline。
        #
        # 避免 move_to_start 的時間被算進 playback，
        # 導致 gripper event 提前執行。
        # ======================================================

        if move_to_start:

            with _playback_lock:
                _playback_phase = "moving_to_start"

            response = (
                arm_service
                .move_arm_joints(
                    arm_name=
                        arm_name,

                    joints=
                        joint_trajectory[
                            0
                        ],

                    speed=
                        move_to_start_speed,

                    acceleration=
                        move_to_start_acceleration,

                    wait=
                        True,
                )
            )

            _require_success(
                response,
                "arm_service.move_arm_joints",
            )

        if _playback_stop_event.is_set():
            return

        # robotiq_eseries 透過 port 30002 執行 URScript，無法和 RTDE
        # ServoJ control script 並行。依夾爪事件切開 arm trajectory：
        # 每段 ServoJ 正常結束後才送夾爪命令，下一段會由 arm driver
        # 自動確認並恢復 RTDE control script。
        if (
            _recording_gripper_info(
                _recording_arm_info(arm_name), gripper_name
            ).get("driver") == "robotiq_eseries"
            and gripper_events
        ):
            start_time = time.perf_counter()
            with _playback_lock:
                _playback_start_time = start_time
                _playback_phase = "playing"

            segment_start = 0
            for event in gripper_events:
                if _playback_stop_event.is_set():
                    return
                event_index = int(round(float(event["t"]) / recorded_interval))
                event_index = max(
                    segment_start + 1,
                    min(event_index, len(joint_trajectory) - 1),
                )
                segment = joint_trajectory[segment_start:event_index + 1]
                response = arm_service.move_arm_joint_trajectory(
                    arm_name=arm_name,
                    joint_trajectory=segment,
                    dt=recorded_interval,
                    speed=speed,
                    acceleration=acceleration,
                    lookahead_time=lookahead_time,
                    gain=gain,
                    wait=True,
                    move_to_start=False,
                    move_to_start_speed=None,
                    move_to_start_acceleration=None,
                )
                _require_success(
                    response,
                    "arm_service.move_arm_joint_trajectory(segment)",
                )
                if _playback_stop_event.is_set():
                    return
                response = gripper_service.move_gripper(
                    gripper_name=gripper_name,
                    position=event["position"],
                    speed=event.get("speed"),
                    force=event.get("force"),
                    wait=False,
                )
                _require_success(
                    response,
                    "gripper_service.move_gripper(segment boundary)",
                )
                if not _settle_robotiq_eseries_script(arm_name):
                    return
                response = arm_service.reconnect_arm(arm_name=arm_name)
                _require_success(
                    response,
                    "arm_service.reconnect_arm(after gripper boundary)",
                )
                segment_start = event_index

            if segment_start < len(joint_trajectory) - 1:
                response = arm_service.move_arm_joint_trajectory(
                    arm_name=arm_name,
                    joint_trajectory=joint_trajectory[segment_start:],
                    dt=recorded_interval,
                    speed=speed,
                    acceleration=acceleration,
                    lookahead_time=lookahead_time,
                    gain=gain,
                    wait=True,
                    move_to_start=False,
                    move_to_start_speed=None,
                    move_to_start_acceleration=None,
                )
                _require_success(
                    response,
                    "arm_service.move_arm_joint_trajectory(final segment)",
                )

            with _playback_lock:
                _playback_completed = True
                _playback_phase = "completed"
            return

        # ======================================================
        # Arm Result
        # ======================================================

        arm_result = {
            "response":
                None,

            "exception":
                None,
        }

        def _run_arm():

            try:

                arm_result[
                    "response"
                ] = (
                    arm_service
                    .move_arm_joint_trajectory(
                        arm_name=
                            arm_name,

                        joint_trajectory=
                            joint_trajectory,

                        dt=
                            recorded_interval,

                        speed=
                            speed,

                        acceleration=
                            acceleration,

                        lookahead_time=
                            lookahead_time,

                        gain=
                            gain,

                        wait=
                            True,

                        # 已經在上面移動到第一點。
                        move_to_start=
                            False,

                        move_to_start_speed=
                            None,

                        move_to_start_acceleration=
                            None,
                    )
                )

            except Exception as exc:

                arm_result[
                    "exception"
                ] = (
                    exc
                )

        arm_thread = (
            threading.Thread(
                target=
                    _run_arm,

                name=
                    "robot-playback-arm",

                daemon=
                    True,
            )
        )

        # ======================================================
        # Start Playback Timeline
        # ======================================================

        start_time = (
            time.perf_counter()
        )

        with _playback_lock:

            _playback_start_time = (
                start_time
            )
            _playback_phase = "playing"

        arm_thread.start()

        # ======================================================
        # Gripper Timeline
        # ======================================================

        _play_gripper_events(
            gripper_name=
                gripper_name,

            events=
                gripper_events,

            start_time=
                start_time,
        )

        # ======================================================
        # Wait Arm
        # ======================================================

        arm_thread.join()

        if (
            arm_result[
                "exception"
            ]
            is not None
        ):
            raise arm_result[
                "exception"
            ]

        _require_success(
            arm_result[
                "response"
            ],
            "arm_service."
            "move_arm_joint_trajectory",
        )

        with _playback_lock:
            _playback_completed = True
            _playback_phase = "completed"

    except Exception as exc:

        with _playback_lock:
            _playback_error = f"{type(exc).__name__}: {exc}"
            _playback_phase = "failed"

        logger.exception(
            "robot trajectory playback failed"
        )

    finally:

        with _playback_lock:

            _playback_thread = None

            _playback_input_path = None

            _playback_arm_name = None

            _playback_gripper_name = None

            _playback_start_time = None

            if _playback_stop_event.is_set() and _playback_error is None:
                _playback_phase = "stopped"

        _playback_stop_event.clear()


# ============================================================
# START PLAYBACK
# ============================================================

def start_robot_playback(
    input_path,
    arm_name,
    gripper_name,
    episode_index=0,
    speed=None,
    acceleration=None,
    lookahead_time=0.1,
    gain=300,
    move_to_start=True,
    move_to_start_speed=None,
    move_to_start_acceleration=None,
    _episode_loader=None,
):

    action = (
        "start_robot_playback"
    )

    try:
        global _playback_thread
        global _playback_input_path
        global _playback_arm_name
        global _playback_gripper_name
        global _playback_start_time
        global _playback_phase
        global _playback_error
        global _playback_completed

        with _playback_lock:

            if _is_playing():

                raise RuntimeError(
                    "playback already in progress"
                )

            if _is_recording():

                raise RuntimeError(
                    "recording is currently running"
                )

            episode_loader = _episode_loader or _load_lerobot_episode
            full_path, _, _, _ = episode_loader(
                input_path,
                episode_index,
            )

            # ==================================================
            # Resolve Arm / Gripper
            # ==================================================
            if arm_name is None:
                raise ValueError(
                    "arm_name 不可為空"
                )

            if gripper_name is None:
                raise ValueError(
                    "gripper_name 不可為空"
                )

            arm_name = (
                str(arm_name)
                .strip()
                .lower()
            )

            gripper_name = (
                str(gripper_name)
                .strip()
                .lower()
            )

            if not arm_name:
                raise ValueError(
                    "arm_name 不可為空"
                )

            if not gripper_name:
                raise ValueError(
                    "gripper_name 不可為空"
                )

            arm_name = (
                str(
                    arm_name
                )
                .strip()
                .lower()
            )

            gripper_name = (
                str(
                    gripper_name
                )
                .strip()
                .lower()
            )

            arm_info = _recording_arm_info(arm_name)
            _recording_gripper_info(arm_info, gripper_name)
            info = _read_dataset_info(full_path) or {}
            robot_type = info.get("robot_type")
            configured_driver = arm_info.get("driver")
            if robot_type != configured_driver:
                raise ValueError(
                    f"dataset robot_type='{robot_type}' 不可播放到 "
                    f"arm '{arm_name}'（driver='{configured_driver}'）"
                )

            # ==================================================
            # Hardware Validation
            # ==================================================

            # Playback may be started before any other arm endpoint has
            # established the RTDE receive connection.  Connect explicitly
            # before requiring a live joint sample; otherwise a valid arm can
            # be rejected with "joints unavailable" even though reconnecting
            # it would make the state immediately available.
            response = arm_service.reconnect_arm(
                arm_name=arm_name,
            )
            _require_success(
                response,
                "arm_service.reconnect_arm(before playback)",
            )

            _get_arm_sample(
                arm_name
            )

            # Playback 前允許確認 gripper。
            _get_gripper_sample(
                gripper_name
            )

            # ==================================================
            # State
            # ==================================================

            _playback_input_path = (
                full_path
            )

            _playback_arm_name = (
                arm_name
            )

            _playback_gripper_name = (
                gripper_name
            )

            _playback_start_time = None
            _playback_phase = "starting"
            _playback_error = None
            _playback_completed = False

            _playback_stop_event.clear()

            # ==================================================
            # Thread
            # ==================================================

            _playback_thread = (
                threading.Thread(
                    target=
                        _playback_worker,

                    kwargs={
                        "input_path":
                            full_path,

                        "episode_index":
                            episode_index,

                        "arm_name":
                            arm_name,

                        "gripper_name":
                            gripper_name,

                        "speed":
                            speed,

                        "acceleration":
                            acceleration,

                        "lookahead_time":
                            lookahead_time,

                        "gain":
                            gain,

                        "move_to_start":
                            move_to_start,

                        "move_to_start_speed":
                            move_to_start_speed,

                        "move_to_start_acceleration":
                            move_to_start_acceleration,

                        "episode_loader":
                            episode_loader,
                    },

                    name=
                        "robot-playback",

                    daemon=
                        True,
                )
            )

            _playback_thread.start()

            return success(
                MODULE,
                action,
                result=True,
                data={
                    "input_path":
                        _relative_path(
                            full_path
                        ),

                    "absolute_path":
                        full_path,

                    "arm_name":
                        arm_name,

                    "gripper_name":
                        gripper_name,

                    "playing":
                        True,

                    "format":
                        "lerobot_v3",

                    "episode_index":
                        int(episode_index),
                },
            )

    except Exception as exc:

        logger.exception(
            "start_robot_playback failed"
        )

        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(
                    exc
                ).__name__,
        )


# ============================================================
# STOP PLAYBACK
# ============================================================
def stop_robot_playback(
    arm_name,
    gripper_name,
):
    action = "stop_robot_playback"

    try:

        if arm_name is None:
            raise ValueError(
                "arm_name 不可為空"
            )

        if gripper_name is None:
            raise ValueError(
                "gripper_name 不可為空"
            )

        arm_name = (
            str(arm_name)
            .strip()
            .lower()
        )

        gripper_name = (
            str(gripper_name)
            .strip()
            .lower()
        )

        with _playback_lock:

            if not _is_playing():
                raise RuntimeError(
                    "目前沒有正在播放的 trajectory"
                )

            if arm_name != _playback_arm_name:
                raise ValueError(
                    f"arm_name 不符合目前 playback："
                    f"requested={arm_name}, "
                    f"playing={_playback_arm_name}"
                )

            if gripper_name != _playback_gripper_name:
                raise ValueError(
                    f"gripper_name 不符合目前 playback："
                    f"requested={gripper_name}, "
                    f"playing={_playback_gripper_name}"
                )

            thread = _playback_thread

        _playback_stop_event.set()

        try:
            arm_service.stop_arm(
                arm_name
            )
        except Exception:
            logger.exception(
                "failed to stop arm "
                "during playback stop"
            )

        try:
            gripper_service.stop_gripper(
                gripper_name
            )
        except Exception:
            logger.exception(
                "failed to stop gripper "
                "during playback stop"
            )

        thread.join(
            timeout=10
        )

        return success(
            MODULE,
            action,
            result=True,
            data={
                "playing": False,
                "arm_name": arm_name,
                "gripper_name": gripper_name,
            },
        )

    except Exception as exc:

        logger.exception(
            "stop_robot_playback failed"
        )

        return error(
            MODULE,
            action,
            error=exc,
            error_type=type(exc).__name__,
        )

# ============================================================
# GET PLAYBACK STATUS
# ============================================================

def get_robot_playback_status():
    action = (
        "get_robot_playback_status"
    )

    try:

        global _playback_phase
        global _playback_error
        global _playback_completed

        with _playback_lock:

            playing = (
                _is_playing()
            )

            elapsed = (
                time.perf_counter()
                - _playback_start_time

                if (
                    playing
                    and _playback_start_time
                    is not None
                )

                else 0.0
            )

            return success(
                MODULE,
                action,
                result=True,
                data={
                    "playing":
                        playing,

                    "input_path":
                        _relative_path(
                            _playback_input_path
                        ),

                    "arm_name":
                        _playback_arm_name,

                    "gripper_name":
                        _playback_gripper_name,

                    "elapsed_seconds":
                        elapsed,

                    "phase":
                        _playback_phase,

                    "completed":
                        _playback_completed,

                    "error":
                        _playback_error,
                },
            )

    except Exception as exc:

        logger.exception(
            "get_robot_playback_status failed"
        )

        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(
                    exc
                ).__name__,
        )
