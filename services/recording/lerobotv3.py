import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

import numpy as np

import config
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


# ============================================================
# Common Helpers
# ============================================================

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
        "version": 1,
        "dataset_root": DEFAULT_DATASET_DIR,
        "naming_rule": "task_NNNN_<task-slug>_<arm-name>",
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
    registry.setdefault("version", 1)
    registry["dataset_root"] = DEFAULT_DATASET_DIR
    registry["naming_rule"] = "task_NNNN_<task-slug>_<arm-name>"
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


def _resolve_task_dataset(task, arm_name, output_path=None):
    normalized_task = " ".join(task.split())
    normalized_arm = str(arm_name).strip().lower()
    requested_path = (
        _normalize_output_path(output_path)
        if output_path is not None
        else None
    )
    with _task_registry_lock:
        registry = _load_task_registry()
        entry = next(
            (
                item for item in registry["tasks"]
                if str(item.get("task", "")).casefold() == normalized_task.casefold()
                and str(item.get("arm_name", "")).lower() == normalized_arm
                and (
                    requested_path is None
                    or os.path.abspath(item.get("dataset_path", ""))
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
            folder_name = f"{task_id}_{_task_slug(normalized_task)}_{normalized_arm}"
            dataset_path = os.path.join(DEFAULT_DATASET_DIR, folder_name)
            if requested_path is not None:
                folder_name = os.path.basename(os.path.normpath(requested_path))
                dataset_path = requested_path
            now = _utc_now()
            entry = {
                "task_id": task_id,
                "task": normalized_task,
                "arm_name": normalized_arm,
                "dataset_folder": folder_name,
                "dataset_path": dataset_path,
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
                entry["episode_count"] = int(episode_count)
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
                    })
                _save_task_registry(registry)
                _write_episode_labels(entry["dataset_path"], episodes)
                return dict(entry)
    raise ValueError(f"Unknown task dataset: task_id={task_id}, arm_name={arm_name}")


def get_task_registry():
    action = "get_task_registry"
    try:
        with _task_registry_lock:
            registry = _load_task_registry()
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
        registered = [
            {**entry, "legacy": False}
            for entry in registry.get("tasks", [])
        ] + [
            {**entry, "legacy": True}
            for entry in registry.get("legacy_datasets", [])
        ]
        datasets = []
        for entry in registered:
            dataset_path = entry.get("dataset_path")
            info = _read_dataset_info(dataset_path) if dataset_path else None
            if info is None:
                continue
            total_episodes = int(info.get("total_episodes", 0))
            robot_type = info.get("robot_type")
            compatible_arms = [
                arm_name
                for arm_name, arm_config in config.ARMS.items()
                if arm_config.get("driver") == robot_type
            ]
            datasets.append({
                **entry,
                "episode_details": entry.get("episodes", []),
                "dataset_path": dataset_path,
                "relative_path": _relative_path(dataset_path),
                "robot_type": robot_type,
                "compatible_arms": compatible_arms,
                "fps": info.get("fps"),
                "codebase_version": info.get("codebase_version"),
                "format": (
                    "lerobot_v2"
                    if info.get("codebase_version") in {"v2.0", "v2.1"}
                    else "lerobot_v3"
                ),
                "total_episodes": total_episodes,
                "total_frames": info.get("total_frames"),
                "episodes": list(range(total_episodes)),
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


def _camera_names_for_arm(arm_name):
    return [
        camera_name
        for camera_name, camera_config in config.CAMERAS.items()
        if camera_config.get("mount", {}).get("arm_name") == arm_name
    ]


def _prepare_video_cameras(arm_name, record_video):
    if not isinstance(record_video, bool):
        raise ValueError("record_video 必須是 bool")
    if not record_video:
        return {}

    camera_names = _camera_names_for_arm(arm_name)
    if not camera_names:
        raise ValueError(
            f"arm '{arm_name}' 沒有綁定相機；請在 config.CAMERAS 的 "
            f"mount.arm_name 設為 '{arm_name}'"
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
                        _sample_to_lerobot_frame(
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
):
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

            if arm_name not in config.ARMS:
                raise ValueError(
                    f"不支援的 arm_name: {arm_name}；"
                    f"可用值為 {', '.join(sorted(config.ARMS))}"
                )
            if gripper_name not in config.GRIPPERS:
                raise ValueError(f"不支援的 gripper_name: {gripper_name}")
            expected_arm = config.GRIPPERS[gripper_name].get("arm_name")
            if expected_arm != arm_name:
                raise ValueError(
                    f"gripper '{gripper_name}' 屬於 arm '{expected_arm}'，"
                    f"不可搭配 arm '{arm_name}'"
                )

            camera_frames = _prepare_video_cameras(arm_name, record_video)
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

            task_entry, _ = _resolve_task_dataset(task, arm_name, output_path)
            task_id = task_entry["task_id"]
            full_output_path = task_entry["dataset_path"]

            fps = max(1, int(round(1.0 / interval)))
            dataset = _open_lerobot_dataset(
                full_output_path,
                fps,
                camera_frames,
                config.ARMS[arm_name]["driver"],
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
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("錄製執行緒無法在 10 秒內停止")

        if freedrive:
            response = arm_service.stop_arm_freedrive(arm_name)
            _require_success(response, "arm_service.stop_arm_freedrive")

        with _record_lock:
            dataset = _record_dataset
            pending_sample = _record_pending_sample
            episode_index = int(dataset.meta.total_episodes) if dataset else None
            if dataset is None or pending_sample is None:
                raise RuntimeError("沒有足夠的 sample 可建立 LeRobot episode")
            # The final observation has no future command; use a zero TCP delta
            # and preserve its current gripper state.
            dataset.add_frame(
                _sample_to_lerobot_frame(
                    pending_sample, pending_sample, _record_task, done=True
                )
            )
            _record_saved_frame_count += 1
            sample_count = _record_saved_frame_count

        try:
            dataset.save_episode()
        finally:
            dataset.finalize()

        stopped_camera_names = _stop_video_cameras(camera_names)

        duration_seconds = max(0.0, time.perf_counter() - start_time)
        info = _read_dataset_info(output_path) or {}
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
            _load_lerobot_episode(input_path, episode_index)
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
            if (
                config.GRIPPERS.get(gripper_name, {}).get("driver")
                == "robotiq_eseries"
            ):
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
            config.GRIPPERS.get(gripper_name, {}).get("driver")
            == "robotiq_eseries"
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

            full_path, _, _, _ = _load_lerobot_episode(
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

            if arm_name not in config.ARMS:
                raise ValueError(f"不支援的 arm_name: {arm_name}")
            if gripper_name not in config.GRIPPERS:
                raise ValueError(f"不支援的 gripper_name: {gripper_name}")
            expected_arm = config.GRIPPERS[gripper_name].get("arm_name")
            if expected_arm != arm_name:
                raise ValueError(
                    f"gripper '{gripper_name}' 屬於 arm '{expected_arm}'，"
                    f"不可搭配 arm '{arm_name}'"
                )
            info = _read_dataset_info(full_path) or {}
            robot_type = info.get("robot_type")
            configured_driver = config.ARMS[arm_name].get("driver")
            if robot_type != configured_driver:
                raise ValueError(
                    f"dataset robot_type='{robot_type}' 不可播放到 "
                    f"arm '{arm_name}'（driver='{configured_driver}'）"
                )

            # ==================================================
            # Hardware Validation
            # ==================================================

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
