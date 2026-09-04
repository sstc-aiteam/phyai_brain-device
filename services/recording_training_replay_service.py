"""Service implementation for the Recording / Replay / Training subsystem."""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

import numpy as np

from services import arm_service, camera_service, gripper_service
from utils.response import error, success


MODULE = "recording_training_replay"
logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "lerobot_datasets")
TASK_REGISTRY_PATH = os.path.join(DEFAULT_DATASET_DIR, "task_registry.json")
# Default names for the robot's arm, gripper, and camera components.
DEFAULT_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
LEROBOT_STATE_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper",
]
LEROBOT_TCP_POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
LEROBOT_TCP_ACTION_NAMES = [
    "tcp_local_delta_x", "tcp_local_delta_y", "tcp_local_delta_z",
    "tcp_local_delta_rx", "tcp_local_delta_ry", "tcp_local_delta_rz", "gripper",
]
LEROBOT_TCP_BASE_ACTION_NAMES = [
    "tcp_base_delta_x", "tcp_base_delta_y", "tcp_base_delta_z",
    "tcp_base_delta_rx", "tcp_base_delta_ry", "tcp_base_delta_rz", "gripper",
]
LEROBOT_JOINT_ACTION_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper",
]


# ============================================================
# RECORDING
# ============================================================

_record_lock = threading.RLock()
_task_registry_lock = threading.RLock()
_record_stop_event = threading.Event()
_record_thread = None
_record_writer = None
_record_pending_sample = None
_record_error = None
_record_start_time = None
_record_output_path = None
_record_interval = 0.1
_record_arm_name = None
_record_gripper_name = None
_record_camera_names = []
_record_task = None
_record_format = None
_record_sample_count = 0
_record_episode_index = None
_record_freedrive = False
_record_gripper_position = 0.0


def _project_root():
    return PROJECT_ROOT


def _read_dataset_info(dataset_path):
    path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as source:
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


def _service_data(response, action):
    if not isinstance(response, dict) or response.get("status") != "success" or response.get("result") is False:
        message = response.get("message", "service call failed") if isinstance(response, dict) else "invalid response"
        raise RuntimeError(f"{action}: {message}")
    return response.get("data") or {}


def _normalize_dataset_mode(value):
    mode = str(value or "multi_task").strip().lower().replace("-", "_")
    if mode not in {"multi_task", "single_task"}:
        raise ValueError("dataset_mode 必須是 multi_task 或 single_task")
    return mode


def _task_slug(task):
    return re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:48] or "task"


def _dataset_slug(dataset_name):
    return (
        re.sub(r"[^a-z0-9._-]+", "_", str(dataset_name).lower())
        .strip("._-")[:64]
        or "dataset"
    )


def _normalize_task(task):
    normalized = " ".join(str(task or "").split()).lower()
    if not normalized:
        raise ValueError("task 不可為空")
    if len(normalized) > 200:
        raise ValueError("task 長度不可超過 200 個字元")
    return normalized


def _read_task_registry():
    if not os.path.isfile(TASK_REGISTRY_PATH):
        return {"version": 1, "tasks": []}
    with open(TASK_REGISTRY_PATH, "r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("task_registry.json 格式錯誤")
    return payload


def _write_task_registry(payload):
    os.makedirs(os.path.dirname(TASK_REGISTRY_PATH), exist_ok=True)
    temporary = f"{TASK_REGISTRY_PATH}.tmp"
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    os.replace(temporary, TASK_REGISTRY_PATH)


def _registered_task(task):
    normalized = _normalize_task(task)
    with _task_registry_lock:
        registry = _read_task_registry()
        return next(
            (item for item in registry["tasks"] if item.get("task") == normalized),
            None,
        )


def register_task(task):
    action = "register_task"
    try:
        normalized = _normalize_task(task)
        with _task_registry_lock:
            registry = _read_task_registry()
            existing = next(
                (item for item in registry["tasks"] if item.get("task") == normalized),
                None,
            )
            if existing is None:
                task_index = max(
                    (int(item.get("task_index", -1)) for item in registry["tasks"]),
                    default=-1,
                ) + 1
                existing = {"task_index": task_index, "task": normalized}
                registry["tasks"].append(existing)
                _write_task_registry(registry)
        return success(MODULE, action, data={
            "task": existing,
            "registry_path": TASK_REGISTRY_PATH,
        })
    except Exception as exc:
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def _dataset_path(
    dataset_format, dataset_mode, task, arm_name, output_path=None,
    dataset_name=None,
):
    if output_path:
        path = os.path.abspath(output_path)
        if os.path.commonpath([DEFAULT_DATASET_DIR, path]) != DEFAULT_DATASET_DIR:
            raise ValueError("output_path 必須位於 lerobot_datasets")
        return path
    mode = _normalize_dataset_mode(dataset_mode)
    if mode == "multi_task":
        if not str(dataset_name or "").strip():
            raise ValueError("multi_task 必須設定 dataset_name")
        folder = os.path.join(mode, _dataset_slug(dataset_name))
    else:
        folder = os.path.join(mode, _task_slug(task))
    return os.path.join(DEFAULT_DATASET_DIR, dataset_format, folder)


def _recording_arm_info(arm_name):
    data = _service_data(arm_service.get_arm_status(arm_name), "get_arm_status")
    arms = data.get("arms") or []
    if not arms or not (arms[0].get("status") or {}).get("connected"):
        raise RuntimeError(f"arm '{arm_name}' 未連線")
    return arms[0]


def _capture_sample(arm_name, gripper_name, camera_names):
    # One status snapshot contains both values.  Calling get_arm_pose() and
    # get_arm_joints() separately makes the service query the RTDE Control
    # channel twice per recording frame, which can disturb UR freedrive.
    arm_data = _service_data(arm_service.get_arm_status(arm_name), "get_arm_status")
    arms = arm_data.get("arms") or []
    status = (arms[0].get("status") or {}) if arms else {}
    pose = status.get("pose")
    joints = status.get("joints")
    if pose is None or joints is None:
        raise RuntimeError("手臂 pose 或 joints 無法取得")

    # Do not poll get_gripper_status() here.  The Robotiq URScript driver
    # refreshes expired status by temporarily replacing the UR RTDE control
    # program, which interrupts freedrive.  Recording gripper endpoints keep
    # this command-state value current instead.
    gripper = _record_gripper_position if gripper_name else 0.0

    images = {}
    for camera_name in camera_names:
        frame = camera_service.get_frame(camera_name)
        image = frame.get("color_image") if isinstance(frame, dict) else None
        if image is None:
            raise RuntimeError(f"camera '{camera_name}' 沒有 color frame")
        images[camera_name] = image.copy()
    return {
        "timestamp": time.time(),
        "tcp_pose": list(pose),
        "joints": list(joints),
        "gripper": max(0.0, min(1.0, gripper)),
        "images": images,
    }


def _format_adapter(dataset_format):
    if dataset_format == "lerobot_v2":
        from recording_training_replay.recording import lerobotv2
        return lerobotv2
    if dataset_format == "lerobot_v3":
        from recording_training_replay.recording import lerobotv3
        return lerobotv3
    raise ValueError("dataset_format 必須是 lerobot_v2 或 lerobot_v3")


def _record_loop(adapter):
    global _record_pending_sample, _record_sample_count, _record_error
    deadline = time.monotonic()
    try:
        while not _record_stop_event.is_set():
            sample = _capture_sample(_record_arm_name, _record_gripper_name, _record_camera_names)
            # stop_recording() may set the event while a camera/robot read is
            # already in flight.  Treat receipt of that request as the episode
            # boundary and discard the just-finished sample instead of adding
            # a frame captured after the user pressed stop.
            if _record_stop_event.is_set():
                break
            with _record_lock:
                if _record_pending_sample is not None:
                    _record_writer.add_frame(adapter.encode_frame(
                        _record_pending_sample, sample, _record_task, done=False
                    ))
                    _record_sample_count += 1
                _record_pending_sample = sample
            deadline += _record_interval
            _record_stop_event.wait(max(0.0, deadline - time.monotonic()))
    except Exception as exc:
        logger.exception("recording loop failed")
        with _record_lock:
            _record_error = str(exc)
        _record_stop_event.set()


def start_recording(
    arm_name, gripper_name=None, output_path=None, interval=0.1,
    freedrive=True, record_video=True, task="robot demonstration",
    initial_gripper_position=0, dataset_mode="multi_task",
    dataset_format="lerobot_v3", dataset_name=None,
):
    """Start camera, enable freedrive, then record samples in a worker thread."""
    global _record_thread, _record_writer, _record_pending_sample, _record_error
    global _record_start_time, _record_output_path, _record_interval
    global _record_arm_name, _record_gripper_name, _record_camera_names
    global _record_task, _record_format, _record_sample_count, _record_episode_index
    global _record_freedrive, _record_gripper_position

    action = "start_recording"
    started_camera = False
    enabled_freedrive = False
    try:
        with _record_lock:
            if _record_thread and _record_thread.is_alive():
                raise RuntimeError("recording already in progress")
        arm_name = str(arm_name or "").strip().lower()
        if not arm_name:
            raise ValueError("arm_name 不可為空")
        task = _normalize_task(task)
        mode = _normalize_dataset_mode(dataset_mode)
        if mode == "multi_task" and _registered_task(task) is None:
            raise ValueError(f"multi-task 尚未註冊 task：{task}")
        interval = float(interval)
        if interval < 0.033 or interval > 10:
            raise ValueError("interval 必須介於 0.033 到 10 秒")
        fps = int(round(1.0 / interval))
        if dataset_format == "lerobot_v2" and fps != 10:
            raise ValueError("LeRobot v2 錄製 interval 必須是 0.1 秒")

        # Camera is deliberately started before querying/controlling the arm.
        camera_names = [arm_name] if record_video else []
        first_images = {}
        for camera_name in camera_names:
            _service_data(camera_service.start_camera(camera_name), "start_camera")
            started_camera = True
            frame = camera_service.get_frame(camera_name)
            image = frame.get("color_image") if isinstance(frame, dict) else None
            if image is None:
                raise RuntimeError(f"camera '{camera_name}' 沒有 color frame")
            first_images[camera_name] = image.copy()

        arm_info = _recording_arm_info(arm_name)
        path = _dataset_path(
            dataset_format, mode, task, arm_name, output_path,
            dataset_name=dataset_name,
        )
        adapter = _format_adapter(dataset_format)
        writer = adapter.create_writer(path, fps, first_images, arm_info.get("driver"))
        episode_index = int(getattr(getattr(writer, "meta", None), "total_episodes", 0))

        if gripper_name:
            position = max(0.0, min(1.0, float(initial_gripper_position) / 255.0))
            _service_data(gripper_service.move_gripper(
                gripper_name,
                position,
                speed=1.0,
                force=150.0 / 255.0,
                wait=True,
                timeout=5.0,
            ), "move_gripper")
            _record_gripper_position = position
        else:
            _record_gripper_position = 0.0
        if freedrive:
            _service_data(arm_service.start_arm_freedrive(arm_name), "start_arm_freedrive")
            enabled_freedrive = True

        with _record_lock:
            _record_writer = writer
            _record_pending_sample = None
            _record_error = None
            _record_start_time = time.monotonic()
            _record_output_path = path
            _record_interval = interval
            _record_arm_name = arm_name
            _record_gripper_name = gripper_name
            _record_camera_names = camera_names
            _record_task = task
            _record_format = dataset_format
            _record_sample_count = 0
            _record_episode_index = episode_index
            _record_freedrive = bool(freedrive)
            _record_stop_event.clear()
            _record_thread = threading.Thread(target=_record_loop, args=(adapter,), daemon=True)
            _record_thread.start()
        return success(MODULE, action, data={
            "recording": True, "output_path": path, "absolute_path": path,
            "arm_name": arm_name, "gripper_name": gripper_name,
            "camera_names": camera_names, "record_video": bool(record_video),
            "dataset_format": dataset_format, "episode_index": episode_index,
            "startup_seconds": 0.0,
        })
    except Exception as exc:
        if enabled_freedrive:
            arm_service.stop_arm_freedrive(arm_name)
        if started_camera:
            for camera_name in [arm_name]:
                camera_service.stop_camera(camera_name)
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def stop_recording(arm_name=None, gripper_name=None, dataset_format=None):
    global _record_thread, _record_writer, _record_pending_sample, _record_sample_count
    action = "stop_recording"
    started = time.monotonic()
    try:
        with _record_lock:
            thread = _record_thread
            adapter = _format_adapter(_record_format) if _record_format else None
        if not thread:
            raise RuntimeError("recording is not active")
        _record_stop_event.set()
        thread.join(timeout=max(10.0, _record_interval * 3))
        if thread.is_alive():
            raise TimeoutError("recording thread 無法停止")
        with _record_lock:
            if _record_error:
                raise RuntimeError(_record_error)
            if _record_pending_sample is not None:
                _record_writer.add_frame(adapter.encode_frame(
                    _record_pending_sample, _record_pending_sample, _record_task, done=True
                ))
                _record_sample_count += 1
            if _record_sample_count == 0:
                raise RuntimeError("沒有可儲存的 recording sample")
            _record_writer.save_episode()
            _record_writer.finalize()
        if _record_freedrive:
            _service_data(arm_service.stop_arm_freedrive(_record_arm_name), "stop_arm_freedrive")
        for camera_name in _record_camera_names:
            _service_data(camera_service.stop_camera(camera_name), "stop_camera")
        data = {
            "recording": False, "output_path": _record_output_path,
            "sample_count": _record_sample_count, "episode_index": _record_episode_index,
            "dataset_format": _record_format,
            "stop_timings": {"total_seconds": time.monotonic() - started},
        }
        with _record_lock:
            _record_thread = None
            _record_writer = None
            _record_pending_sample = None
        return success(MODULE, action, data=data)
    except Exception as exc:
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def get_recording_status(dataset_format=None):
    with _record_lock:
        recording = bool(_record_thread and _record_thread.is_alive())
        return success(MODULE, "get_recording_status", data={
            "recording": recording,
            "elapsed_seconds": time.monotonic() - _record_start_time if recording else 0.0,
            "sample_count": _record_sample_count,
            "episode_index": _record_episode_index,
            "arm_name": _record_arm_name,
            "gripper_name": _record_gripper_name,
            "camera_names": list(_record_camera_names),
            "record_video": bool(_record_camera_names),
            "output_path": _record_output_path,
            "selected_format": _record_format or dataset_format,
            "error": _record_error,
        })


def get_task_registry():
    try:
        with _task_registry_lock:
            registry = _read_task_registry()
        return success(MODULE, "get_task_registry", data={
            "version": registry.get("version", 1),
            "dataset_root": DEFAULT_DATASET_DIR,
            "registry_path": TASK_REGISTRY_PATH,
            "tasks": registry["tasks"],
            "legacy_datasets": [],
        })
    except Exception as exc:
        return error(
            MODULE, "get_task_registry", error=exc,
            error_type=type(exc).__name__,
        )


def move_recording_gripper(position, speed=None, force=None, wait=False, timeout=None):
    global _record_gripper_position
    if not _record_gripper_name:
        return error(MODULE, "move_recording_gripper", message="recording gripper 未設定")
    normalize = lambda value: None if value is None else max(0.0, min(1.0, float(value) / 255.0 if float(value) > 1 else float(value)))
    normalized_position = normalize(position)
    response = gripper_service.move_gripper(
        _record_gripper_name, normalized_position, speed=normalize(speed),
        force=normalize(force), wait=wait, timeout=timeout,
    )
    if response.get("status") == "success" and response.get("result") is not False:
        _record_gripper_position = normalized_position
    return response


def open_recording_gripper(gripper_name=None, **kwargs):
    global _record_gripper_position
    for key in ("speed", "force"):
        if kwargs.get(key) is not None and float(kwargs[key]) > 1:
            kwargs[key] = float(kwargs[key]) / 255.0
    response = gripper_service.open_gripper(gripper_name or _record_gripper_name, **kwargs)
    if response.get("status") == "success" and response.get("result") is not False:
        _record_gripper_position = 0.0
    return response


def close_recording_gripper(gripper_name=None, **kwargs):
    global _record_gripper_position
    for key in ("speed", "force"):
        if kwargs.get(key) is not None and float(kwargs[key]) > 1:
            kwargs[key] = float(kwargs[key]) / 255.0
    response = gripper_service.close_gripper(gripper_name or _record_gripper_name, **kwargs)
    if response.get("status") == "success" and response.get("result") is not False:
        _record_gripper_position = 1.0
    return response


start_recording_service = start_recording
stop_recording_service = stop_recording
get_recording_status_service = get_recording_status

# ============================================================
# REPLAY
# ============================================================

# Replay implementation intentionally cleared for redesign.


# ============================================================
# TRAINING
# ============================================================

# Training implementation intentionally cleared for redesign.