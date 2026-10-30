"""LeRobot recording lifecycle and dataset writer selection."""

import logging
import os
import re
import threading
import time

from services import arm_service, camera_service, gripper_service
from utils.response import error, success
from recording_training_replay.dataset import read_dataset_info
from recording_training_replay.dataset import save_episode_devices
from .schema import (
    _lerobot_features, sample_to_lerobot_frame,
    LEROBOT_TCP_ACTION_NAMES, LEROBOT_TCP_BASE_ACTION_NAMES, LEROBOT_JOINT_ACTION_NAMES,
)
from recording_training_replay.recording.task_registry import (
    TaskRegistry, normalize_task, normalize_dataset_mode,
)

from recording_training_replay.storage import (
    TASK_REGISTRY_PATH,
    LEROBOT_DATASET_DIR as DEFAULT_DATASET_DIR,
)


MODULE = "recording_training_replay"
logger = logging.getLogger(__name__)
# ============================================================
# RECORDING
# ============================================================

_record_lock = threading.RLock()
_tasks = TaskRegistry()
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
_record_sample_count = 0
_record_episode_index = None
_record_freedrive = False
_record_gripper_position = 0.0


def _dataset_repo_id(dataset_path):
    name = os.path.basename(os.path.normpath(dataset_path)) or "ur7e_dataset"
    safe_name = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in name
    )
    return f"local/{safe_name}"


def open_lerobot_dataset(dataset_path, fps, camera_frames, robot_type):

    info = read_dataset_info(dataset_path)
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
        expected_video_keys = {
            key for key, feature in _lerobot_features(camera_frames).items()
            if feature.get("dtype") == "video"
        }
        if set(video_keys) != expected_video_keys:
            raise ValueError(
                "既有資料集的影像欄位與 observation.images.arm 不相容；"
                "請使用新的 dataset_name 或 output_path"
            )


    from .dataset_writer import HttpDatasetWriter as Writer
    return Writer(dataset_path, {
        **common, "fps": fps, "features": _lerobot_features(camera_frames),
        "robot_type": robot_type, "use_videos": record_video,
    })


def _service_data(response, action):
    if not isinstance(response, dict) or response.get("status") != "success" or response.get("result") is False:
        message = response.get("message", "service call failed") if isinstance(response, dict) else "invalid response"
        raise RuntimeError(f"{action}: {message}")
    return response.get("data") or {}


def task_slug(task):
    return re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:48] or "task"


def dataset_slug(dataset_name):
    return (
        re.sub(r"[^a-z0-9._-]+", "_", str(dataset_name).lower())
        .strip("._-")[:64]
        or "dataset"
    )


def _dataset_path(
    dataset_mode, task, output_path=None,
    dataset_name=None, arm_name=None, robot_type=None,
):
    if output_path:
        path = os.path.realpath(output_path)
        root = os.path.realpath(DEFAULT_DATASET_DIR)
        if os.path.commonpath([root, path]) != root:
            raise ValueError("output_path 必須位於 lerobot_datasets")
        return path
    mode = normalize_dataset_mode(dataset_mode)
    if mode == "multi_task":
        if not str(dataset_name or "").strip():
            raise ValueError("multi_task 必須設定 dataset_name")
        folder = os.path.join(mode, dataset_slug(dataset_name))
    else:
        folder = os.path.join(mode, task_slug(task))
    if not arm_name or not robot_type:
        raise ValueError("自動資料集路徑需要 arm_name 與 robot_type")
    robot_folder = dataset_slug(f"{arm_name}_{robot_type}")
    return os.path.join(DEFAULT_DATASET_DIR, "lerobot_v3", robot_folder, folder)


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


def _require_v3(dataset_format):
    """Validate the legacy HTTP argument; recording has no format selection."""
    if dataset_format not in (None, "lerobot_v3"):
        raise ValueError("此頁面固定使用 LeRobot v3 格式")


def _record_loop():
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
                    _record_writer.add_frame(sample_to_lerobot_frame(
                        _record_pending_sample, sample, _record_task, done=False
                    ))
                    _record_sample_count += 1
                _record_pending_sample = sample
            deadline += _record_interval
            _record_stop_event.wait(max(0.0, deadline - time.monotonic()))
    except Exception as exc:
        logger.exception("recording loop failed")

        # 保留原始錄製錯誤
        with _record_lock:
            _record_error = str(exc)

        # 通知停止錄製
        _record_stop_event.set()

        # 嘗試中止 Dataset Writer
        try:
            if _record_writer is not None:
                _record_writer.abort()

        except Exception:
            logger.exception(
                "Failed to abort LeRobot dataset writer"
            )


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
    global _record_task, _record_sample_count, _record_episode_index
    global _record_freedrive, _record_gripper_position

    action = "start_recording"
    started = time.monotonic()
    writer = None
    started_camera = False
    enabled_freedrive = False
    try:
        _require_v3(dataset_format)  # HTTP compatibility only; validate before hardware access.
        with _record_lock:
            if _record_thread and _record_thread.is_alive():
                raise RuntimeError("recording already in progress")
        arm_name = str(arm_name or "").strip().lower()
        if not arm_name:
            raise ValueError("arm_name 不可為空")
        task = normalize_task(task)
        mode = normalize_dataset_mode(dataset_mode)
        if mode == "multi_task":
            _tasks.require(TASK_REGISTRY_PATH, task)
        interval = 0.1
        fps = 10

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
            from recording_training_replay import startup
            startup.update(camera_ready=True, arm_name=arm_name)

        arm_info = _recording_arm_info(arm_name)
        path = _dataset_path(
            mode, task, output_path,
            dataset_name=dataset_name, arm_name=arm_name,
            robot_type=arm_info.get("driver"),
        )
        writer = open_lerobot_dataset(path, fps, first_images, arm_info.get("driver"))
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
            _record_sample_count = 0
            _record_episode_index = episode_index
            _record_freedrive = bool(freedrive)
            _record_stop_event.clear()
            _record_thread = threading.Thread(target=_record_loop, daemon=True)
            _record_thread.start()
        return success(MODULE, action, data={
            "recording": True, "output_path": path, "absolute_path": path,
            "arm_name": arm_name, "gripper_name": gripper_name,
            "camera_names": camera_names, "record_video": bool(record_video),
            "dataset_format": "lerobot_v3",
            "episode_index": episode_index,
            "startup_seconds": time.monotonic() - started,
        })
    except Exception as exc:

        if writer is not None:
            try:
                writer.abort()
            except Exception:
                logger.exception(
                    "Failed to abort LeRobot writer during startup"
                )

        if enabled_freedrive:
            try:
                arm_service.stop_arm_freedrive(arm_name)
            except Exception:
                logger.exception("Failed to stop freedrive")

        if started_camera:
            try:
                camera_service.stop_camera(arm_name)
            except Exception:
                logger.exception("Failed to stop camera")

        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def stop_recording(arm_name=None, gripper_name=None, dataset_format=None):
    global _record_thread, _record_writer, _record_pending_sample, _record_sample_count
    action = "stop_recording"
    started = time.monotonic()
    try:
        _require_v3(dataset_format)
        with _record_lock:
            thread = _record_thread
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
                _record_writer.add_frame(sample_to_lerobot_frame(
                    _record_pending_sample, _record_pending_sample, _record_task, done=True
                ))
                _record_sample_count += 1
            if _record_sample_count == 0:
                raise RuntimeError("沒有可儲存的 recording sample")
            _record_writer.save_episode()
            save_episode_devices(_record_output_path, _record_episode_index,
                                  _record_arm_name, _record_gripper_name, _record_task)
            _record_writer.finalize()
        if _record_freedrive:
            _service_data(arm_service.stop_arm_freedrive(_record_arm_name), "stop_arm_freedrive")
        for camera_name in _record_camera_names:
            _service_data(camera_service.stop_camera(camera_name), "stop_camera")
        data = {
            "recording": False, "output_path": _record_output_path,
            "sample_count": _record_sample_count, "episode_index": _record_episode_index,
            "dataset_format": "lerobot_v3",
            "stop_timings": {"total_seconds": time.monotonic() - started},
        }
        with _record_lock:
            _record_thread = None
            _record_writer = None
            _record_pending_sample = None
        return success(MODULE, action, data=data)
    except Exception as exc:
        # Keep ownership if the worker is still reading devices. Otherwise
        # release failed sessions so both formats can be used again.
        if _record_thread is not None and not _record_thread.is_alive():
            try:
                if hasattr(_record_writer, "abort"):
                    _record_writer.abort()
            except Exception:
                logger.exception("Failed to abort LeRobot writer")
            if _record_freedrive:
                arm_service.stop_arm_freedrive(_record_arm_name)
            for camera_name in _record_camera_names:
                camera_service.stop_camera(camera_name)
            with _record_lock:
                _record_thread = None
                _record_writer = None
                _record_pending_sample = None
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def get_recording_status(dataset_format=None):
    _require_v3(dataset_format)
    with _record_lock:
        recording = bool(_record_thread and _record_thread.is_alive())
        return success(MODULE, "get_recording_status", data={
            "recording": recording,
            "session_active": _record_thread is not None,
            "elapsed_seconds": time.monotonic() - _record_start_time if recording else 0.0,
            "sample_count": _record_sample_count,
            "episode_index": _record_episode_index,
            "arm_name": _record_arm_name,
            "gripper_name": _record_gripper_name,
            "camera_names": list(_record_camera_names),
            "record_video": bool(_record_camera_names),
            "output_path": _record_output_path,
            "selected_format": "lerobot_v3",
            "error": _record_error,
        })


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


