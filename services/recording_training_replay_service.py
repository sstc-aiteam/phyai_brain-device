import os
import threading
from functools import wraps
import numpy as np
from services import arm_service, camera_service, gripper_service
from utils.response import error, success
from recording_training_replay.recording.task_registry import (
    normalize_task, normalize_dataset_mode, TaskRegistry,
)
from recording_training_replay.replay.dataset import raw_episode_details, replay_dataset_catalog
from recording_training_replay.training.dataset import training_dataset_catalog
from recording_training_replay.recording.executor import RawRecorder
from recording_training_replay.replay.executor import ReplayExecutor
from recording_training_replay.training import executor as training_executor


MODULE = "recording_training_replay"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "lerobot_datasets")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
TASK_REGISTRY_PATH = os.path.join(DEFAULT_DATASET_DIR, "task_registry.json")
# ============================================================
# RECORDING
# ============================================================

_recorder = RawRecorder()
_command_lock = threading.RLock()
_tasks = TaskRegistry()


def _serialize_device_command(function):
    @wraps(function)
    def execute(*args, **kwargs):
        with _command_lock:
            return function(*args, **kwargs)
    return execute


def _service_data(response, action):
    if not isinstance(response, dict) or response.get("status") != "success" or response.get("result") is False:
        message = response.get("message", "service call failed") if isinstance(response, dict) else "invalid response"
        raise RuntimeError(f"{action}: {message}")
    return response.get("data") or {}


def register_task(task):
    try:
        entry = _tasks.register(TASK_REGISTRY_PATH, task)
        return success(MODULE, "register_task", data={
            "task": entry, "registry_path": TASK_REGISTRY_PATH})
    except Exception as exc:
        return error(MODULE, "register_task", error=exc, error_type=type(exc).__name__)


def _recording_path(output_path):
    root = os.path.realpath(RAW_DATA_DIR)
    path = os.path.realpath(output_path or root)
    if os.path.commonpath([root, path]) != root:
        raise ValueError("output_path 必須位於 raw_data")
    return path


def _require_connected_arm(arm_name):
    data = _service_data(arm_service.get_arm_status(arm_name), "get_arm_status")
    arms = data.get("arms") or []
    if not arms or not (arms[0].get("status") or {}).get("connected"):
        raise RuntimeError(f"arm '{arm_name}' 未連線")
    return arms[0]


@_serialize_device_command
def start_recording(
    arm_name, gripper_name=None, output_path=None, interval=0.1,
    freedrive=True, record_video=True, task="robot demonstration",
    initial_gripper_position=0, dataset_mode="multi_task",
    dataset_format="raw", dataset_name=None, training_target=None,
):
    """Prepare devices and start raw recording. Legacy format hints are accepted
    for existing clients; recording always uses the raw schema.
    """

    action = "start_recording"
    started_camera = False
    enabled_freedrive = False
    try:
        if _replay.execute("status")["playing"]:
            raise RuntimeError("Replay 進行中，無法同時錄製")
        if _recorder.execute("status")["session_active"]:
            raise RuntimeError("recording already in progress")
        arm_name = str(arm_name or "").strip().lower()
        if not arm_name:
            raise ValueError("arm_name 不可為空")
        task = normalize_task(task)
        mode = normalize_dataset_mode(dataset_mode)
        if mode == "multi_task":
            _tasks.require(TASK_REGISTRY_PATH, task)
        interval = float(interval)
        if not np.isfinite(interval) or not np.isclose(interval, 0.1, rtol=0, atol=1e-9):
            raise ValueError("錄製規格固定為 10 FPS，interval 必須是 0.1 秒")
        interval = 0.1

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
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError("錄製相機必須提供 640×480 uint8 彩色影像")
            first_images[camera_name] = image.copy()

        arm_info = _require_connected_arm(arm_name)
        if arm_info.get("driver") not in {"ur7e", "ur5", "ur5e"}:
            raise ValueError("此錄製 schema 僅支援 UR 的公尺／弧度／axis-angle 狀態")
        path = _recording_path(output_path)
        import config
        camera_clock_sources = {}
        for name in camera_names:
            driver = config.CAMERAS.get(name, {}).get("driver")
            camera_clock_sources[name] = {
                "driver": driver, "clock": "driver_native_unmapped",
                "sensor_timestamp_unit": "millisecond" if driver == "d405" else "second" if driver == "logitech" else "unknown",
                "host_timestamp_unit": "second", "host_clock": "host_monotonic",
                "native_timestamp_event": "sensor_frame_timestamp" if driver == "d405" else "host_wall_clock_after_capture_read" if driver == "logitech" else "unknown",
                "frame_id_source": "sensor_frame_number" if driver == "d405" else "unavailable",
                "mapping_to_host": None,
            }
        metadata = {
            "task": task, "arm_name": arm_name, "gripper_name": gripper_name,
            "robot_type": arm_info.get("driver"), "interval": interval,
            "camera_names": camera_names, "camera_clock_sources": camera_clock_sources,
            "dataset_mode": mode, "dataset_name": dataset_name,
            "cameras": {name: {"shape": list(image.shape), "dtype": str(image.dtype)}
                        for name, image in first_images.items()},
            "gripper_source": "commanded", "image_color_order": "BGR",
            "freedrive": bool(freedrive),
        }

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
            gripper_command = position
        else:
            gripper_command = 0.0
        if freedrive:
            _service_data(arm_service.start_arm_freedrive(arm_name), "start_arm_freedrive")
            enabled_freedrive = True

        arm = arm_service._get_arm_context(arm_name)[1]
        cameras = {name: camera_service._get_camera_context(name)[1] for name in camera_names}
        data = _recorder.execute("start", path=path, metadata=metadata, arm=arm,
                                 cameras=cameras, gripper_command=gripper_command)
        return success(MODULE, action, data=data)
    except Exception as exc:
        if enabled_freedrive:
            arm_service.stop_arm_freedrive(arm_name)
        if started_camera:
            for camera_name in [arm_name]:
                camera_service.stop_camera(camera_name)
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


@_serialize_device_command
def stop_recording(arm_name=None, gripper_name=None, dataset_format=None):
    action = "stop_recording"
    details = _recorder.execute("status")
    had_session = details["session_active"]
    try:
        data = _recorder.execute("stop")
        return success(MODULE, action, data=data)
    except Exception as exc:
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)
    finally:
        # Only release devices after the worker has stopped, including failures.
        if had_session and not _recorder.execute("status")["recording"]:
            if details.get("freedrive"):
                arm_service.stop_arm_freedrive(details["arm_name"])
            for name in details.get("camera_names", []):
                camera_service.stop_camera(name)


def get_recording_status(dataset_format=None):
    return success(MODULE, "get_recording_status", data=_recorder.execute("status"))


def get_task_registry():
    try:
        registry = _tasks.snapshot(TASK_REGISTRY_PATH)
        return success(MODULE, "get_task_registry", data={
            "version": registry.get("version", 1),
            "dataset_root": RAW_DATA_DIR,
            "registry_path": TASK_REGISTRY_PATH,
            "tasks": registry["tasks"],
            "legacy_datasets": [],
        })
    except Exception as exc:
        return error(
            MODULE, "get_task_registry", error=exc,
            error_type=type(exc).__name__,
        )


def _gripper_ratio(value):
    """Accept the UI's 0..255 values and the drivers' 0..1 ratios."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("gripper value 必須是有限數值")
    return max(0.0, min(1.0, value / 255.0 if value > 1 else value))


@_serialize_device_command
def _recording_gripper_command(action, position, gripper_name=None, **kwargs):
    try:
        status = _recorder.execute("status")
        name = gripper_name or status.get("gripper_name")
        if not name:
            raise ValueError("recording gripper 未設定")
        position = _gripper_ratio(position)
        for key in ("speed", "force"):
            if kwargs.get(key) is not None:
                kwargs[key] = _gripper_ratio(kwargs[key])
        if action == "move":
            response = gripper_service.move_gripper(name, position, **kwargs)
        else:
            operation = {"open": gripper_service.open_gripper,
                         "close": gripper_service.close_gripper}[action]
            response = operation(name, **kwargs)
        if (response.get("status") == "success" and response.get("result") is not False
                and status["session_active"] and name == status.get("gripper_name")):
            _recorder.execute("set_gripper_command", position=position)
        return response
    except Exception as exc:
        return error(MODULE, f"{action}_recording_gripper", error=exc,
                     error_type=type(exc).__name__)


def move_recording_gripper(position, speed=None, force=None, wait=False, timeout=None):
    return _recording_gripper_command("move", position, speed=speed, force=force,
                                      wait=wait, timeout=timeout)


def open_recording_gripper(gripper_name=None, **kwargs):
    return _recording_gripper_command("open", 0.0, gripper_name, **kwargs)


def close_recording_gripper(gripper_name=None, **kwargs):
    return _recording_gripper_command("close", 1.0, gripper_name, **kwargs)


# ============================================================
# REPLAY
# ============================================================

_replay = ReplayExecutor()


def _normalize_input_path(input_path):
    if not isinstance(input_path, str) or not input_path.strip():
        raise ValueError("input_path 必須是非空字串")
    path = input_path.strip()
    path = path if os.path.isabs(path) else os.path.join(RAW_DATA_DIR, path)
    path = os.path.realpath(path)
    roots = [os.path.realpath(RAW_DATA_DIR), os.path.realpath(DEFAULT_DATASET_DIR)]
    if not any(os.path.commonpath([path, root]) == root for root in roots):
        raise ValueError("input_path 必須位於 raw_data 內")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"dataset 不存在：{path}")
    if not raw_episode_details(path):
        raise ValueError(f"沒有已完成的 raw data episode：{path}")
    return path


def _available_replay_arms():
    data = _service_data(arm_service.get_arm_status(), "get_arm_status")
    return [
        arm for arm in (data.get("arms") or [])
        if (arm.get("status") or {}).get("connected", True)
    ]


def get_replay_catalog():
    try:
        roots_exist = any(os.path.isdir(root) for root in (RAW_DATA_DIR, DEFAULT_DATASET_DIR))
        arms = _available_replay_arms() if roots_exist else []
        datasets = replay_dataset_catalog(RAW_DATA_DIR, DEFAULT_DATASET_DIR, arms)
        return success(MODULE, "get_replay_catalog", data={
            "dataset_root": RAW_DATA_DIR, "datasets": datasets,
            "training_datasets": training_dataset_catalog(DEFAULT_DATASET_DIR)})
    except Exception as exc:
        return error(MODULE, "get_replay_catalog", error=exc, error_type=type(exc).__name__)


def start_robot_playback(input_path, arm_name=None, gripper_name=None,
                         episode_index=0, speed=None, acceleration=None,
                         lookahead_time=0.1, gain=300, move_to_start=True,
                         move_to_start_speed=None, move_to_start_acceleration=None,
                         dataset_format=None):
    """Prepare replay devices; legacy lookahead_time/gain remain API-compatible."""
    try:
        with _command_lock:
            if _recorder.execute("status")["session_active"]:
                raise RuntimeError("錄製進行中，無法同時 Replay")
            if dataset_format and dataset_format != "raw":
                raise ValueError("Replay 僅支援 raw data")
            path = _normalize_input_path(input_path)
            episode = next((item for item in raw_episode_details(path)
                            if item["episode_index"] == int(episode_index)), None)
            if episode is None:
                raise ValueError(f"找不到已完成的 episode {episode_index}")
            metadata = episode["metadata"]
            import config as device_config
            recorded_arm = metadata.get("arm_name")
            if not recorded_arm:
                candidates = [name for name, item in device_config.ARMS.items()
                              if item.get("driver") == metadata.get("robot_type")]
                if len(candidates) != 1:
                    raise ValueError("episode 缺少 arm_name，無法唯一決定播放手臂")
                recorded_arm = candidates[0]
            if arm_name and arm_name != recorded_arm:
                raise ValueError("指定手臂與 episode 錄製手臂不一致")
            recorded_gripper = metadata.get("gripper_name") or None
            if gripper_name and gripper_name != recorded_gripper:
                raise ValueError("指定夾爪與 episode 錄製夾爪不一致")
            arm_name, gripper_name = recorded_arm, recorded_gripper
            _require_connected_arm(arm_name)
            name, arm, driver, config = arm_service._get_arm_context(arm_name)
            speed, acceleration = arm_service._get_motion_params(config, speed, acceleration)
            gripper = None
            params = None
            if gripper_name:
                gripper_name, gripper, _, gripper_config = gripper_service._get_gripper_context(gripper_name)
                gs, gf, gw, gt = gripper_service._get_motion_params(gripper_config, None, None, True, None)
                params = dict(speed=gs, force=gf, wait=gw, timeout=gt)
            data = _replay.execute("start", input_path=path, episode_index=int(episode_index),
                arm_name=name, driver=driver, arm=arm, gripper_name=gripper_name,
                gripper=gripper, gripper_params=params, speed=speed, acceleration=acceleration,
                move_first_segment_to_start=bool(move_to_start),
                move_to_start_speed=move_to_start_speed,
                move_to_start_acceleration=move_to_start_acceleration)
            return success(MODULE, "start_robot_playback", data=data)
    except Exception as exc:
        return error(MODULE, "start_robot_playback", error=exc, error_type=type(exc).__name__)


def stop_robot_playback(arm_name=None, gripper_name=None):
    try:
        return success(MODULE, "stop_robot_playback", data=_replay.execute("stop"))
    except Exception as exc:
        return error(MODULE, "stop_robot_playback", error=exc, error_type=type(exc).__name__)


def get_robot_playback_status():
    return success(MODULE, "get_robot_playback_status", data=_replay.execute("status"))


# ============================================================
# TRAINING
# ============================================================

def register_training_user(username=None, password=None, **kwargs):
    return training_executor.execute("register", username=username, password=password, **kwargs)


def login_training_user(username=None, password=None, **kwargs):
    return training_executor.execute("login", username=username, password=password, **kwargs)


def training_logout(token):
    return training_executor.execute("logout", token=token)


def get_training_jobs(token=None, query="", job_id=None):
    return training_executor.execute("jobs", token=token, query=query, job_id=job_id)


def get_dataset_upload_status(token=None):
    return training_executor.execute("status", token=token)


def start_dataset_upload(dataset_path=None, model=None, token=None):
    return training_executor.execute("start", dataset_path=dataset_path, model=model, token=token,
        dataset_root=DEFAULT_DATASET_DIR,
        recording=_recorder.execute("status")["session_active"])
