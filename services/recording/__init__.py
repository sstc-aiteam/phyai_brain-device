"""LeRobot recording service router.

API routes use this module to select the v2 or v3 backend. Code that needs a
specific implementation can import ``lerobotv2`` or ``lerobotv3`` directly.
"""

from services.recording.lerobotv3 import *  # noqa: F401,F403
import os
import threading

from services.recording import lerobotv2 as _v2
from services.recording import lerobotv3 as _implementation

_services = {"lerobot_v2": _v2, "lerobot_v3": _implementation}
_selection_lock = threading.RLock()
_active_format = None
_active_playback_format = None


def _service(dataset_format):
    normalized = str(dataset_format or "lerobot_v3").strip().lower()
    if normalized not in _services:
        raise ValueError("dataset_format 必須是 lerobot_v2 或 lerobot_v3")
    return _services[normalized], normalized


def start_robot_recording(*args, dataset_format="lerobot_v3", **kwargs):
    global _active_format
    service, normalized = _service(dataset_format)
    with _selection_lock:
        if _active_format is not None:
            raise RuntimeError("recording already in progress")
        if normalized == "lerobot_v3" and kwargs.get("output_path") is None:
            arm_name = kwargs.get("arm_name") or (args[0] if args else None)
            task = kwargs.get("task", "robot demonstration")
            kwargs["output_path"] = os.path.join(
                _implementation.DEFAULT_DATASET_DIR,
                "lerobot_v3",
                f"{_implementation._task_slug(str(task))}_{arm_name}",
            )
        response = service.start_robot_recording(*args, **kwargs)
        if response.get("status") == "success" and response.get("result") is not False:
            _active_format = normalized
        return response


def stop_robot_recording(*args, dataset_format=None, **kwargs):
    global _active_format
    with _selection_lock:
        normalized = _active_format or dataset_format or "lerobot_v3"
        service, _ = _service(normalized)
        response = service.stop_robot_recording(*args, **kwargs)
        if response.get("status") != "error":
            _active_format = None
        return response


def get_robot_recording_status(dataset_format=None):
    with _selection_lock:
        normalized = _active_format or dataset_format or "lerobot_v3"
        service, normalized = _service(normalized)
        response = service.get_robot_recording_status()
        response.setdefault("data", {})["selected_format"] = normalized
        return response


def start_robot_playback(*args, dataset_format=None, **kwargs):
    global _active_playback_format
    input_path = kwargs.get("input_path") or (args[0] if args else None)
    full_path = input_path
    if isinstance(full_path, str) and not os.path.isabs(full_path):
        full_path = os.path.join(_implementation.DEFAULT_DATASET_DIR, full_path)
    info = _implementation._read_dataset_info(full_path) if full_path else None
    detected_format = (
        "lerobot_v2"
        if info and info.get("codebase_version") in {"v2.0", "v2.1"}
        else "lerobot_v3"
    )
    normalized = (
        _service(dataset_format)[1]
        if dataset_format is not None
        else detected_format
    )
    if info is not None and normalized != detected_format:
        raise ValueError(
            f"dataset_format={normalized} 與資料集內容 {detected_format} 不符"
        )
    response = _services[normalized].start_robot_playback(*args, **kwargs)
    if response.get("status") == "success" and response.get("result") is not False:
        _active_playback_format = normalized
    return response


def stop_robot_playback(*args, **kwargs):
    global _active_playback_format
    normalized = _active_playback_format or "lerobot_v3"
    try:
        return _services[normalized].stop_robot_playback(*args, **kwargs)
    finally:
        _active_playback_format = None


def get_robot_playback_status():
    global _active_playback_format
    normalized = _active_playback_format or "lerobot_v3"
    response = _services[normalized].get_robot_playback_status()
    if not (response.get("data") or {}).get("playing", False):
        _active_playback_format = None
    return response


def __getattr__(name):
    """Expose private helpers used by the legacy playback tests."""
    return getattr(_implementation, name)
