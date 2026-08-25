import json
import logging
import os
import threading
import time

from services import arm_service, gripper_service
from utils.response import success, error


MODULE = "record"
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


DEFAULT_TRAJECTORY_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "robot_trajectory",
    )
)


# ============================================================
# Recording State
# ============================================================

_record_lock = threading.RLock()
_record_stop_event = threading.Event()

_record_thread = None

_record_trajectory = []
_record_gripper_events = []

_record_start_time = None
_record_output_path = None

_record_interval = DEFAULT_RECORD_INTERVAL

_record_arm_name = None
_record_gripper_name = None

_record_freedrive = False


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


def _normalize_output_path(
    output_path,
):
    if output_path is None:

        filename = time.strftime(
            "recorded_robot_trajectory_%Y%m%d_%H%M%S.json"
        )

        return os.path.join(
            DEFAULT_TRAJECTORY_DIR,
            filename,
        )

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

    if os.path.isabs(
        output_path
    ):
        return output_path

    return os.path.join(
        DEFAULT_TRAJECTORY_DIR,
        output_path,
    )


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

    if os.path.isabs(
        input_path
    ):
        full_path = (
            input_path
        )

    else:
        full_path = (
            os.path.join(
                DEFAULT_TRAJECTORY_DIR,
                input_path,
            )
        )

    if not os.path.isfile(
        full_path
    ):
        raise FileNotFoundError(
            f"trajectory file not found: "
            f"{full_path}"
        )

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
    global _record_trajectory

    next_sample_time = (
        time.perf_counter()
    )

    while not _record_stop_event.is_set():

        with _record_lock:

            start_time = (
                _record_start_time
            )

            interval = (
                _record_interval
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

        if (
            start_time is None
            or arm_name is None
            or gripper_name is None
        ):
            break

        now = (
            time.perf_counter()
        )

        if now < next_sample_time:

            if _record_stop_event.wait(
                next_sample_time
                - now
            ):
                break

        sample_time = (
            time.perf_counter()
        )

        elapsed = (
            sample_time
            - start_time
        )

        try:

            # Freedrive=True：
            # 不 polling e-Series gripper。
            #
            # Gripper 改用 event-based recording。
            sample = (
                _get_record_sample(
                    arm_name=
                        arm_name,

                    gripper_name=
                        gripper_name,

                    include_gripper=
                        not freedrive,
                )
            )

            entry = {
                "t":
                    elapsed,

                "arm":
                    sample[
                        "arm"
                    ],

                "gripper":
                    sample[
                        "gripper"
                    ],
            }

            with _record_lock:
                _record_trajectory.append(
                    entry
                )

        except Exception:

            logger.exception(
                "failed to sample robot state "
                "during recording"
            )

        next_sample_time += (
            interval
        )

        current = (
            time.perf_counter()
        )

        if (
            next_sample_time
            < current
        ):
            next_sample_time = (
                current
                + interval
            )


# ============================================================
# START RECORDING
# ============================================================

def start_robot_recording(
    arm_name,
    gripper_name,
    output_path=None,
    interval=DEFAULT_RECORD_INTERVAL,
    freedrive=False,
):
    action = (
        "start_robot_recording"
    )

    freedrive_started = False

    try:
        global _record_thread
        global _record_trajectory
        global _record_gripper_events
        global _record_start_time
        global _record_output_path
        global _record_interval
        global _record_arm_name
        global _record_gripper_name
        global _record_freedrive

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

            # ==================================================
            # names
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

            if not arm_name:
                raise ValueError(
                    "arm_name 不可為空"
                )

            if not gripper_name:
                raise ValueError(
                    "gripper_name 不可為空"
                )

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

            full_output_path = (
                _normalize_output_path(
                    output_path
                )
            )

            _ensure_directory(
                full_output_path
            )

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

            _record_trajectory = []

            _record_gripper_events = []

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
                },
            )

    except Exception as exc:

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
                    0,

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
def stop_robot_recording(
    arm_name,
    gripper_name,
):
    action = "stop_robot_recording"

    try:
        global _record_thread
        global _record_start_time
        global _record_output_path
        global _record_interval
        global _record_trajectory
        global _record_gripper_events
        global _record_arm_name
        global _record_gripper_name
        global _record_freedrive

        if arm_name is None:
            raise ValueError("arm_name 不可為空")

        if gripper_name is None:
            raise ValueError("gripper_name 不可為空")

        arm_name = str(arm_name).strip().lower()
        gripper_name = str(gripper_name).strip().lower()

        if not arm_name:
            raise ValueError("arm_name 不可為空")

        if not gripper_name:
            raise ValueError("gripper_name 不可為空")

        with _record_lock:

            if not _is_recording():
                raise RuntimeError(
                    "目前沒有正在進行的記錄"
                )

            if arm_name != _record_arm_name:
                raise ValueError(
                    f"arm_name 不符合目前 recording："
                    f"requested={arm_name}, "
                    f"recording={_record_arm_name}"
                )

            if gripper_name != _record_gripper_name:
                raise ValueError(
                    f"gripper_name 不符合目前 recording："
                    f"requested={gripper_name}, "
                    f"recording={_record_gripper_name}"
                )

            thread = _record_thread
            output_path = _record_output_path
            interval = _record_interval
            freedrive = _record_freedrive
            
        # ======================================================
        # Stop Sampling
        # ======================================================

        _record_stop_event.set()

        thread.join(
            timeout=10
        )

        if thread.is_alive():

            raise RuntimeError(
                "錄製執行緒無法在 10 秒內停止"
            )

        # ======================================================
        # Stop Freedrive
        # ======================================================

        if freedrive:

            response = (
                arm_service
                .stop_arm_freedrive(
                    arm_name
                )
            )

            _require_success(
                response,
                "arm_service."
                "stop_arm_freedrive",
            )

        # ======================================================
        # Copy State
        # ======================================================

        with _record_lock:

            trajectory = list(
                _record_trajectory
            )

            gripper_events = list(
                _record_gripper_events
            )

            _record_thread = None

            _record_start_time = None

            _record_output_path = None

            _record_interval = (
                DEFAULT_RECORD_INTERVAL
            )

            _record_trajectory = []

            _record_gripper_events = []

            _record_arm_name = None

            _record_gripper_name = None

            _record_freedrive = False

        # ======================================================
        # Build JSON
        # ======================================================

        duration_seconds = (
            trajectory[-1][
                "t"
            ]
            if trajectory
            else 0.0
        )

        payload = {
            "format_version":
                4,

            "type":
                "arm_gripper_trajectory",

            "arm_name":
                arm_name,

            "gripper_name":
                gripper_name,

            "interval":
                interval,

            "freedrive":
                freedrive,

            "gripper_sampling":
                not freedrive,

            "gripper_event_recording":
                True,

            "sample_count":
                len(
                    trajectory
                ),

            "gripper_event_count":
                len(
                    gripper_events
                ),

            "duration_seconds":
                duration_seconds,

            "trajectory":
                trajectory,

            "gripper_events":
                gripper_events,
        }

        # ======================================================
        # Save
        # ======================================================

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as output_file:

            json.dump(
                payload,
                output_file,
                ensure_ascii=False,
                indent=2,
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

                "output_path":
                    _relative_path(
                        output_path
                    ),

                "absolute_path":
                    output_path,

                "sample_count":
                    len(
                        trajectory
                    ),

                "gripper_event_count":
                    len(
                        gripper_events
                    ),

                "duration_seconds":
                    duration_seconds,

                "freedrive":
                    freedrive,
            },
        )

    except Exception as exc:

        logger.exception(
            "stop_robot_recording failed"
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
                        len(
                            _record_trajectory
                        ),

                    "gripper_event_count":
                        len(
                            _record_gripper_events
                        ),

                    "elapsed_seconds":
                        elapsed,
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
# Trajectory Load / Parse
# ============================================================

def _load_trajectory(
    input_path,
):
    full_path = (
        _normalize_input_path(
            input_path
        )
    )

    with open(
        full_path,
        "r",
        encoding="utf-8",
    ) as input_file:

        payload = (
            json.load(
                input_file
            )
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "trajectory JSON 格式錯誤"
        )

    trajectory = (
        payload.get(
            "trajectory"
        )
    )

    if (
        not isinstance(
            trajectory,
            list,
        )
        or not trajectory
    ):
        raise ValueError(
            "trajectory 不可為空"
        )

    return (
        full_path,
        payload,
        trajectory,
    )


def _extract_joint_trajectory(
    trajectory,
):
    result = []

    for index, entry in enumerate(
        trajectory
    ):

        try:

            joints = (
                entry[
                    "arm"
                ][
                    "joints"
                ]
            )

        except (
            KeyError,
            TypeError,
        ) as exc:

            raise ValueError(
                f"trajectory[{index}] "
                f"缺少 arm.joints"
            ) from exc

        if not isinstance(
            joints,
            list,
        ):

            raise ValueError(
                f"trajectory[{index}]."
                f"arm.joints 必須是 list"
            )

        result.append(
            [
                float(
                    value
                )
                for value
                in joints
            ]
        )

    return result


# ============================================================
# Legacy Gripper Parse
# ============================================================

def _get_gripper_target(
    gripper_data,
):
    if not isinstance(
        gripper_data,
        dict,
    ):
        return None

    requested = (
        gripper_data.get(
            "requested_position"
        )
    )

    if requested is not None:

        return int(
            requested
        )

    position = (
        gripper_data.get(
            "position"
        )
    )

    if position is not None:

        return int(
            position
        )

    return None


def _extract_legacy_gripper_events(
    trajectory,
    position_threshold=2,
):
    events = []

    previous_target = None

    for entry in trajectory:

        gripper = (
            entry.get(
                "gripper"
            )
        )

        target = (
            _get_gripper_target(
                gripper
            )
        )

        if target is None:
            continue

        if (
            previous_target is None
            or abs(
                target
                - previous_target
            )
            >= position_threshold
        ):

            events.append({
                "t":
                    float(
                        entry.get(
                            "t",
                            0.0,
                        )
                    ),

                "action":
                    "move",

                "position":
                    target,

                "speed":
                    gripper.get(
                        "speed"
                    ),

                "force":
                    gripper.get(
                        "force"
                    ),
            })

            previous_target = (
                target
            )

    return events


def _load_gripper_events(
    payload,
    trajectory,
):
    events = (
        payload.get(
            "gripper_events"
        )
    )

    # ========================================================
    # New Format v4
    # ========================================================

    if isinstance(
        events,
        list,
    ):

        result = []

        for index, event in enumerate(
            events
        ):

            if not isinstance(
                event,
                dict,
            ):
                continue

            position = (
                event.get(
                    "position"
                )
            )

            if position is None:

                logger.warning(
                    "gripper_events[%s] "
                    "missing position",
                    index,
                )

                continue

            result.append({
                "t":
                    float(
                        event.get(
                            "t",
                            0.0,
                        )
                    ),

                "action":
                    event.get(
                        "action",
                        "move",
                    ),

                "position":
                    int(
                        position
                    ),

                "speed":
                    event.get(
                        "speed"
                    ),

                "force":
                    event.get(
                        "force"
                    ),
            })

        result.sort(
            key=lambda item:
                item[
                    "t"
                ]
        )

        return result

    # ========================================================
    # Legacy Format
    # ========================================================

    return (
        _extract_legacy_gripper_events(
            trajectory
        )
    )


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

    try:

        (
            _,
            payload,
            trajectory,
        ) = (
            _load_trajectory(
                input_path
            )
        )

        recorded_interval = (
            float(
                payload.get(
                    "interval",
                    DEFAULT_RECORD_INTERVAL,
                )
            )
        )

        joint_trajectory = (
            _extract_joint_trajectory(
                trajectory
            )
        )

        gripper_events = (
            _load_gripper_events(
                payload=
                    payload,

                trajectory=
                    trajectory,
            )
        )

        if not joint_trajectory:

            raise RuntimeError(
                "沒有可播放的 arm trajectory"
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

    except Exception:

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

        _playback_stop_event.clear()


# ============================================================
# START PLAYBACK
# ============================================================

def start_robot_playback(
    input_path,
    arm_name,
    gripper_name,
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

        with _playback_lock:

            if _is_playing():

                raise RuntimeError(
                    "playback already in progress"
                )

            if _is_recording():

                raise RuntimeError(
                    "recording is currently running"
                )

            (
                full_path,
                payload,
                _,
            ) = (
                _load_trajectory(
                    input_path
                )
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