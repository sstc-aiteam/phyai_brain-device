"""LeRobot robot playback and gripper event execution."""
import logging
import threading
import time
from services import arm_service, gripper_service
from utils.response import error, success
from recording_training_replay.replay.lerobot_dataset import (
    _normalize_input_path, _parquet_episode_details, read_dataset_info,
)

MODULE = "recording_training_replay"
logger = logging.getLogger(__name__)

def _service_data(response, action):
    if not isinstance(response, dict) or response.get("status") != "success" or response.get("result") is False:
        message = response.get("message", "service call failed") if isinstance(response, dict) else "invalid response"
        raise RuntimeError(f"{action}: {message}")
    return response.get("data") or {}


def _require_replay_arm(arm_name):
    data = _service_data(
        arm_service.get_arm_status(arm_name), "get_arm_status"
    )
    arms = data.get("arms") or []
    if not arms:
        raise ValueError(f"arm_service 找不到 arm：{arm_name}")
    arm = arms[0]
    if not (arm.get("status") or {}).get("connected", False):
        raise RuntimeError(f"arm '{arm_name}' 尚未連線")
    return arm


_playback_lock = threading.RLock()
_playback_stop_event = threading.Event()
_playback_thread = None
_playback_input_path = None
_playback_arm_name = None
_playback_gripper_name = None
_playback_episode_index = None
_playback_start_time = None
_playback_elapsed_seconds = 0.0
_playback_phase = "idle"
_playback_error = None
_playback_completed = False


def _is_playing():
    return bool(_playback_thread and _playback_thread.is_alive())


def _move_replay_gripper(
    gripper_name, event, action="move_gripper", wait=False,
):
    _service_data(gripper_service.move_gripper(
        gripper_name,
        event["position"],
        speed=event.get("speed"),
        force=event.get("force"),
        wait=wait,
    ), action)


def _execute_replay_gripper_event(arm_name, gripper_name, event, action,
                                  reconnect_arm=False):
    _move_replay_gripper(gripper_name, event, action, wait=True)
    if reconnect_arm:
        _service_data(
            arm_service.reconnect_arm(arm_name),
            f"reconnect_arm(after gripper frame {event['frame_index']})",
        )


def _split_replay_plan(trajectory, gripper_events, interval):
    """Build an ordered arm/gripper plan before touching robot hardware."""
    if interval <= 0:
        raise ValueError("Replay interval 必須大於 0")
    if not trajectory:
        raise ValueError("Replay trajectory 不可為空")

    last_frame = len(trajectory) - 1
    indexed_events = []
    for order, event in enumerate(gripper_events):
        event_frame = event.get("frame_index")
        if event_frame is None:
            event_frame = round(float(event.get("t", 0.0)) / interval)
        event_frame = max(0, min(int(event_frame), last_frame))
        indexed_events.append((event_frame, order, dict(event)))
    indexed_events.sort(key=lambda item: (item[0], item[1]))

    playback_events = []
    for event_frame, _, event in indexed_events:
        event["frame_index"] = event_frame
        if event_frame > 0:
            playback_events.append(event)

    steps = []
    # Every ServoJ segment starts with the last pose already reached.  This
    # anchor sample gives a newly started real-time stream one stable cycle
    # before it advances to the next recorded frame.
    segment_start = 0
    for event in playback_events:
        event_frame = event["frame_index"]
        if segment_start < event_frame:
            steps.append({
                "type": "arm",
                "start_frame": segment_start,
                "end_frame": event_frame,
                "trajectory": trajectory[segment_start:event_frame + 1],
            })
            segment_start = event_frame
        steps.append({"type": "gripper", "event": event})

    if segment_start < last_frame or not steps:
        steps.append({
            "type": "arm",
            "start_frame": segment_start,
            "end_frame": last_frame,
            "trajectory": trajectory[segment_start:],
        })
    return {"steps": steps}


def _play_replay_plan(plan, arm_name, gripper_name, interval,
                      speed, acceleration, reconnect_after_gripper=False,
                      move_first_segment_to_start=True,
                      move_to_start_speed=None,
                      move_to_start_acceleration=None):
    """Execute the pre-split plan, pausing arm playback at gripper frames."""
    plan_summary = [
        (
            f"arm:{step['start_frame']}-{step['end_frame']}"
            if step["type"] == "arm"
            else f"gripper:{step['event']['frame_index']}="
                 f"{step['event']['position']:.3f}"
        )
        for step in plan["steps"]
    ]
    logger.info("Replay plan: %s", " -> ".join(plan_summary))

    first_arm_segment = True
    for step in plan["steps"]:
        if _playback_stop_event.is_set():
            return
        if step["type"] == "arm":
            segment = step["trajectory"]
            max_joint_delta = max(
                (
                    max(abs(current - previous) for previous, current in zip(a, b))
                    for a, b in zip(segment, segment[1:])
                ),
                default=0.0,
            )
            try:
                use_move_to_start = bool(
                    first_arm_segment and move_first_segment_to_start
                )
                _service_data(arm_service.move_arm_joint_trajectory(
                    arm_name, segment, dt=interval, speed=speed,
                    acceleration=acceleration, wait=True,
                    move_to_start=use_move_to_start,
                    move_to_start_speed=move_to_start_speed,
                    move_to_start_acceleration=move_to_start_acceleration,
                ), "move_arm_joint_trajectory(segment)")
            except Exception as exc:
                raise RuntimeError(
                    "Replay arm segment failed: "
                    f"frames={step['start_frame']}-{step['end_frame']}, "
                    f"points={len(segment)}, interval={interval:.4f}, "
                    f"max_joint_delta={max_joint_delta:.6f}, "
                    f"move_to_start={use_move_to_start}; {exc}"
                ) from exc
            first_arm_segment = False
            continue
        if gripper_name:
            _execute_replay_gripper_event(
                arm_name, gripper_name, step["event"],
                f"move_gripper(frame {step['event']['frame_index']})",
                reconnect_arm=reconnect_after_gripper,
            )


def _playback_worker(input_path, episode_index, arm_name, gripper_name,
                     episode_loader, speed, acceleration, move_to_start,
                     move_to_start_speed, move_to_start_acceleration):
    global _playback_thread, _playback_start_time, _playback_phase
    global _playback_elapsed_seconds, _playback_error, _playback_completed
    try:
        with _playback_lock:
            _playback_phase = "loading"
        dataset_path, interval, trajectory, gripper_events = episode_loader(
            input_path, episode_index
        )
        if not trajectory:
            raise ValueError(f"episode {episode_index} 沒有可播放的 trajectory")
        with _playback_lock:
            _playback_phase = "validating_arm"
        arm = _require_replay_arm(arm_name)
        info = read_dataset_info(dataset_path) or {}
        if info.get("robot_type") != arm.get("driver"):
            raise ValueError("dataset robot_type 與選擇的 arm 不相容")
        # Do not query gripper status before motion.  The e-Series status path
        # executes a custom URScript and can replace the RTDE control program.
        # A selected gripper is validated by its first real path event instead.
        reconnect_after_gripper = bool(gripper_name)
        replay_plan = _split_replay_plan(
            trajectory,
            gripper_events if gripper_name else [],
            interval,
        )
        # frame 0 is the recorded baseline, not an in-path state transition.
        # Re-sending it here would run Robotiq URScript immediately before
        # ServoJ and can make the RTDE stream fail at sample index 1.
        if _playback_stop_event.is_set():
            return
        with _playback_lock:
            _playback_start_time = time.monotonic()
            _playback_phase = "playing"

        _play_replay_plan(
            replay_plan, arm_name, gripper_name, interval, speed, acceleration,
            reconnect_after_gripper=reconnect_after_gripper,
            move_first_segment_to_start=move_to_start,
            move_to_start_speed=move_to_start_speed,
            move_to_start_acceleration=move_to_start_acceleration,
        )
        if not _playback_stop_event.is_set():
            with _playback_lock:
                _playback_completed = True
                _playback_phase = "completed"
    except Exception as exc:
        with _playback_lock:
            _playback_error = f"{type(exc).__name__}: {exc}"
            _playback_phase = "failed"
    finally:
        with _playback_lock:
            if _playback_start_time is not None:
                _playback_elapsed_seconds = (
                    time.monotonic() - _playback_start_time
                )
            _playback_thread = None
        _playback_stop_event.clear()


def start_robot_playback(input_path, arm_name, gripper_name=None,
                         episode_index=0, speed=None, acceleration=None,
                         lookahead_time=0.1, gain=300, move_to_start=True,
                         move_to_start_speed=None,
                         move_to_start_acceleration=None,
                         dataset_format=None):
    del lookahead_time, gain
    global _playback_thread, _playback_input_path, _playback_arm_name
    global _playback_gripper_name, _playback_episode_index, _playback_phase
    global _playback_error, _playback_completed, _playback_start_time
    global _playback_elapsed_seconds
    action = "start_robot_playback"
    try:
        with _playback_lock:
            if _is_playing():
                raise RuntimeError("Replay 已在進行中")
        normalized_path = _normalize_input_path(input_path)
        info = read_dataset_info(normalized_path) or {}
        if info.get("codebase_version") != "v3.0" or dataset_format not in (None, "lerobot_v3"):
            raise ValueError("此頁面僅支援 LeRobot v3 回放")
        from recording_training_replay.replay.lerobot_episode_loader import load_lerobot_v3_episode
        loader = load_lerobot_v3_episode
        detail = next((item for item in _parquet_episode_details(normalized_path, info)
                       if item["episode_index"] == int(episode_index)), None)
        metadata = (detail or {}).get("metadata", {})
        if metadata.get("device_mapping_error") and not gripper_name:
            raise ValueError(metadata["device_mapping_error"])
        if metadata.get("device_source") != "inferred_from_configuration":
            if arm_name and metadata.get("arm_name") and arm_name != metadata["arm_name"]:
                raise ValueError("指定手臂與 episode 錄製手臂不一致")
            if gripper_name and "gripper_name" in metadata and gripper_name != metadata["gripper_name"]:
                raise ValueError("指定夾爪與 episode 錄製夾爪不一致")
        arm_name = arm_name or metadata.get("arm_name")
        # Only apply an inferred gripper to its corresponding robot.
        if arm_name == metadata.get("arm_name"):
            gripper_name = gripper_name or metadata.get("gripper_name")
        arm_name = str(arm_name or "").strip().lower()
        gripper_name = str(gripper_name or "").strip().lower() or None
        if not arm_name:
            raise ValueError("arm_name 不可為空")
        dataset_path = _normalize_input_path(input_path)
        with _playback_lock:
            _playback_stop_event.clear()
            _playback_input_path = dataset_path
            _playback_arm_name = arm_name
            _playback_gripper_name = gripper_name
            _playback_episode_index = int(episode_index)
            _playback_start_time = None
            _playback_elapsed_seconds = 0.0
            _playback_phase = "starting"
            _playback_error = None
            _playback_completed = False
            _playback_thread = threading.Thread(
                target=_playback_worker,
                args=(dataset_path, int(episode_index), arm_name, gripper_name,
                      loader, speed, acceleration, bool(move_to_start),
                      move_to_start_speed,
                      move_to_start_acceleration),
                name="robot-replay", daemon=True,
            )
            _playback_thread.start()
        return success(MODULE, action, data={
            "playing": True, "input_path": dataset_path,
            "episode_index": int(episode_index), "arm_name": arm_name,
        })
    except Exception as exc:
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def stop_robot_playback(arm_name=None, gripper_name=None):
    action = "stop_robot_playback"
    try:
        with _playback_lock:
            thread = _playback_thread
            playing_arm = _playback_arm_name
        if not thread or not thread.is_alive():
            raise RuntimeError("目前沒有正在 Replay 的 trajectory")
        _playback_stop_event.set()
        _service_data(arm_service.stop_arm(playing_arm), "stop_arm")
        thread.join(timeout=10.0)
        return success(MODULE, action, data={"playing": False})
    except Exception as exc:
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


def get_robot_playback_status():
    with _playback_lock:
        playing = _is_playing()
        elapsed = (
            time.monotonic() - _playback_start_time
            if playing and _playback_start_time is not None
            else _playback_elapsed_seconds
        )
        return success(MODULE, "get_robot_playback_status", data={
            "playing": playing,
            "input_path": _playback_input_path,
            "episode_index": _playback_episode_index,
            "arm_name": _playback_arm_name,
            "gripper_name": _playback_gripper_name,
            "elapsed_seconds": elapsed,
            "phase": _playback_phase,
            "completed": _playback_completed,
            "error": _playback_error,
        })

