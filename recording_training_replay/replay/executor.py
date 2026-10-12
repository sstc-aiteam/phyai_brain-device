"""Replay execution with injected drivers; no dependency on services."""
import logging
import threading
import time
from recording_training_replay.replay.episode_loader import load_raw_episode
from recording_training_replay.replay.dataset import raw_episode_details
logger = logging.getLogger(__name__)

def split_replay_plan(trajectory, gripper_events, interval):
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
    sample_times = getattr(trajectory, "sample_times", None)
    if sample_times is not None:
        for step in steps:
            if step["type"] == "arm":
                start, end = step["start_frame"], step["end_frame"]
                step["sample_times"] = [t - sample_times[start] for t in sample_times[start:end + 1]]
    return {"steps": steps}


class ReplayExecutor:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.details = {"playing": False, "phase": "idle", "completed": False,
                        "error": None, "elapsed_seconds": 0.0, "input_path": None,
                        "episode_index": None, "arm_name": None, "gripper_name": None}
        self.started = None

    def execute(self, command, **kwargs):
        return {"start": self.start, "stop": self.stop, "status": self.status}[command](**kwargs)

    def status(self):
        with self.lock:
            result = dict(self.details)
            result["playing"] = bool(self.thread and self.thread.is_alive())
            if result["playing"] and self.started is not None:
                result["elapsed_seconds"] = time.monotonic() - self.started
            return result

    def start(self, input_path, episode_index, arm_name, driver, arm,
              gripper_name=None, gripper=None, gripper_params=None,
              episode_loader=load_raw_episode, **motion):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Replay 已在進行中")
            path, interval, trajectory, events = episode_loader(input_path, episode_index)
            episode = next((item for item in raw_episode_details(path)
                            if item["episode_index"] == episode_index), None)
            if episode is None or episode["metadata"].get("robot_type") != driver:
                raise ValueError("episode robot_type 與選擇的 arm 不相容")
            plan = split_replay_plan(trajectory, events if gripper else [], interval)
            self.arm, self.gripper = arm, gripper
            self.gripper_params = dict(gripper_params or {})
            self.stop_event.clear()
            self.started = None
            self.details.update(input_path=path, episode_index=episode_index,
                                arm_name=arm_name, gripper_name=gripper_name,
                                phase="starting", completed=False, error=None, elapsed_seconds=0.0)
            self.thread = threading.Thread(target=self._run, args=(plan, interval, motion),
                                           name="robot-replay", daemon=True)
            try:
                self.thread.start()
            except Exception:
                self.thread = None
                self.details["phase"] = "failed"
                raise
            return {"playing": True, "input_path": path,
                    "episode_index": episode_index, "arm_name": arm_name}

    def _run(self, plan, interval, motion):
        try:
            with self.lock:
                self.started = time.monotonic()
                self.details["phase"] = "playing"
            self.play_plan(plan, interval, reconnect_after_gripper=self.gripper is not None, **motion)
            with self.lock:
                self.details.update(completed=not self.stop_event.is_set(),
                                    phase="stopped" if self.stop_event.is_set() else "completed")
        except Exception as exc:
            with self.lock:
                if self.stop_event.is_set():
                    self.details.update(error=None, phase="stopped", completed=False)
                else:
                    self.details.update(error=f"{type(exc).__name__}: {exc}", phase="failed")
        finally:
            with self.lock:
                self.details["elapsed_seconds"] = time.monotonic() - self.started

    def stop(self):
        with self.lock:
            thread = self.thread
            if not thread or not thread.is_alive():
                raise RuntimeError("目前沒有正在 Replay 的 trajectory")
            self.stop_event.set()
        if self.arm.stop_arm() is False:
            raise RuntimeError("stop_arm failed")
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise TimeoutError("Replay thread 無法停止")
        return {"playing": False}

    def play_plan(self, plan, interval, speed=0.1, acceleration=0.1,
                  reconnect_after_gripper=False, move_first_segment_to_start=True,
                  move_to_start_speed=None, move_to_start_acceleration=None):
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
            if self.stop_event.is_set():
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
                    timed_segment = list(zip(step["sample_times"], segment)) if "sample_times" in step else segment
                    result = self.arm.move_arm_joint_trajectory(
                        joint_trajectory=timed_segment, dt=interval, speed=speed,
                        acceleration=acceleration, wait=True,
                        move_to_start=use_move_to_start,
                        move_to_start_speed=move_to_start_speed,
                        move_to_start_acceleration=move_to_start_acceleration)
                    if result is not True:
                        raise RuntimeError("arm trajectory did not complete")
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
            if self.gripper is not None:
                event = step["event"]
                params = dict(self.gripper_params)
                for key in ("speed", "force"):
                    if event.get(key) is not None:
                        params[key] = event[key]
                params["wait"] = True
                if not self.gripper.move_gripper(position=event["position"], **params):
                    raise RuntimeError(f"gripper failed at frame {event['frame_index']}")
                if reconnect_after_gripper and self.arm.reconnect_arm() is False:
                    raise RuntimeError("arm reconnect failed after gripper")
