import atexit
import logging
import math
import threading
import time

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


logger = logging.getLogger(__name__)


# ============================================================
# Driver defaults
# ============================================================

DEFAULT_TRAJECTORY_DT = 0.1
DEFAULT_SPEED = 0.1
DEFAULT_ACCELERATION = 0.1

DEFAULT_JOG_LINEAR_SPEED = 0.05
DEFAULT_JOG_ANGULAR_SPEED = 0.10
DEFAULT_JOG_ACCELERATION = 0.10
DEFAULT_JOG_TIMEOUT = 0.20


# ============================================================
# UR7e hard safety limits
# ============================================================
#
# UR7e is a 6-DOF e-Series arm. Current UR specifications list ±360°
# working range for all six joints. arm_service should still impose narrower
# deployment-specific workspace/speed limits from config.
#
# arm_service may impose narrower, deployment-specific soft limits.
# These limits are the final driver-side guard and must not be widened by the
# service layer.

UR7e_HARD_X_RANGE = (-3.0, 3.0)
UR7e_HARD_Y_RANGE = (-3.0, 3.0)
UR7e_HARD_Z_RANGE = (-3.0, 3.0)

ARM_JOINT_LIMITS = [
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.0 * math.pi, 2.0 * math.pi),
]

MIN_SPEED = 0.01
MAX_SPEED = 3.0

MIN_ACCELERATION = 0.01
MAX_ACCELERATION = 3.0

MIN_STOP_ACCELERATION = 0.10
MAX_STOP_ACCELERATION = 0.50

MIN_TRAJECTORY_DT = 0.001
MAX_TRAJECTORY_DT = 0.50

MIN_SERVO_LOOKAHEAD_TIME = 0.03
MAX_SERVO_LOOKAHEAD_TIME = 0.20

MIN_SERVO_GAIN = 100
MAX_SERVO_GAIN = 2000

MAX_TRAJECTORY_POINTS = 1_000_000


# ============================================================
# Feedback / reached detection
# ============================================================

ARM_POSE_TOLERANCE = 0.005          # m
ARM_ROTATION_TOLERANCE = 0.03       # rad
ARM_JOINT_TOLERANCE = 0.01          # rad
ARM_WAIT_TIMEOUT = 15.0             # s

MOVING_TCP_SPEED_THRESHOLD = 1e-3
MOVING_JOINT_SPEED_THRESHOLD = 1e-3

FINAL_HOLD_SECONDS = 0.20
FEEDBACK_POLL_INTERVAL = 0.05


# ============================================================
# RTDE Receive monitor
# ============================================================

RTDE_RECEIVE_MONITOR_HZ = 20.0
RTDE_RECEIVE_MONITOR_INTERVAL = 1.0 / RTDE_RECEIVE_MONITOR_HZ
RTDE_RECEIVE_STALE_TIMEOUT = 0.5
RTDE_RECEIVE_INITIAL_TIMEOUT = 3.0
RTDE_RECEIVE_MONITOR_JOIN_TIMEOUT = 2.0
RTDE_RECEIVE_ERROR_RETRY_INTERVAL = 1.0


class UR7eDriver:
    """Universal Robots UR7e RTDE arm driver."""

    ARM_DOF = 6

    DRIVER_METADATA = {
        "name": "ur7e",
        "manufacturer": "Universal Robots",
        "model": "UR7e",
        "dof": ARM_DOF,
        "capabilities": [
            "status",
            "pose",
            "joints",
            "move_pose",
            "move_joints",
            "joint_trajectory",
            "pose_trajectory",
            "jog",
            "freedrive",
            "stop",
            "reconnect",
        ],
    }

    def __init__(self, ip):
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError("ip must be a non-empty string")

        self.ip = ip.strip()

        # RTDE channels have independent locks.
        self._rtde_control_lock = threading.RLock()
        self._rtde_receive_lock = threading.RLock()

        # All motion lifecycle transitions are serialized here.
        self._motion_lock = threading.RLock()

        self._rtde_c = None
        self._rtde_r = None

        self._motion_command_id = 0
        self._motion_mode = None
        self._jog_direction = None

        # Receive monitor/cache.
        self._state_lock = threading.RLock()
        self._state_condition = threading.Condition(self._state_lock)
        self._receive_monitor_lock = threading.RLock()
        self._receive_monitor_stop_event = threading.Event()
        self._receive_monitor_thread = None
        self._receive_sequence = 0
        self._receive_last_progress_monotonic = None
        self._latest_state = self._empty_receive_state()

        # Lifecycle.
        self._shutdown_lock = threading.RLock()
        self._shutdown_done = False

        atexit.register(self.shutdown)

    # ========================================================
    # Receive state/cache
    # ========================================================

    def _empty_receive_state(self, error=None):
        return {
            "connected": False,
            "data_valid": False,
            "data_stale": True,
            "sequence": self._receive_sequence,
            "controller_timestamp": None,
            "updated_monotonic": None,
            "last_progress_monotonic": None,
            "data_age_seconds": None,
            "receive_error": error,
            "arm_mode": None,
            "safety_mode": None,
            "emergency_stop": None,
            "protective_stop": None,
            "pose": None,
            "joints": None,
            "tcp_speed": None,
            "joint_speed": None,
        }

    def _reset_receive_state(self, error=None):
        with self._state_condition:
            self._receive_last_progress_monotonic = None
            self._latest_state = self._empty_receive_state(error=error)
            self._state_condition.notify_all()

    def _copy_receive_state_locked(self):
        state = dict(self._latest_state)

        for key in ("pose", "joints", "tcp_speed", "joint_speed"):
            if state.get(key) is not None:
                state[key] = list(state[key])

        last_progress = state.get("last_progress_monotonic")
        if last_progress is not None:
            age = time.monotonic() - last_progress
            state["data_age_seconds"] = age
            state["data_stale"] = age > RTDE_RECEIVE_STALE_TIMEOUT
            if state["data_stale"]:
                state["data_valid"] = False

        return state

    def _start_receive_monitor(self):
        with self._shutdown_lock:
            if self._shutdown_done:
                raise RuntimeError(
                    "UR7eDriver is already shutdown and cannot restart the receive monitor"
                )

        with self._receive_monitor_lock:
            thread = self._receive_monitor_thread
            if thread is not None and thread.is_alive():
                return True

            self._receive_monitor_stop_event.clear()
            thread = threading.Thread(
                target=self._receive_monitor_loop,
                name=f"UR7eReceiveMonitor-{self.ip}",
                daemon=True,
            )
            self._receive_monitor_thread = thread
            thread.start()

        logger.info("[UR7e] RTDE receive monitor started: %s", self.ip)
        return True

    def _stop_receive_monitor(self):
        with self._receive_monitor_lock:
            thread = self._receive_monitor_thread
            if thread is None:
                return True
            self._receive_monitor_stop_event.set()

        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=RTDE_RECEIVE_MONITOR_JOIN_TIMEOUT)

        if thread.is_alive():
            raise RuntimeError(
                "RTDE Receive monitor did not stop; refusing to disconnect receive object"
            )

        with self._receive_monitor_lock:
            if self._receive_monitor_thread is thread:
                self._receive_monitor_thread = None

        logger.info("[UR7e] RTDE receive monitor stopped: %s", self.ip)
        return True

    def _receive_monitor_loop(self):
        """
        The only owner that performs I/O on RTDEReceiveInterface.

        All public/read/feedback methods consume the cache through
        _get_receive_state(); they never touch RTDEReceiveInterface directly.
        """
        previous_controller_timestamp = None

        try:
            while not self._receive_monitor_stop_event.is_set():
                cycle_start = time.monotonic()

                try:
                    with self._rtde_receive_lock:
                        rtde_r = self._get_receive_for_monitor()

                        controller_timestamp = float(rtde_r.getTimestamp())
                        pose = list(rtde_r.getActualTCPPose())
                        joints = list(rtde_r.getActualQ())

                        tcp_speed = self._read_optional_rtde_vector(
                            rtde_r,
                            "getActualTCPSpeed",
                            expected_length=6,
                        )
                        joint_speed = self._read_optional_rtde_vector(
                            rtde_r,
                            "getActualQd",
                            expected_length=self.ARM_DOF,
                        )

                        arm_mode = self._read_optional_rtde_value(
                            rtde_r,
                            "getRobotMode",
                        )
                        safety_mode = self._read_optional_rtde_value(
                            rtde_r,
                            "getSafetyMode",
                        )
                        emergency_stop = self._read_optional_rtde_bool(
                            rtde_r,
                            "isEmergencyStopped",
                        )
                        protective_stop = self._read_optional_rtde_bool(
                            rtde_r,
                            "isProtectiveStopped",
                        )

                    now = time.monotonic()
                    timestamp_progressed = (
                        previous_controller_timestamp is None
                        or controller_timestamp > previous_controller_timestamp
                    )

                    if timestamp_progressed:
                        previous_controller_timestamp = controller_timestamp
                        self._receive_last_progress_monotonic = now
                        self._receive_sequence += 1

                    last_progress = self._receive_last_progress_monotonic
                    data_age = (
                        None
                        if last_progress is None
                        else now - last_progress
                    )
                    data_stale = (
                        data_age is None
                        or data_age > RTDE_RECEIVE_STALE_TIMEOUT
                    )

                    with self._state_condition:
                        self._latest_state = {
                            "connected": True,
                            "data_valid": not data_stale,
                            "data_stale": data_stale,
                            "sequence": self._receive_sequence,
                            "controller_timestamp": controller_timestamp,
                            "updated_monotonic": now,
                            "last_progress_monotonic": last_progress,
                            "data_age_seconds": data_age,
                            "receive_error": (
                                "controller timestamp stale"
                                if data_stale
                                else None
                            ),
                            "arm_mode": arm_mode,
                            "safety_mode": safety_mode,
                            "emergency_stop": emergency_stop,
                            "protective_stop": protective_stop,
                            "pose": pose,
                            "joints": joints,
                            "tcp_speed": tcp_speed,
                            "joint_speed": joint_speed,
                        }
                        self._state_condition.notify_all()

                except Exception as exc:
                    error_message = str(exc)
                    logger.exception(
                        "[UR7e] RTDE receive monitor read failed: %s",
                        exc,
                    )

                    with self._state_condition:
                        self._latest_state = self._empty_receive_state(
                            error=error_message
                        )
                        self._state_condition.notify_all()

                    # Receive reconnection is owned only by this monitor.
                    with self._rtde_receive_lock:
                        self._disconnect_receive_for_monitor()

                    previous_controller_timestamp = None
                    self._receive_last_progress_monotonic = None

                    if self._receive_monitor_stop_event.wait(
                        RTDE_RECEIVE_ERROR_RETRY_INTERVAL
                    ):
                        break

                elapsed = time.monotonic() - cycle_start
                remaining = RTDE_RECEIVE_MONITOR_INTERVAL - elapsed
                if remaining > 0:
                    if self._receive_monitor_stop_event.wait(remaining):
                        break

        finally:
            with self._rtde_receive_lock:
                self._disconnect_receive_for_monitor()

            self._reset_receive_state(
                error="RTDE Receive monitor stopped"
            )

    @staticmethod
    def _read_optional_rtde_value(rtde_r, method_name):
        method = getattr(rtde_r, method_name, None)
        if not callable(method):
            return None
        try:
            return method()
        except Exception:
            logger.debug(
                "[UR7e] optional RTDE receive method unavailable: %s",
                method_name,
                exc_info=True,
            )
            return None

    @classmethod
    def _read_optional_rtde_bool(cls, rtde_r, method_name):
        value = cls._read_optional_rtde_value(rtde_r, method_name)
        return None if value is None else bool(value)

    @classmethod
    def _read_optional_rtde_vector(
        cls,
        rtde_r,
        method_name,
        expected_length,
    ):
        value = cls._read_optional_rtde_value(rtde_r, method_name)
        if value is None:
            return None
        try:
            value = list(value)
        except TypeError:
            return None
        if len(value) != expected_length:
            return None
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None

    def _wait_for_receive_state(
        self,
        timeout=RTDE_RECEIVE_INITIAL_TIMEOUT,
        after_sequence=None,
    ):
        timeout = self._check_positive_number("timeout", timeout)

        self._start_receive_monitor()
        deadline = time.monotonic() + timeout

        with self._state_condition:
            while True:
                state = self._copy_receive_state_locked()

                sequence_valid = (
                    after_sequence is None
                    or state["sequence"] > after_sequence
                )
                age = state.get("data_age_seconds")
                age_valid = (
                    age is not None
                    and age <= RTDE_RECEIVE_STALE_TIMEOUT
                )

                if (
                    state["connected"]
                    and state["data_valid"]
                    and not state["data_stale"]
                    and sequence_valid
                    and age_valid
                ):
                    return state

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "RTDE Receive has no valid fresh data: "
                        f"connected={state['connected']}, "
                        f"data_valid={state['data_valid']}, "
                        f"data_stale={state['data_stale']}, "
                        f"age={age}, "
                        f"sequence={state['sequence']}, "
                        f"error={state.get('receive_error')}"
                    )

                self._state_condition.wait(timeout=remaining)

    def _get_receive_state(self):
        return self._wait_for_receive_state(
            timeout=RTDE_RECEIVE_INITIAL_TIMEOUT
        )

    # ========================================================
    # RTDE Receive connection - monitor only
    # ========================================================

    def _get_receive_for_monitor(self):
        if self._rtde_r is None:
            logger.info(
                "[UR7e] connect RTDEReceiveInterface: %s",
                self.ip,
            )
            self._rtde_r = RTDEReceiveInterface(self.ip)
        return self._rtde_r

    def _disconnect_receive_for_monitor(self):
        rtde_r = self._rtde_r
        self._rtde_r = None

        if rtde_r is not None:
            try:
                rtde_r.disconnect()
            except Exception:
                logger.exception(
                    "[UR7e] RTDE receive disconnect failed"
                )

        return True

    # ========================================================
    # RTDE Control gateway
    # ========================================================

    def _get_control_for_gateway(self):
        with self._rtde_control_lock:
            if self._rtde_c is None:
                logger.info(
                    "[UR7e] connect RTDEControlInterface: %s",
                    self.ip,
                )
                self._rtde_c = RTDEControlInterface(self.ip)
            return self._rtde_c

    def _ensure_control_ready(self):
        with self._rtde_control_lock:
            rtde_c = self._get_control_for_gateway()

            if not rtde_c.isConnected():
                logger.warning(
                    "[UR7e] RTDE Control disconnected; reconnecting"
                )
                if not bool(rtde_c.reconnect()):
                    raise ConnectionError(
                        "RTDE Control reconnect failed"
                    )

            if not rtde_c.isProgramRunning():
                logger.warning(
                    "[UR7e] RTDE control script is not running; reuploading"
                )
                if not bool(rtde_c.reuploadScript()):
                    raise RuntimeError(
                        "RTDE control script reupload failed"
                    )
                time.sleep(0.2)

            if not rtde_c.isConnected():
                raise ConnectionError(
                    "RTDE Control is not connected"
                )

            if not rtde_c.isProgramRunning():
                raise RuntimeError(
                    "RTDE control script is not running after reupload"
                )

        return True

    def _call_control(
        self,
        method_name,
        *args,
        ensure_ready=True,
        **kwargs,
    ):
        if not isinstance(method_name, str) or not method_name.strip():
            raise ValueError("method_name must be a non-empty string")
        if not isinstance(ensure_ready, bool):
            raise ValueError("ensure_ready must be bool")

        method_name = method_name.strip()

        with self._rtde_control_lock:
            if ensure_ready:
                self._ensure_control_ready()

            rtde_c = self._get_control_for_gateway()
            method = getattr(rtde_c, method_name, None)

            if not callable(method):
                raise RuntimeError(
                    f"RTDE Control method not found: {method_name}"
                )

            try:
                return method(*args, **kwargs)
            except Exception as exc:
                logger.exception(
                    "[UR7e] RTDE control method failed: method=%s error=%s",
                    method_name,
                    exc,
                )
                raise

    def _disconnect_control(self):
        with self._rtde_control_lock:
            rtde_c = self._rtde_c
            self._rtde_c = None

            if rtde_c is not None:
                try:
                    rtde_c.disconnect()
                except Exception:
                    logger.exception(
                        "[UR7e] RTDE control disconnect failed"
                    )

        return True

    def _get_program_running_status(self):
        """
        Status-only best effort.

        Do not create a new Control connection just to answer get_arm_status().
        If the Control channel has not been used yet, return None.
        """
        with self._rtde_control_lock:
            rtde_c = self._rtde_c
            if rtde_c is None:
                return None
            try:
                if not rtde_c.isConnected():
                    return False
                return bool(rtde_c.isProgramRunning())
            except Exception:
                logger.debug(
                    "[UR7e] unable to read control program status",
                    exc_info=True,
                )
                return None

    # ========================================================
    # Motion lifecycle
    # ========================================================

    def _generate_command_id(self):
        self._motion_command_id += 1
        return self._motion_command_id

    def _check_command_id(self, command_id):
        with self._motion_lock:
            return command_id == self._motion_command_id

    def _set_motion_mode(self, mode):
        allowed_modes = {
            None,
            "move_j",
            "move_l",
            "servo_j",
            "servo_l",
            "speed_l",
            "freedrive",
        }
        if mode not in allowed_modes:
            raise ValueError(f"invalid motion mode: {mode}")
        self._motion_mode = mode

    def _clear_motion_mode(
        self,
        command_id=None,
        expected_mode=None,
    ):
        with self._motion_lock:
            if (
                command_id is not None
                and command_id != self._motion_command_id
            ):
                return False

            if (
                expected_mode is not None
                and self._motion_mode != expected_mode
            ):
                return False

            self._motion_mode = None
            return True

    def _stop_motion(self, acceleration=None):
        if acceleration is None:
            acceleration = DEFAULT_JOG_ACCELERATION

        motion_mode = self._motion_mode
        if motion_mode is None:
            return True

        acceleration = self._check_value_range(
            "stop_acceleration",
            acceleration,
            MIN_STOP_ACCELERATION,
            MAX_STOP_ACCELERATION,
        )

        logger.info(
            "[UR7e] stop current motion mode=%s",
            motion_mode,
        )

        if motion_mode == "servo_j":
            self._call_control("servoStop")

        elif motion_mode == "servo_l":
            # servoStop() stops servoJ/servoL control loops in ur_rtde.
            self._call_control("servoStop")

        elif motion_mode == "speed_l":
            self._call_control(
                "speedStop",
                float(acceleration),
            )

        elif motion_mode == "freedrive":
            self._call_control("endFreedriveMode")

        elif motion_mode == "move_j":
            self._call_control(
                "stopJ",
                float(acceleration),
            )

        elif motion_mode == "move_l":
            self._call_control(
                "stopL",
                float(acceleration),
            )

        else:
            raise RuntimeError(
                f"unknown motion mode: {motion_mode}"
            )

        self._motion_mode = None
        if motion_mode == "speed_l":
            self._jog_direction = None

        return True

    def _begin_motion(self, new_mode, stop_acceleration=None):
        if stop_acceleration is None:
            stop_acceleration = DEFAULT_JOG_ACCELERATION

        allowed_modes = {
            "move_j",
            "move_l",
            "servo_j",
            "servo_l",
            "speed_l",
            "freedrive",
        }
        if new_mode not in allowed_modes:
            raise ValueError(f"invalid new motion mode: {new_mode}")

        command_id = self._generate_command_id()
        previous_mode = self._motion_mode

        logger.info(
            "[UR7e] begin motion command_id=%s previous_mode=%s new_mode=%s",
            command_id,
            previous_mode,
            new_mode,
        )

        if previous_mode is not None:
            self._stop_motion(acceleration=stop_acceleration)
            time.sleep(0.05)

        self._set_motion_mode(new_mode)
        return command_id

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _check_positive_number(name, value):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc

        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if number <= 0:
            raise ValueError(f"{name} must be > 0")
        return number

    @staticmethod
    def _check_value_range(name, value, min_value, max_value):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc

        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")

        if not min_value <= number <= max_value:
            raise ValueError(
                f"{name} out of range: {number}; "
                f"allowed {min_value} ~ {max_value}"
            )

        return number

    def _normalize_pose(self, pose):
        if not isinstance(pose, (list, tuple)) or len(pose) != 6:
            raise ValueError(
                "pose must contain 6 values: [x, y, z, rx, ry, rz]"
            )

        normalized = []
        for index, value in enumerate(pose):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"pose[{index}] must be numeric"
                ) from exc

            if not math.isfinite(number):
                raise ValueError(
                    f"pose[{index}] must be finite"
                )

            normalized.append(number)

        return normalized

    def _normalize_joints(self, joints):
        if (
            not isinstance(joints, (list, tuple))
            or len(joints) != self.ARM_DOF
        ):
            raise ValueError(
                f"joints must contain {self.ARM_DOF} values"
            )

        normalized = []
        for index, value in enumerate(joints):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"joints[{index}] must be numeric"
                ) from exc

            if not math.isfinite(number):
                raise ValueError(
                    f"joints[{index}] must be finite"
                )

            normalized.append(number)

        return normalized

    def _check_pose_range(self, pose):
        pose = self._normalize_pose(pose)
        x, y, z = pose[:3]

        for axis, value, allowed in (
            ("x", x, UR7e_HARD_X_RANGE),
            ("y", y, UR7e_HARD_Y_RANGE),
            ("z", z, UR7e_HARD_Z_RANGE),
        ):
            if not allowed[0] <= value <= allowed[1]:
                raise ValueError(
                    f"{axis} out of UR7e hard safety range: {value}; "
                    f"allowed {allowed}"
                )

        return True

    def _check_joint_range(self, joints):
        joints = self._normalize_joints(joints)

        if len(ARM_JOINT_LIMITS) != self.ARM_DOF:
            raise RuntimeError(
                "ARM_JOINT_LIMITS must contain ARM_DOF ranges"
            )

        for index, joint_rad in enumerate(joints):
            min_rad, max_rad = ARM_JOINT_LIMITS[index]
            if not min_rad <= joint_rad <= max_rad:
                raise ValueError(
                    f"J{index + 1} out of hard range: "
                    f"{joint_rad:.4f} rad; "
                    f"allowed {min_rad:.4f} ~ {max_rad:.4f} rad"
                )

        return True

    def _normalize_joint_trajectory(self, joint_trajectory, dt):
        if not isinstance(joint_trajectory, (list, tuple)):
            raise ValueError("joint_trajectory must be list or tuple")
        if not joint_trajectory:
            raise ValueError("joint_trajectory must not be empty")
        if len(joint_trajectory) > MAX_TRAJECTORY_POINTS:
            raise ValueError(
                f"too many trajectory points: {len(joint_trajectory)}; "
                f"max={MAX_TRAJECTORY_POINTS}"
            )

        normalized = []
        previous_time = None

        for index, sample in enumerate(joint_trajectory):
            try:
                if (
                    isinstance(sample, (list, tuple))
                    and len(sample) == 2
                    and isinstance(sample[1], (list, tuple))
                ):
                    sample_time = float(sample[0])
                    joints = sample[1]

                elif isinstance(sample, dict) and "q_arm" in sample:
                    sample_time = float(sample.get("t", index * dt))
                    joints = sample["q_arm"]

                else:
                    sample_time = index * dt
                    joints = sample

                if not math.isfinite(sample_time):
                    raise ValueError("sample time must be finite")
                if sample_time < 0:
                    raise ValueError("sample time must be >= 0")
                if previous_time is not None and sample_time < previous_time:
                    raise ValueError("sample time must be non-decreasing")

                joints = self._normalize_joints(joints)
                self._check_joint_range(joints)

                normalized.append((sample_time, joints))
                previous_time = sample_time

            except Exception as exc:
                raise ValueError(
                    f"invalid joint_trajectory sample {index}: {exc}"
                ) from exc

        return normalized

    def _normalize_pose_trajectory(self, pose_trajectory, dt):
        if not isinstance(pose_trajectory, (list, tuple)):
            raise ValueError("pose_trajectory must be list or tuple")
        if not pose_trajectory:
            raise ValueError("pose_trajectory must not be empty")
        if len(pose_trajectory) > MAX_TRAJECTORY_POINTS:
            raise ValueError(
                f"too many trajectory points: {len(pose_trajectory)}; "
                f"max={MAX_TRAJECTORY_POINTS}"
            )

        normalized = []
        previous_time = None

        for index, sample in enumerate(pose_trajectory):
            try:
                if (
                    isinstance(sample, (list, tuple))
                    and len(sample) == 2
                    and isinstance(sample[1], (list, tuple))
                ):
                    sample_time = float(sample[0])
                    pose = sample[1]

                elif isinstance(sample, dict) and "pose" in sample:
                    sample_time = float(sample.get("t", index * dt))
                    pose = sample["pose"]

                else:
                    sample_time = index * dt
                    pose = sample

                if not math.isfinite(sample_time):
                    raise ValueError("sample time must be finite")
                if sample_time < 0:
                    raise ValueError("sample time must be >= 0")
                if previous_time is not None and sample_time < previous_time:
                    raise ValueError("sample time must be non-decreasing")

                pose = self._normalize_pose(pose)
                self._check_pose_range(pose)

                normalized.append((sample_time, pose))
                previous_time = sample_time

            except Exception as exc:
                raise ValueError(
                    f"invalid pose_trajectory sample {index}: {exc}"
                ) from exc

        return normalized

    # ========================================================
    # Public: connection lifecycle
    # ========================================================

    def reconnect_arm(self):
        """
        Rebuild the RTDE Control channel.

        The Receive monitor is independent and remains active; if Receive itself
        fails, the monitor owns its own reconnect cycle.
        """
        with self._motion_lock:
            logger.warning(
                "[UR7e] reconnecting RTDE Control: %s",
                self.ip,
            )

            self._disconnect_control()
            time.sleep(0.5)
            self._ensure_control_ready()

        self._start_receive_monitor()
        self._get_receive_state()

        logger.info(
            "[UR7e] RTDE Control reconnected; Receive monitor active: %s",
            self.ip,
        )
        return True

    # ========================================================
    # Public: state
    # ========================================================

    def get_arm_pose(self):
        state = self._get_receive_state()
        pose = state.get("pose")
        if pose is None:
            raise RuntimeError("RTDE Receive pose unavailable")
        return list(pose)

    def get_arm_joints(self):
        state = self._get_receive_state()
        joints = state.get("joints")
        if joints is None:
            raise RuntimeError("RTDE Receive joints unavailable")
        return list(joints)

    def get_arm_status(self):
        """
        Return the generic status schema expected by arm_service.

        Required service-facing keys:
            connected, ready, moving, protective_stop, emergency_stop,
            fault, program_running, arm_mode, safety_mode, pose, joints
        """
        try:
            state = self._get_receive_state()
        except Exception as exc:
            return {
                "connected": False,
                "ready": False,
                "moving": None,
                "protective_stop": None,
                "emergency_stop": None,
                "fault": None,
                "program_running": self._get_program_running_status(),
                "arm_mode": None,
                "safety_mode": None,
                "pose": None,
                "joints": None,
                "data_valid": False,
                "data_stale": True,
                "data_age_seconds": None,
                "controller_timestamp": None,
                "sequence": None,
                "receive_error": str(exc),
            }

        emergency_stop = state.get("emergency_stop")
        protective_stop = state.get("protective_stop")

        moving = self._derive_moving(state)
        ready = bool(
            state.get("connected")
            and state.get("data_valid")
            and not state.get("data_stale")
            and emergency_stop is not True
            and protective_stop is not True
        )

        # Generic fault is intentionally conservative.  UR protective/emergency
        # stop are exposed separately, but either means the arm is not in a
        # healthy executable state from the generic service perspective.
        fault = bool(
            emergency_stop is True
            or protective_stop is True
        )

        return {
            "connected": bool(state["connected"]),
            "ready": ready,
            "moving": moving,
            "protective_stop": protective_stop,
            "emergency_stop": emergency_stop,
            "fault": fault,
            "program_running": self._get_program_running_status(),
            "arm_mode": state.get("arm_mode"),
            "safety_mode": state.get("safety_mode"),
            "pose": (
                list(state["pose"])
                if state.get("pose") is not None
                else None
            ),
            "joints": (
                list(state["joints"])
                if state.get("joints") is not None
                else None
            ),
            # Extra low-level diagnostics. arm_service may ignore these.
            "data_valid": bool(state["data_valid"]),
            "data_stale": bool(state["data_stale"]),
            "data_age_seconds": state.get("data_age_seconds"),
            "controller_timestamp": state.get("controller_timestamp"),
            "sequence": state.get("sequence"),
            "receive_error": state.get("receive_error"),
        }

    def _derive_moving(self, state):
        tcp_speed = state.get("tcp_speed")
        joint_speed = state.get("joint_speed")

        if tcp_speed is not None:
            if any(
                abs(float(value)) > MOVING_TCP_SPEED_THRESHOLD
                for value in tcp_speed
            ):
                return True

        if joint_speed is not None:
            if any(
                abs(float(value)) > MOVING_JOINT_SPEED_THRESHOLD
                for value in joint_speed
            ):
                return True

        if tcp_speed is not None or joint_speed is not None:
            return False

        return None

    # ========================================================
    # Public: move pose / joints
    # ========================================================

    def move_arm_pose(
        self,
        x,
        y,
        z,
        rx,
        ry,
        rz,
        speed=None,
        acceleration=None,
        wait=True,
    ):
        if speed is None:
            speed = DEFAULT_SPEED
        if acceleration is None:
            acceleration = DEFAULT_ACCELERATION
        if not isinstance(wait, bool):
            raise ValueError("wait must be bool")

        target_pose = self._normalize_pose(
            [x, y, z, rx, ry, rz]
        )
        self._check_pose_range(target_pose)

        speed = self._check_value_range(
            "speed",
            speed,
            MIN_SPEED,
            MAX_SPEED,
        )
        acceleration = self._check_value_range(
            "acceleration",
            acceleration,
            MIN_ACCELERATION,
            MAX_ACCELERATION,
        )

        with self._motion_lock:
            command_id = self._begin_motion(
                new_mode="move_l",
                stop_acceleration=acceleration,
            )

            logger.info(
                "[UR7e] moveL command_id=%s target=%s",
                command_id,
                target_pose,
            )

            try:
                # Always asynchronous at the RTDE layer.  This lets this driver
                # own completion detection using RTDE feedback instead of
                # delegating completion semantics to arm_service.
                accepted = bool(
                    self._call_control(
                        "moveL",
                        target_pose,
                        speed,
                        acceleration,
                        True,
                    )
                )
                if not accepted:
                    raise RuntimeError("RTDE moveL command was rejected")

            except Exception:
                self._clear_motion_mode(
                    command_id=command_id,
                    expected_mode="move_l",
                )
                raise

        if not wait:
            return True

        reached = self._wait_until_pose_reached(
            target_pose=target_pose,
            command_id=command_id,
        )

        self._clear_motion_mode(
            command_id=command_id,
            expected_mode="move_l",
        )
        return reached

    def move_arm_joints(
        self,
        joints,
        speed=None,
        acceleration=None,
        wait=True,
    ):
        if speed is None:
            speed = DEFAULT_SPEED
        if acceleration is None:
            acceleration = DEFAULT_ACCELERATION
        if not isinstance(wait, bool):
            raise ValueError("wait must be bool")

        target_joints = self._normalize_joints(joints)
        self._check_joint_range(target_joints)

        speed = self._check_value_range(
            "speed",
            speed,
            MIN_SPEED,
            MAX_SPEED,
        )
        acceleration = self._check_value_range(
            "acceleration",
            acceleration,
            MIN_ACCELERATION,
            MAX_ACCELERATION,
        )

        with self._motion_lock:
            command_id = self._begin_motion(
                new_mode="move_j",
                stop_acceleration=acceleration,
            )

            logger.info(
                "[UR7e] moveJ command_id=%s target=%s",
                command_id,
                target_joints,
            )

            try:
                accepted = bool(
                    self._call_control(
                        "moveJ",
                        target_joints,
                        speed,
                        acceleration,
                        True,
                    )
                )
                if not accepted:
                    raise RuntimeError("RTDE moveJ command was rejected")

            except Exception:
                self._clear_motion_mode(
                    command_id=command_id,
                    expected_mode="move_j",
                )
                raise

        if not wait:
            return True

        reached = self._wait_until_joints_reached(
            target_joints=target_joints,
            command_id=command_id,
        )

        self._clear_motion_mode(
            command_id=command_id,
            expected_mode="move_j",
        )
        return reached

    # ========================================================
    # Public: joint trajectory / pose trajectory
    # ========================================================

    def move_arm_joint_trajectory(
        self,
        joint_trajectory,
        dt=None,
        speed=None,
        acceleration=None,
        lookahead_time=0.1,
        gain=300,
        wait=True,
        move_to_start=True,
        move_to_start_speed=None,
        move_to_start_acceleration=None,
    ):
        """
        Stream a joint trajectory using moveJ -> servoJ.

        arm_service currently supplies a simple list of joint arrays. Structured
        samples with explicit timestamps are also accepted by the driver.

        `wait=False` skips only final reached verification; the streaming call
        itself remains synchronous because the caller must continuously feed the
        real-time servo loop.
        """
        if dt is None:
            dt = DEFAULT_TRAJECTORY_DT
        if speed is None:
            speed = DEFAULT_SPEED
        if acceleration is None:
            acceleration = DEFAULT_ACCELERATION

        if not isinstance(wait, bool):
            raise ValueError("wait must be bool")
        if not isinstance(move_to_start, bool):
            raise ValueError("move_to_start must be bool")

        dt = self._check_value_range(
            "dt",
            dt,
            MIN_TRAJECTORY_DT,
            MAX_TRAJECTORY_DT,
        )
        speed = self._check_value_range(
            "speed",
            speed,
            MIN_SPEED,
            MAX_SPEED,
        )
        acceleration = self._check_value_range(
            "acceleration",
            acceleration,
            MIN_ACCELERATION,
            MAX_ACCELERATION,
        )
        lookahead_time = self._check_value_range(
            "lookahead_time",
            lookahead_time,
            MIN_SERVO_LOOKAHEAD_TIME,
            MAX_SERVO_LOOKAHEAD_TIME,
        )
        gain = int(
            self._check_value_range(
                "gain",
                gain,
                MIN_SERVO_GAIN,
                MAX_SERVO_GAIN,
            )
        )

        if move_to_start_speed is None:
            move_to_start_speed = speed
        else:
            move_to_start_speed = self._check_value_range(
                "move_to_start_speed",
                move_to_start_speed,
                MIN_SPEED,
                MAX_SPEED,
            )

        if move_to_start_acceleration is None:
            move_to_start_acceleration = acceleration
        else:
            move_to_start_acceleration = self._check_value_range(
                "move_to_start_acceleration",
                move_to_start_acceleration,
                MIN_ACCELERATION,
                MAX_ACCELERATION,
            )

        samples = self._normalize_joint_trajectory(
            joint_trajectory,
            dt=dt,
        )
        first_joints = samples[0][1]
        final_joints = samples[-1][1]

        with self._motion_lock:
            command_id = self._begin_motion(
                new_mode="move_j" if move_to_start else "servo_j",
                stop_acceleration=acceleration,
            )

            logger.info(
                "[UR7e] joint trajectory command_id=%s points=%s dt=%s",
                command_id,
                len(samples),
                dt,
            )

            try:
                if move_to_start:
                    reached_start = bool(
                        self._call_control(
                            "moveJ",
                            first_joints,
                            move_to_start_speed,
                            move_to_start_acceleration,
                            False,
                        )
                    )
                    if not reached_start:
                        raise RuntimeError(
                            "moveJ to first joint trajectory waypoint failed"
                        )

                    if not self._check_command_id(command_id):
                        return False

                    self._set_motion_mode("servo_j")

                first_sample_time = samples[0][0]
                wall_start = time.monotonic()
                stream_ready = False

                for sample_index, (sample_time, target_joints) in enumerate(samples):
                    if not self._check_command_id(command_id):
                        logger.info(
                            "[UR7e] servoJ stream interrupted command_id=%s",
                            command_id,
                        )
                        return False

                    target_wall_time = (
                        wall_start
                        + max(0.0, sample_time - first_sample_time)
                    )
                    sleep_seconds = target_wall_time - time.monotonic()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

                    result = bool(
                        self._call_control(
                            "servoJ",
                            target_joints,
                            speed,
                            acceleration,
                            dt,
                            lookahead_time,
                            gain,
                            ensure_ready=not stream_ready,
                        )
                    )
                    stream_ready = True

                    if not result:
                        raise RuntimeError(
                            f"servoJ failed at sample index {sample_index}"
                        )

                time.sleep(max(dt, 0.03))

            finally:
                if (
                    self._check_command_id(command_id)
                    and self._motion_mode == "servo_j"
                ):
                    try:
                        self._call_control(
                            "servoStop",
                            ensure_ready=False,
                        )
                    finally:
                        self._motion_mode = None

                elif (
                    self._check_command_id(command_id)
                    and self._motion_mode == "move_j"
                ):
                    self._motion_mode = None

        if not wait:
            return True

        return self._wait_until_joints_reached(
            target_joints=final_joints,
            command_id=command_id,
        )

    def move_arm_pose_trajectory(
        self,
        pose_trajectory,
        dt=None,
        speed=None,
        acceleration=None,
        lookahead_time=0.1,
        gain=300,
        wait=True,
        move_to_start=True,
        move_to_start_speed=None,
        move_to_start_acceleration=None,
    ):
        """
        Stream a Cartesian trajectory using moveL -> servoL.

        This method exists because the generic arm_service exposes
        move_arm_pose_trajectory as an optional capability.
        """
        if dt is None:
            dt = DEFAULT_TRAJECTORY_DT
        if speed is None:
            speed = DEFAULT_SPEED
        if acceleration is None:
            acceleration = DEFAULT_ACCELERATION

        if not isinstance(wait, bool):
            raise ValueError("wait must be bool")
        if not isinstance(move_to_start, bool):
            raise ValueError("move_to_start must be bool")

        dt = self._check_value_range(
            "dt",
            dt,
            MIN_TRAJECTORY_DT,
            MAX_TRAJECTORY_DT,
        )
        speed = self._check_value_range(
            "speed",
            speed,
            MIN_SPEED,
            MAX_SPEED,
        )
        acceleration = self._check_value_range(
            "acceleration",
            acceleration,
            MIN_ACCELERATION,
            MAX_ACCELERATION,
        )
        lookahead_time = self._check_value_range(
            "lookahead_time",
            lookahead_time,
            MIN_SERVO_LOOKAHEAD_TIME,
            MAX_SERVO_LOOKAHEAD_TIME,
        )
        gain = int(
            self._check_value_range(
                "gain",
                gain,
                MIN_SERVO_GAIN,
                MAX_SERVO_GAIN,
            )
        )

        if move_to_start_speed is None:
            move_to_start_speed = speed
        else:
            move_to_start_speed = self._check_value_range(
                "move_to_start_speed",
                move_to_start_speed,
                MIN_SPEED,
                MAX_SPEED,
            )

        if move_to_start_acceleration is None:
            move_to_start_acceleration = acceleration
        else:
            move_to_start_acceleration = self._check_value_range(
                "move_to_start_acceleration",
                move_to_start_acceleration,
                MIN_ACCELERATION,
                MAX_ACCELERATION,
            )

        samples = self._normalize_pose_trajectory(
            pose_trajectory,
            dt=dt,
        )
        first_pose = samples[0][1]
        final_pose = samples[-1][1]

        with self._motion_lock:
            command_id = self._begin_motion(
                new_mode="move_l" if move_to_start else "servo_l",
                stop_acceleration=acceleration,
            )

            logger.info(
                "[UR7e] pose trajectory command_id=%s points=%s dt=%s",
                command_id,
                len(samples),
                dt,
            )

            try:
                if move_to_start:
                    reached_start = bool(
                        self._call_control(
                            "moveL",
                            first_pose,
                            move_to_start_speed,
                            move_to_start_acceleration,
                            False,
                        )
                    )
                    if not reached_start:
                        raise RuntimeError(
                            "moveL to first pose trajectory waypoint failed"
                        )

                    if not self._check_command_id(command_id):
                        return False

                    self._set_motion_mode("servo_l")

                first_sample_time = samples[0][0]
                wall_start = time.monotonic()
                stream_ready = False

                for sample_index, (sample_time, target_pose) in enumerate(samples):
                    if not self._check_command_id(command_id):
                        logger.info(
                            "[UR7e] servoL stream interrupted command_id=%s",
                            command_id,
                        )
                        return False

                    target_wall_time = (
                        wall_start
                        + max(0.0, sample_time - first_sample_time)
                    )
                    sleep_seconds = target_wall_time - time.monotonic()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

                    result = bool(
                        self._call_control(
                            "servoL",
                            target_pose,
                            speed,
                            acceleration,
                            dt,
                            lookahead_time,
                            gain,
                            ensure_ready=not stream_ready,
                        )
                    )
                    stream_ready = True

                    if not result:
                        raise RuntimeError(
                            f"servoL failed at sample index {sample_index}"
                        )

                time.sleep(max(dt, 0.03))

            finally:
                if (
                    self._check_command_id(command_id)
                    and self._motion_mode == "servo_l"
                ):
                    try:
                        self._call_control(
                            "servoStop",
                            ensure_ready=False,
                        )
                    finally:
                        self._motion_mode = None

                elif (
                    self._check_command_id(command_id)
                    and self._motion_mode == "move_l"
                ):
                    self._motion_mode = None

        if not wait:
            return True

        return self._wait_until_pose_reached(
            target_pose=final_pose,
            command_id=command_id,
        )

    # ========================================================
    # Public: jog
    # ========================================================

    def start_arm_jog(
        self,
        direction,
        linear_speed=None,
        angular_speed=None,
        acceleration=None,
        timeout=None,
    ):
        if linear_speed is None:
            linear_speed = DEFAULT_JOG_LINEAR_SPEED
        if angular_speed is None:
            angular_speed = DEFAULT_JOG_ANGULAR_SPEED
        if acceleration is None:
            acceleration = DEFAULT_JOG_ACCELERATION
        if timeout is None:
            timeout = DEFAULT_JOG_TIMEOUT

        linear_speed = self._check_value_range(
            "linear_speed",
            linear_speed,
            0.001,
            3.0,
        )
        angular_speed = self._check_value_range(
            "angular_speed",
            angular_speed,
            0.001,
            3.0,
        )
        acceleration = self._check_value_range(
            "jog_acceleration",
            acceleration,
            MIN_STOP_ACCELERATION,
            MAX_ACCELERATION,
        )
        timeout = self._check_value_range(
            "jog_timeout",
            timeout,
            0.01,
            10.0,
        )

        if not isinstance(direction, str) or not direction.strip():
            raise ValueError("direction must be a non-empty string")

        direction = direction.lower().strip()
        direction_map = {
            "x+": (0, linear_speed),
            "x-": (0, -linear_speed),
            "y+": (1, linear_speed),
            "y-": (1, -linear_speed),
            "z+": (2, linear_speed),
            "z-": (2, -linear_speed),
            "rx+": (3, angular_speed),
            "rx-": (3, -angular_speed),
            "ry+": (4, angular_speed),
            "ry-": (4, -angular_speed),
            "rz+": (5, angular_speed),
            "rz-": (5, -angular_speed),
        }

        if direction not in direction_map:
            raise ValueError(
                f"unsupported jog direction: {direction}"
            )

        speed_vector = [0.0] * 6
        index, jog_speed = direction_map[direction]
        speed_vector[index] = float(jog_speed)

        state = self._get_receive_state()

        if state.get("emergency_stop") is True:
            raise RuntimeError(
                "arm is in Emergency Stop"
            )
        if state.get("protective_stop") is True:
            raise RuntimeError(
                "arm is in Protective Stop"
            )

        current_pose = state.get("pose")
        if current_pose is None:
            raise RuntimeError("RTDE Receive pose unavailable")

        current_pose = self._normalize_pose(current_pose)
        predicted_pose = list(current_pose)

        # Translation components of speedL are m/s. Rotation safety cannot be
        # validated by simply adding rotvec components, so only XYZ hard-range
        # prediction is performed here.
        if index <= 2:
            predicted_pose[index] += float(jog_speed) * float(timeout)
            self._check_pose_range(predicted_pose)

        with self._motion_lock:
            # Heartbeat refresh for the same jog direction.
            if (
                self._jog_direction == direction
                and self._motion_mode == "speed_l"
            ):
                result = bool(
                    self._call_control(
                        "speedL",
                        speed_vector,
                        float(acceleration),
                        float(timeout),
                    )
                )
                if not result:
                    raise RuntimeError(
                        "RTDE speedL jog heartbeat failed"
                    )
                return True

            command_id = self._begin_motion(
                new_mode="speed_l",
                stop_acceleration=acceleration,
            )

            logger.info(
                "[UR7e] start jog direction=%s command_id=%s vector=%s",
                direction,
                command_id,
                speed_vector,
            )

            try:
                result = bool(
                    self._call_control(
                        "speedL",
                        speed_vector,
                        float(acceleration),
                        float(timeout),
                    )
                )
                if not result:
                    raise RuntimeError(
                        "RTDE speedL jog failed"
                    )

            except Exception:
                self._clear_motion_mode(
                    command_id=command_id,
                    expected_mode="speed_l",
                )
                self._jog_direction = None
                raise

            self._jog_direction = direction

        return True

    def stop_arm_jog(self):
        with self._motion_lock:
            if (
                self._jog_direction is None
                and self._motion_mode != "speed_l"
            ):
                return True

            self._generate_command_id()
            return self._stop_motion(
                acceleration=DEFAULT_JOG_ACCELERATION
            )

    # ========================================================
    # Public: freedrive
    # ========================================================

    def start_arm_freedrive(self):
        with self._motion_lock:
            command_id = self._begin_motion(
                new_mode="freedrive",
                stop_acceleration=DEFAULT_JOG_ACCELERATION,
            )

            logger.info(
                "[UR7e] start freedrive command_id=%s",
                command_id,
            )

            try:
                result = bool(
                    self._call_control("freedriveMode")
                )
                if not result:
                    raise RuntimeError(
                        "RTDE freedriveMode failed"
                    )
            except Exception:
                self._clear_motion_mode(
                    command_id=command_id,
                    expected_mode="freedrive",
                )
                raise

            return True

    def stop_arm_freedrive(self):
        with self._motion_lock:
            if self._motion_mode != "freedrive":
                return True

            self._generate_command_id()
            result = bool(
                self._call_control("endFreedriveMode")
            )
            if not result:
                raise RuntimeError(
                    "RTDE endFreedriveMode failed"
                )

            self._motion_mode = None
            return True

    # ========================================================
    # Public: stop
    # ========================================================

    def stop_arm(self, acceleration=None):
        if acceleration is None:
            acceleration = DEFAULT_JOG_ACCELERATION

        acceleration = self._check_value_range(
            "stop_acceleration",
            acceleration,
            MIN_STOP_ACCELERATION,
            MAX_STOP_ACCELERATION,
        )

        with self._motion_lock:
            command_id = self._generate_command_id()
            logger.info(
                "[UR7e] stop motion command_id=%s",
                command_id,
            )
            return self._stop_motion(
                acceleration=acceleration
            )

    # ========================================================
    # Private: feedback/reached math
    # ========================================================

    def _pose_distance(self, pose1, pose2):
        pose1 = self._normalize_pose(pose1)
        pose2 = self._normalize_pose(pose2)

        dx = pose1[0] - pose2[0]
        dy = pose1[1] - pose2[1]
        dz = pose1[2] - pose2[2]

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _rotation_distance(self, pose1, pose2):
        pose1 = self._normalize_pose(pose1)
        pose2 = self._normalize_pose(pose2)

        rotation1 = self._rotvec_to_matrix(pose1[3:6])
        rotation2 = self._rotvec_to_matrix(pose2[3:6])

        # R_relative = R2^T * R1
        relative_rotation = [
            [
                sum(
                    rotation2[k][i] * rotation1[k][j]
                    for k in range(3)
                )
                for j in range(3)
            ]
            for i in range(3)
        ]

        trace_value = (
            relative_rotation[0][0]
            + relative_rotation[1][1]
            + relative_rotation[2][2]
        )

        cos_angle = (trace_value - 1.0) / 2.0
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)

    @staticmethod
    def _rotvec_to_matrix(rotvec):
        angle = math.sqrt(
            sum(float(value) * float(value) for value in rotvec)
        )

        if angle < 1e-12:
            return [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]

        x, y, z = [float(value) / angle for value in rotvec]
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        one_minus_cos = 1.0 - cos_value

        return [
            [
                x * x * one_minus_cos + cos_value,
                x * y * one_minus_cos - z * sin_value,
                x * z * one_minus_cos + y * sin_value,
            ],
            [
                y * x * one_minus_cos + z * sin_value,
                y * y * one_minus_cos + cos_value,
                y * z * one_minus_cos - x * sin_value,
            ],
            [
                z * x * one_minus_cos - y * sin_value,
                z * y * one_minus_cos + x * sin_value,
                z * z * one_minus_cos + cos_value,
            ],
        ]

    def _joint_distance(self, joints1, joints2):
        joints1 = self._normalize_joints(joints1)
        joints2 = self._normalize_joints(joints2)

        return max(
            abs(joints1[index] - joints2[index])
            for index in range(self.ARM_DOF)
        )

    def _wait_until_pose_reached(
        self,
        target_pose,
        timeout=ARM_WAIT_TIMEOUT,
        position_tolerance=ARM_POSE_TOLERANCE,
        rotation_tolerance=ARM_ROTATION_TOLERANCE,
        command_id=None,
    ):
        target_pose = self._normalize_pose(target_pose)
        timeout = self._check_positive_number("timeout", timeout)
        position_tolerance = self._check_positive_number(
            "position_tolerance",
            position_tolerance,
        )
        rotation_tolerance = self._check_positive_number(
            "rotation_tolerance",
            rotation_tolerance,
        )

        deadline = time.monotonic() + timeout
        current_pose = None
        stable_since = None

        while time.monotonic() < deadline:
            if (
                command_id is not None
                and not self._check_command_id(command_id)
            ):
                logger.info(
                    "[UR7e] pose wait interrupted command_id=%s",
                    command_id,
                )
                return False

            try:
                state = self._get_receive_state()
                current_pose = state.get("pose")
            except Exception as exc:
                logger.warning(
                    "[UR7e] pose feedback unavailable while waiting: %s",
                    exc,
                )
                time.sleep(FEEDBACK_POLL_INTERVAL)
                continue

            if current_pose is None:
                time.sleep(FEEDBACK_POLL_INTERVAL)
                continue

            current_pose = self._normalize_pose(current_pose)
            position_error = self._pose_distance(
                current_pose,
                target_pose,
            )
            rotation_error = self._rotation_distance(
                current_pose,
                target_pose,
            )

            inside_tolerance = (
                position_error <= position_tolerance
                and rotation_error <= rotation_tolerance
            )

            if inside_tolerance:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= FINAL_HOLD_SECONDS:
                    return True
            else:
                stable_since = None

            time.sleep(FEEDBACK_POLL_INTERVAL)

        logger.warning(
            "[UR7e] pose target not reached before timeout: "
            "target=%s current=%s",
            target_pose,
            current_pose,
        )
        return False

    def _wait_until_joints_reached(
        self,
        target_joints,
        timeout=ARM_WAIT_TIMEOUT,
        tolerance=ARM_JOINT_TOLERANCE,
        command_id=None,
    ):
        target_joints = self._normalize_joints(target_joints)
        timeout = self._check_positive_number("timeout", timeout)
        tolerance = self._check_positive_number(
            "tolerance",
            tolerance,
        )

        deadline = time.monotonic() + timeout
        current_joints = None
        stable_since = None

        while time.monotonic() < deadline:
            if (
                command_id is not None
                and not self._check_command_id(command_id)
            ):
                logger.info(
                    "[UR7e] joints wait interrupted command_id=%s",
                    command_id,
                )
                return False

            try:
                state = self._get_receive_state()
                current_joints = state.get("joints")
            except Exception as exc:
                logger.warning(
                    "[UR7e] joint feedback unavailable while waiting: %s",
                    exc,
                )
                time.sleep(FEEDBACK_POLL_INTERVAL)
                continue

            if current_joints is None:
                time.sleep(FEEDBACK_POLL_INTERVAL)
                continue

            current_joints = self._normalize_joints(current_joints)
            error = self._joint_distance(
                current_joints,
                target_joints,
            )

            if error <= tolerance:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= FINAL_HOLD_SECONDS:
                    return True
            else:
                stable_since = None

            time.sleep(FEEDBACK_POLL_INTERVAL)

        logger.warning(
            "[UR7e] joint target not reached before timeout: "
            "target=%s current=%s",
            target_joints,
            current_joints,
        )
        return False

    # ========================================================
    # Public: shutdown
    # ========================================================

    def shutdown(self):
        """
        Idempotent shutdown order:
            1. stop current motion best-effort
            2. stop Receive monitor and let it disconnect Receive
            3. disconnect RTDE Control
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return True

            logger.info(
                "[UR7e] shutting down driver: %s",
                self.ip,
            )

            try:
                # Motion stop is best-effort during process teardown.  Do not
                # fail shutdown solely because the controller is already gone.
                try:
                    with self._motion_lock:
                        if self._motion_mode is not None:
                            self._generate_command_id()
                            self._stop_motion(
                                acceleration=DEFAULT_JOG_ACCELERATION
                            )
                except Exception:
                    logger.warning(
                        "[UR7e] motion stop failed during shutdown",
                        exc_info=True,
                    )

                self._stop_receive_monitor()

                with self._motion_lock:
                    self._disconnect_control()

                self._reset_receive_state(
                    error="UR7e driver shutdown"
                )

                self._shutdown_done = True

                logger.info(
                    "[UR7e] driver shutdown completed: %s",
                    self.ip,
                )
                return True

            except Exception:
                logger.exception(
                    "[UR7e] driver shutdown failed: %s",
                    self.ip,
                )
                raise