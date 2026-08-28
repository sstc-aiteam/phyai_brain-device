import logging
import math
import threading
import time
import atexit

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


logger = logging.getLogger(__name__)


# =========================
# UR5 Driver Defaults
# =========================

# Driver fallback defaults.
# Normally arm_service passes per-arm values into this Driver.
# These values are used only when the Driver is called directly
# or the upper layer explicitly passes None.
DEFAULT_TRAJECTORY_DT = 0.1
DEFAULT_SPEED = 0.1
DEFAULT_ACCELERATION = 0.1

DEFAULT_JOG_LINEAR_SPEED = 0.05
DEFAULT_JOG_ANGULAR_SPEED = 0.10
DEFAULT_JOG_ACCELERATION = 0.10
DEFAULT_JOG_TIMEOUT = 0.2


# =========================
# UR5 Hard Safety Limits
# =========================

# Absolute Driver-side safety boundary.
# arm_service may impose a narrower per-arm soft safety range,
# but it must never widen these Driver hard limits.
UR7E_HARD_X_RANGE = (-3.0, 3.0)
UR7E_HARD_Y_RANGE = (-3.0, 3.0)
UR7E_HARD_Z_RANGE = (-3.0, 3.0)

# =========================
# Hard Limits
# =========================

ARM_JOINT_LIMITS = [
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
]

MIN_SPEED = 0.01
MAX_SPEED = 3

MIN_ACCELERATION = 0.01
MAX_ACCELERATION = 3

MIN_STOP_ACCELERATION = 0.1
MAX_STOP_ACCELERATION = 0.5

MIN_TRAJECTORY_DT = 0.001
MAX_TRAJECTORY_DT = 0.5

MIN_SERVOJ_LOOKAHEAD_TIME = 0.03
MAX_SERVOJ_LOOKAHEAD_TIME = 10

MIN_SERVOJ_GAIN = 100
MAX_SERVOJ_GAIN = 1000

MAX_TRAJECTORY_POINTS = 1000000

ARM_POSE_TOLERANCE = 0.005
ARM_ROTATION_TOLERANCE = 0.03
ARM_JOINT_TOLERANCE = 0.01
ARM_WAIT_TIMEOUT = 15

FINAL_HOLD_SECONDS = 0.2

# =========================
# RTDE Receive Monitor
# =========================

RTDE_RECEIVE_MONITOR_HZ = 20.0
RTDE_RECEIVE_MONITOR_INTERVAL = 1.0 / RTDE_RECEIVE_MONITOR_HZ
RTDE_RECEIVE_STALE_TIMEOUT = 0.5
RTDE_RECEIVE_INITIAL_TIMEOUT = 3.0
RTDE_RECEIVE_MONITOR_JOIN_TIMEOUT = 2.0
RTDE_RECEIVE_ERROR_RETRY_INTERVAL = 1.0


class UR7eDriver:
    """
    Universal Robots UR5 RTDE driver。
    """

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
            "jog",
            "freedrive",
            "stop",
        ],
    }

    def __init__(self, ip):
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError("ip 必須是非空字串")

        self.ip = ip.strip()

        # Control 與 Receive 分開鎖定。
        self._rtde_control_lock = threading.RLock()
        self._rtde_receive_lock = threading.RLock()
        self._motion_lock = threading.RLock()

        self._rtde_c = None
        self._rtde_r = None
        self._motion_command_id = 0
        self._motion_mode = None

        self._jog_direction = None

        # Receive monitor / cache
        self._state_lock = threading.RLock()
        self._state_condition = threading.Condition(self._state_lock)
        self._receive_monitor_lock = threading.RLock()
        self._receive_monitor_stop_event = threading.Event()
        self._receive_monitor_thread = None
        self._receive_sequence = 0
        self._receive_last_progress_monotonic = None
        self._latest_state = self._empty_receive_state()

        self._shutdown_lock = threading.RLock()
        self._shutdown_done = False

        atexit.register(self.shutdown)

    # =========================
    # Communication
    # =========================

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
            "is_emergency_stopped": None,
            "is_protective_stopped": None,
            "pose": None,
            "joints": None,
        }

    def _reset_receive_state(self, error=None):
        with self._state_condition:
            self._receive_last_progress_monotonic = None
            self._latest_state = self._empty_receive_state(error=error)
            self._state_condition.notify_all()

    def _copy_receive_state_locked(self):
        state = dict(self._latest_state)

        if state["pose"] is not None:
            state["pose"] = list(state["pose"])

        if state["joints"] is not None:
            state["joints"] = list(state["joints"])

        last_progress = state.get("last_progress_monotonic")
        if last_progress is not None:
            state["data_age_seconds"] = time.monotonic() - last_progress
            state["data_stale"] = (
                state["data_age_seconds"] > RTDE_RECEIVE_STALE_TIMEOUT
            )
            if state["data_stale"]:
                state["data_valid"] = False

        return state

    def _start_receive_monitor(self):
        with self._shutdown_lock:
            if self._shutdown_done:
                raise RuntimeError(
                    "UR5 Driver 已 shutdown，不能重新啟動 Receive monitor"
                )

        with self._receive_monitor_lock:
            thread = self._receive_monitor_thread

            if thread is not None and thread.is_alive():
                return True

            self._receive_monitor_stop_event.clear()

            thread = threading.Thread(
                target=self._receive_monitor_loop,
                name=f"UR5ReceiveMonitor-{self.ip}",
                daemon=True,
            )

            self._receive_monitor_thread = thread
            thread.start()

        logger.info(
            "[UR7e] RTDE receive monitor started: %s",
            self.ip,
        )

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
                "RTDE Receive monitor 無法正常停止，禁止 disconnect RTDE object"
            )

        with self._receive_monitor_lock:
            if self._receive_monitor_thread is thread:
                self._receive_monitor_thread = None

        logger.info("[UR7e] RTDE receive monitor stopped: %s", self.ip)
        return True

    def _receive_monitor_loop(self):
        """
        RTDEReceiveInterface 的唯一 I/O owner。

        Receive monitor 全程常駐並持續讀取 UR 狀態。
        其他 function 不得直接操作 RTDEReceiveInterface，
        只能透過 _get_receive_state() 讀取 _latest_state cache。
        """
        previous_controller_timestamp = None

        try:
            while not self._receive_monitor_stop_event.is_set():
                cycle_start = time.monotonic()

                try:
                    with self._rtde_receive_lock:
                        rtde_r = self._get_receive_for_monitor()

                        controller_timestamp = float(
                            rtde_r.getTimestamp()
                        )
                        pose = list(
                            rtde_r.getActualTCPPose()
                        )
                        joints = list(
                            rtde_r.getActualQ()
                        )
                        arm_mode = rtde_r.getRobotMode()
                        is_emergency_stopped = bool(
                            rtde_r.isEmergencyStopped()
                        )
                        is_protective_stopped = bool(
                            rtde_r.isProtectiveStopped()
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
                            "is_emergency_stopped": is_emergency_stopped,
                            "is_protective_stopped": is_protective_stopped,
                            "pose": pose,
                            "joints": joints,
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

                    # Receive 的斷線 / 重建只能由 monitor 自己處理。
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
            # monitor 結束時，由 monitor 自己關閉 Receive connection。
            with self._rtde_receive_lock:
                self._disconnect_receive_for_monitor()

            self._reset_receive_state(
                error="RTDE Receive monitor stopped"
            )

    def _wait_for_receive_state(
        self,
        timeout=RTDE_RECEIVE_INITIAL_TIMEOUT,
        after_sequence=None,
    ):
        """
        等待 Receive monitor 提供有效且未 stale 的最新資料。
        """
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("timeout 必須大於 0")

        # Receive monitor 採常駐模式；若尚未啟動則啟動。
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
                        "RTDE Receive 無有效新資料："
                        f"connected={state['connected']}, "
                        f"data_valid={state['data_valid']}, "
                        f"data_stale={state['data_stale']}, "
                        f"age={age}, "
                        f"sequence={state['sequence']}, "
                        f"error={state.get('receive_error')}"
                    )

                self._state_condition.wait(timeout=remaining)

    def _get_receive_state(self):
        """
        取得 Receive monitor cache 中最新且有效的狀態。

        此 function 不直接操作 RTDEReceiveInterface。
        """
        return self._wait_for_receive_state(
            timeout=RTDE_RECEIVE_INITIAL_TIMEOUT
        )

    # =========================
    # Control Channel
    # =========================

    def _get_control_for_gateway(self):
        """
        取得或建立 RTDEControlInterface。

        只有 Control Gateway 內部可以使用這個 function。
        一般 motion function 應統一透過 _call_control() 發送命令。
        """
        with self._rtde_control_lock:
            if self._rtde_c is None:
                logger.info(
                    "[UR7e] connect RTDEControlInterface: %s",
                    self.ip,
                )
                self._rtde_c = RTDEControlInterface(self.ip)

            return self._rtde_c

    def _ensure_control_ready(self):
        """
        確認 RTDE Control socket 與 control script 可正常使用。

        Receive Channel 完全不受此 function 影響。
        """
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
        """
        RTDEControlInterface 的唯一命令 Gateway。

        一般命令預設先確認 Control ready。
        ServoJ trajectory 等高頻串流只在第一個 Control command
        執行 readiness check，後續 sample 以 ensure_ready=False 發送，
        避免每個 trajectory point 都額外檢查 Control 狀態。
        """
        if not isinstance(method_name, str) or not method_name.strip():
            raise ValueError("method_name 必須是非空字串")

        if not isinstance(ensure_ready, bool):
            raise ValueError("ensure_ready 必須是 bool")

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
                    "[UR7e] RTDE control method failed: "
                    "method=%s error=%s",
                    method_name,
                    exc,
                )
                raise

    def _disconnect_control(self):
        """
        關閉並清除 RTDEControlInterface。
        """
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

    # =========================
    # Receive Channel
    # =========================

    def _get_receive_for_monitor(self):
        """
        取得或建立 RTDEReceiveInterface。

        僅允許 _receive_monitor_loop() 使用。
        """
        with self._rtde_receive_lock:
            if self._rtde_r is None:
                logger.info(
                    "[UR7e] connect RTDEReceiveInterface: %s",
                    self.ip,
                )
                self._rtde_r = RTDEReceiveInterface(self.ip)

            return self._rtde_r

    def _disconnect_receive_for_monitor(self):
        """
        關閉並清除 RTDEReceiveInterface。

        僅由 Receive monitor 自己呼叫。
        """
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

    # =========================
    # Connection Lifecycle
    # =========================

    def reconnect_arm(self):
        """
        重建 Control Channel，並確認 Receive monitor 持續運作。

        Receive monitor 不會因 Control reconnect 而停止。
        如果 Receive 自己發生錯誤，會由 monitor 自動重連。
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

        logger.info(
            "[UR7e] RTDE Control reconnected; Receive monitor active: %s",
            self.ip,
        )
        return True

    def send_custom_script_function(self, function_name, script):
        """
        透過 Control Gateway 呼叫 sendCustomScriptFunction。

        Receive monitor 全程保持運作。
        """
        if not isinstance(function_name, str) or not function_name.strip():
            raise ValueError("function_name 必須是非空字串")

        if not isinstance(script, str) or not script.strip():
            raise ValueError("script 必須是非空字串")

        with self._motion_lock:
            result = self._call_control(
                "sendCustomScriptFunction",
                function_name.strip(),
                script,
            )

        # custom script 可能讓 UR controller 上的 control script 停止。
        # 透過 Control Gateway 再做一次 readiness check，不直接操作 Control。
        time.sleep(0.1)
        self._call_control("isProgramRunning")

        return bool(result)

    def _generate_command_id(self):
        self._motion_command_id += 1
        return self._motion_command_id

    def _check_command_id(self, command_id):
        with self._motion_lock:
            return command_id == self._motion_command_id


    # 建立手臂動的類型
    def _set_motion_mode(self, mode):
        allowed_modes = {None, "move_j", "move_l", "servo_j", "speed_l", "jog", "freedrive",}

        if mode not in allowed_modes:
            raise ValueError(
                f"invalid motion mode: {mode}"
            )

        self._motion_mode = mode

    # 清楚手臂動的類型
    def _clear_motion_mode(self, command_id=None, expected_mode=None):
        """
        僅在目前仍是同一個 command 時清除 motion mode，
        避免舊動作結束後清掉新動作的狀態。
        """
        with self._motion_lock:
            if (command_id is not None and command_id != self._motion_command_id):
                return False
            if (expected_mode is not None and self._motion_mode != expected_mode):
                return False

            self._motion_mode = None
            return True
            
    # 依照不同動的類型 去停止手臂
    def _stop_motion(self, acceleration=None):

        if acceleration is None:
            acceleration = DEFAULT_JOG_ACCELERATION

        motion_mode = self._motion_mode

        if motion_mode is None:
            return True

        logger.info(
            "[UR7e] stop current motion mode=%s",
            motion_mode,
        )

        acceleration = self._check_value_range(
            "stop_acceleration",
            acceleration,
            MIN_STOP_ACCELERATION,
            MAX_STOP_ACCELERATION,
        )

        if motion_mode == "servo_j":
            self._call_control(
                "servoStop",
            )

        elif motion_mode == "speed_l":
            self._call_control(
                "speedStop",
                float(acceleration),
            )

        elif motion_mode == "jog":
            self._call_control(
                "jogStop",
            )

        elif motion_mode == "freedrive":
            self._call_control(
                "endFreedriveMode"
            )

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

        if motion_mode in ("speed_l", "jog"):
            self._jog_direction = None

        return True

    def _begin_motion(self, new_mode, stop_acceleration=None):

        if stop_acceleration is None:
            stop_acceleration = DEFAULT_JOG_ACCELERATION

        allowed_modes = {"move_j", "move_l", "servo_j", "speed_l", "jog", "freedrive"}

        if new_mode not in allowed_modes:
            raise ValueError(f"invalid new motion mode: {new_mode}")
        command_id = self._generate_command_id()
        previous_mode = self._motion_mode

        logger.info("[UR7e] begin motion command_id=%s previous_mode=%s new_mode=%s", command_id, previous_mode, new_mode)

        if previous_mode is not None:
            self._stop_motion(acceleration=stop_acceleration)
            time.sleep(0.05)

        self._set_motion_mode(new_mode)

        return command_id

    # =========================
    # Validation
    # =========================

    def _normalize_pose(self, pose):
        if not isinstance(pose, (list, tuple)) or len(pose) != 6:
            raise ValueError(
                "pose 必須是包含 6 個值的 list 或 tuple："
                "[x, y, z, rx, ry, rz]"
            )

        normalized = []

        for index, value in enumerate(pose):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"pose[{index}] 必須是數值"
                ) from exc

            if not math.isfinite(number):
                raise ValueError(
                    f"pose[{index}] 必須是有限數值"
                )

            normalized.append(number)

        return normalized

    def _normalize_joints(self, joints):
        if (
            not isinstance(joints, (list, tuple))
            or len(joints) != self.ARM_DOF
        ):
            raise ValueError(
                f"joints 必須是包含 {self.ARM_DOF} 個關節角度的 "
                "list 或 tuple"
            )

        normalized = []

        for index, value in enumerate(joints):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"joints[{index}] 必須是數值"
                ) from exc

            if not math.isfinite(number):
                raise ValueError(
                    f"joints[{index}] 必須是有限數值"
                )

            normalized.append(number)

        return normalized

    @staticmethod
    def _check_value_range(name, value, min_value, max_value):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必須是數值") from exc

        if not math.isfinite(number):
            raise ValueError(f"{name} 必須是有限數值")

        if not min_value <= number <= max_value:
            raise ValueError(f"{name} 超出允許範圍：{number}，允許範圍 {min_value} ~ {max_value}")

        return number

    def _normalize_joint_trajectory(self, joint_trajectory, dt=None):
        if dt is None:
            dt = DEFAULT_TRAJECTORY_DT

        if not isinstance(joint_trajectory, (list, tuple)):
            raise ValueError(
                "joint_trajectory 必須是 list 或 tuple"
            )

        if not joint_trajectory:
            raise ValueError("joint_trajectory 不可為空")

        if len(joint_trajectory) > MAX_TRAJECTORY_POINTS:
            raise ValueError(
                f"joint_trajectory 筆數過多："
                f"{len(joint_trajectory)}，"
                f"最多允許 {MAX_TRAJECTORY_POINTS} 筆"
            )

        normalized_samples = []
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
                    raise ValueError("sample time 必須是有限數值")
                if previous_time is not None and sample_time < previous_time:
                    raise ValueError("sample time 必須依序遞增")

                target_joints = self._normalize_joints(joints)
                self._check_joint_range(target_joints)
                normalized_samples.append((sample_time, target_joints))
                previous_time = sample_time

            except Exception as exc:
                raise ValueError(
                    f"joint_trajectory 第 {index} 筆資料錯誤：{exc}"
                ) from exc

        return normalized_samples

    def _check_pose_range(self, pose):
        pose = self._normalize_pose(pose)

        x, y, z = pose[:3]

        x_min, x_max = UR7E_HARD_X_RANGE
        y_min, y_max = UR7E_HARD_Y_RANGE
        z_min, z_max = UR7E_HARD_Z_RANGE

        if not x_min <= x <= x_max:
            raise ValueError(
                f"x 超出安全範圍：{x}，"
                f"允許範圍 {UR7E_HARD_X_RANGE}"
            )

        if not y_min <= y <= y_max:
            raise ValueError(
                f"y 超出安全範圍：{y}，"
                f"允許範圍 {UR7E_HARD_Y_RANGE}"
            )

        if not z_min <= z <= z_max:
            raise ValueError(
                f"z 超出安全範圍：{z}，"
                f"允許範圍 {UR7E_HARD_Z_RANGE}"
            )

        return True

    def _check_joint_range(self, joints):
        joints = self._normalize_joints(joints)

        if len(ARM_JOINT_LIMITS) != self.ARM_DOF:
            raise RuntimeError(
                "ARM_JOINT_LIMITS 必須包含 ARM_DOF 組限制"
            )

        for index, joint_rad in enumerate(joints):
            min_rad, max_rad = ARM_JOINT_LIMITS[index]

            if not min_rad <= joint_rad <= max_rad:
                raise ValueError(
                    f"J{index + 1} 超出安全範圍："
                    f"{joint_rad:.4f} rad，"
                    f"允許範圍 "
                    f"{min_rad:.4f} ~ {max_rad:.4f} rad"
                )

        return True

    # =========================
    # RTDE Read
    # =========================

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
        try:
            state = self._get_receive_state()
        except Exception as exc:
            return {
                "connected": False,
                "data_valid": False,
                "data_stale": True,
                "data_age_seconds": None,
                "controller_timestamp": None,
                "sequence": None,
                "receive_error": str(exc),
                "arm_mode": None,
                "is_emergency_stopped": None,
                "is_protective_stopped": None,
                "pose": None,
                "joints": None,
            }

        return {
            "connected": state["connected"],
            "data_valid": state["data_valid"],
            "data_stale": state["data_stale"],
            "data_age_seconds": state["data_age_seconds"],
            "controller_timestamp": state["controller_timestamp"],
            "sequence": state["sequence"],
            "receive_error": state["receive_error"],
            "arm_mode": state["arm_mode"],
            "is_emergency_stopped": state["is_emergency_stopped"],
            "is_protective_stopped": state["is_protective_stopped"],
            "pose": (
                list(state["pose"])
                if state["pose"] is not None
                else None
            ),
            "joints": (
                list(state["joints"])
                if state["joints"] is not None
                else None
            ),
        }

    def move_arm_pose(self, x, y, z, rx, ry, rz, speed=None, acceleration=None, wait=True):
        if speed is None:
            speed = DEFAULT_SPEED

        if acceleration is None:
            acceleration = DEFAULT_ACCELERATION

        if not isinstance(wait, bool):
            raise ValueError("wait 必須是 bool")


        # 檢查格式以及範圍
        target_pose = self._normalize_pose([x, y, z, rx, ry, rz])
        self._check_pose_range(target_pose)
        speed = self._check_value_range("speed", speed, MIN_SPEED, MAX_SPEED)
        acceleration = self._check_value_range("acceleration", acceleration, MIN_ACCELERATION, MAX_ACCELERATION)

        # 開始運動前上鎖
        with self._motion_lock:
            command_id = self._begin_motion( new_mode="move_l", stop_acceleration=acceleration)
            logger.info("[UR7e] RTDE moveL target: %s", target_pose)

            try:
                self._call_control("moveL", target_pose, speed, acceleration, True)
            except Exception:
                self._clear_motion_mode(command_id=command_id, expected_mode="move_l")
                raise
        # 結束解鎖

        if not wait:
            return True
        reached = self.wait_until_pose_reached(target_pose=target_pose, command_id=command_id)
        self._clear_motion_mode(command_id=command_id, expected_mode="move_l")

        return reached

    def move_arm_joints(self, joints, speed=None, acceleration=None, wait=True):
        if speed is None:
            speed = DEFAULT_SPEED

        if acceleration is None:
            acceleration = DEFAULT_ACCELERATION

        if not isinstance(wait, bool):
            raise ValueError("wait 必須是 bool")

        # 檢查格式以及範圍
        target_joints = self._normalize_joints(joints)
        self._check_joint_range(target_joints)
        speed = self._check_value_range("speed", speed, MIN_SPEED, MAX_SPEED)
        acceleration = self._check_value_range("acceleration", acceleration, MIN_ACCELERATION, MAX_ACCELERATION)

        # 開始運動前上鎖
        with self._motion_lock:
            command_id = self._begin_motion(new_mode="move_j", stop_acceleration=acceleration)
            logger.info("[UR7e] RTDE moveJ target: %s", target_joints,)

            try:
                reached = bool(self._call_control(
                    "moveJ",
                    target_joints,
                    speed,
                    acceleration,
                    not wait,
                ))
            except Exception:
                self._clear_motion_mode(command_id=command_id, expected_mode="move_j")
                raise

        if not wait:
            return True

        self._clear_motion_mode(command_id=command_id, expected_mode="move_j")
        return reached


    def move_arm_joint_trajectory(self, joint_trajectory, dt=None, speed=None, acceleration=None, lookahead_time=0.1, gain=300, wait=True, move_to_start=True, move_to_start_speed=None, move_to_start_acceleration=None):
        """依 sample timestamp 串流多點 ServoJ trajectory。"""
        if dt is None:
            dt = DEFAULT_TRAJECTORY_DT

        if speed is None:
            speed = DEFAULT_SPEED

        if acceleration is None:
            acceleration = DEFAULT_ACCELERATION

        if not isinstance(wait, bool):
            raise ValueError("wait 必須是 bool")
        if not isinstance(move_to_start, bool):
            raise ValueError("move_to_start 必須是 bool")

        dt = self._check_value_range("dt", dt, MIN_TRAJECTORY_DT, MAX_TRAJECTORY_DT)
        speed = self._check_value_range("speed", speed, MIN_SPEED, MAX_SPEED)
        acceleration = self._check_value_range("acceleration", acceleration, MIN_ACCELERATION, MAX_ACCELERATION)
        if move_to_start_speed is None:
            move_to_start_speed = speed
        else:
            move_to_start_speed = self._check_value_range("move_to_start_speed", move_to_start_speed, MIN_SPEED, MAX_SPEED)
        if move_to_start_acceleration is None:
            move_to_start_acceleration = acceleration
        else:
            move_to_start_acceleration = self._check_value_range("move_to_start_acceleration", move_to_start_acceleration, MIN_ACCELERATION, MAX_ACCELERATION)
        lookahead_time = self._check_value_range("lookahead_time", lookahead_time, MIN_SERVOJ_LOOKAHEAD_TIME, MAX_SERVOJ_LOOKAHEAD_TIME)
        gain = int(self._check_value_range("gain", gain, MIN_SERVOJ_GAIN, MAX_SERVOJ_GAIN))
        samples = self._normalize_joint_trajectory(joint_trajectory, dt=dt)
        first_joints = samples[0][1]

        # 整段 moveJ + servoJ 共用同一個 motion lock，防止 Flask 的
        # jog/stop 請求在 trajectory 執行期間插隊並取消動作。
        # Receive monitor 不停止，trajectory 執行期間仍持續更新狀態 cache。
        with self._motion_lock:
            command_id = self._begin_motion(new_mode="move_j", stop_acceleration=acceleration)
            logger.info("[UR7e] stream ServoJ command_id=%s points=%s dt=%s", command_id, len(samples), dt)

            try:
                if move_to_start:
                    # asynchronous=False：由 RTDE moveJ 阻塞到第一點確實完成。
                    raw_reached_start = self._call_control(
                        "moveJ",
                        first_joints,
                        move_to_start_speed,
                        move_to_start_acceleration,
                        False,
                    )
                    logger.warning(
                        "[UR7e] moveJ first waypoint target=%s result=%r type=%s",
                        first_joints,
                        raw_reached_start,
                        type(raw_reached_start).__name__,
                    )
                    reached_start = bool(raw_reached_start)
                    if not reached_start:
                        raise RuntimeError("moveJ to first waypoint failed before servoJ streaming")

                if not self._check_command_id(command_id):
                    return False
                self._set_motion_mode("servo_j")

                # ServoJ 必須依 UR controller 的實際 control step 持續送點。
                # 錄製資料通常只有 10 Hz，先線性插值到 e-Series 500 Hz，
                # 並使用 ur_rtde 官方建議的 initPeriod()/waitPeriod() 迴圈。
                self._ensure_control_ready()
                control_dt = float(self._call_control("getStepTime", ensure_ready=False))
                if not math.isfinite(control_dt) or control_dt <= 0.0:
                    # ur_rtde 文件規定 getStepTime() 發生錯誤時回傳 0。
                    # 一般 Linux + Python replay 使用 125 Hz，降低排程與
                    # 網路 jitter；對原始 10 Hz dataset 仍有足夠插值解析度。
                    logger.warning(
                        "[UR7e] getStepTime returned %r; fallback to stable 125 Hz replay",
                        control_dt,
                    )
                    control_dt = 1.0 / 125.0

                first_sample_time = samples[0][0]
                last_sample_time = samples[-1][0]
                servo_samples = []
                segment_index = 0
                servo_time = first_sample_time
                while servo_time < last_sample_time:
                    while (
                        segment_index + 1 < len(samples) - 1
                        and samples[segment_index + 1][0] < servo_time
                    ):
                        segment_index += 1
                    t0, q0 = samples[segment_index]
                    t1, q1 = samples[min(segment_index + 1, len(samples) - 1)]
                    ratio = 0.0 if t1 <= t0 else min(1.0, max(0.0, (servo_time - t0) / (t1 - t0)))
                    servo_samples.append([
                        start + (end - start) * ratio
                        for start, end in zip(q0, q1)
                    ])
                    servo_time += control_dt
                servo_samples.append(samples[-1][1])

                logger.info(
                    "[UR7e] resampled ServoJ points=%s control_dt=%s duration=%s",
                    len(servo_samples), control_dt, last_sample_time - first_sample_time,
                )
                stream_ready = False
                consecutive_servo_failures = 0

                for sample_index, joints in enumerate(servo_samples):
                    if not self._check_command_id(command_id):
                        logger.info("[UR7e] ServoJ stream interrupted command_id=%s", command_id)
                        return False

                    cycle_start = self._call_control("initPeriod", ensure_ready=not stream_ready)
                    result = bool(self._call_control(
                        "servoJ",
                        joints,
                        speed,
                        acceleration,
                        control_dt,
                        lookahead_time,
                        gain,
                        ensure_ready=False,
                    ))
                    stream_ready = True

                    if result:
                        consecutive_servo_failures = 0
                    else:
                        consecutive_servo_failures += 1
                        logger.warning(
                            "[UR7e] ServoJ transient failure sample=%s consecutive=%s",
                            sample_index,
                            consecutive_servo_failures,
                        )
                        if consecutive_servo_failures >= 3:
                            raise RuntimeError(
                                "servoJ failed for 3 consecutive control cycles "
                                f"ending at sample index {sample_index}"
                            )
                    self._call_control("waitPeriod", cycle_start, ensure_ready=False)

                time.sleep(max(control_dt, 0.03))
                logger.info("[UR7e] ServoJ stream completed command_id=%s", command_id)
                return True
            finally:
                if self._check_command_id(command_id) and self._motion_mode == "servo_j":
                    try:
                        self._call_control("servoStop", ensure_ready=False)
                    finally:
                        self._motion_mode = None
                elif self._check_command_id(command_id) and self._motion_mode == "move_j":
                    self._motion_mode = None

    # =========================
    # Jog Control
    # =========================
    def start_arm_jog(self, direction, linear_speed=None, angular_speed=None, acceleration=None, timeout=None):

        if linear_speed is None:
            linear_speed = DEFAULT_JOG_LINEAR_SPEED

        if angular_speed is None:
            angular_speed = DEFAULT_JOG_ANGULAR_SPEED

        if acceleration is None:
            acceleration = DEFAULT_JOG_ACCELERATION

        if timeout is None:
            timeout = DEFAULT_JOG_TIMEOUT

        linear_speed = self._check_value_range(
            "linear_speed", linear_speed, 0.001, 3.0
        )
        angular_speed = self._check_value_range(
            "angular_speed", angular_speed, 0.001, 3.0
        )
        acceleration = self._check_value_range(
            "jog_acceleration",
            acceleration,
            MIN_STOP_ACCELERATION,
            MAX_ACCELERATION,
        )
        timeout = self._check_value_range(
            "jog_timeout", timeout, 0.01, 10.0
        )

        if not isinstance(direction, str) or not direction.strip():
            raise ValueError("direction 必須是非空字串")

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
                f"不支援的 jog direction: {direction}"
            )

        speed_vector = [0.0] * 6
        index, jog_speed = direction_map[direction]
        speed_vector[index] = float(jog_speed)

        # 所有 Receive 狀態只從 monitor cache 取得。
        state = self._get_receive_state()

        if state.get("is_emergency_stopped"):
            raise RuntimeError(
                "手臂目前是 Emergency Stop，請先在示教器確認並解除"
            )

        if state.get("is_protective_stopped"):
            raise RuntimeError(
                "手臂目前是 Protective Stop，請先在示教器確認安全後解除"
            )

        current_pose = state.get("pose")
        if current_pose is None:
            raise RuntimeError("RTDE Receive pose unavailable")

        current_pose = self._normalize_pose(current_pose)
        predicted_pose = list(current_pose)

        # speedL 與 UR TCP pose 的平移單位都是 m / m/s。
        if index <= 2:
            predicted_pose[index] += (
                float(jog_speed) * float(timeout)
            )

        self._check_pose_range(predicted_pose)

        with self._motion_lock:

            # 相同方向已在移動：重送 speedL 速度向量。
            if (
                self._jog_direction == direction
                and self._motion_mode == "speed_l"
            ):
                logger.debug(
                    "[UR7e] jog heartbeat direction=%s",
                    direction,
                )

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
                        "RTDE speedL heartbeat 執行失敗"
                    )

                return True

            command_id = self._begin_motion(
                new_mode="speed_l",
                stop_acceleration=acceleration,
            )

            logger.info(
                "[UR7e] start jog "
                "direction=%s command_id=%s speed_vector=%s",
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
                        "RTDE speedL Jog 執行失敗"
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

            if (self._jog_direction is None and self._motion_mode != "speed_l"):
                return True

            stopped_direction = self._jog_direction
            command_id = self._generate_command_id()

            logger.info("[UR7e] stop jog direction=%s command_id=%s", stopped_direction, command_id)

            return self._stop_motion(acceleration=DEFAULT_JOG_ACCELERATION)
            
    # =========================
    # Freedrive / Manual Mode
    # =========================

    def start_arm_freedrive(
        self,
    ):
        """
        開啟 Freedrive 手動拖曳模式。

        開啟後可直接用手拖動機械手臂。
        Freedrive 啟用期間不可執行一般 move / jog trajectory。
        """

        with self._motion_lock:

            # 先停止目前任何 motion。
            if self._motion_mode is not None:
                self._stop_motion(
                    acceleration=
                        DEFAULT_JOG_ACCELERATION
                )

                time.sleep(
                    0.05
                )

            command_id = (
                self._generate_command_id()
            )

            logger.info(
                "[UR7e] start freedrive "
                "command_id=%s",
                command_id,
            )

            result = (
                self._call_control(
                    "freedriveMode"
                )
            )

            if not result:
                raise RuntimeError(
                    "RTDE freedriveMode 執行失敗"
                )

            self._set_motion_mode(
                "freedrive"
            )

            return True


    def stop_arm_freedrive(
        self,
    ):
        """
        關閉 Freedrive，恢復一般 position control。
        """

        with self._motion_lock:

            if self._motion_mode != "freedrive":
                return True

            command_id = (
                self._generate_command_id()
            )

            logger.info(
                "[UR7e] stop freedrive "
                "command_id=%s",
                command_id,
            )

            result = (
                self._call_control(
                    "endFreedriveMode"
                )
            )

            if not result:
                raise RuntimeError(
                    "RTDE endFreedriveMode 執行失敗"
                )

            self._motion_mode = None

            return True
            
    # =========================
    # Stop / Safety
    # =========================

    def stop_arm(self, acceleration=None):
        if acceleration is None:
            acceleration = DEFAULT_JOG_ACCELERATION

        acceleration = self._check_value_range("acceleration", acceleration, MIN_STOP_ACCELERATION, MAX_STOP_ACCELERATION)

        with self._motion_lock:
            command_id = self._generate_command_id()

            logger.info("[UR7e] stop motion command_id=%s", command_id)

            return self._stop_motion(acceleration=acceleration)

    # =========================
    # Feedback
    # =========================

    def pose_distance(self, pose1, pose2):
        pose1 = self._normalize_pose(pose1)
        pose2 = self._normalize_pose(pose2)

        dx = pose1[0] - pose2[0]
        dy = pose1[1] - pose2[1]
        dz = pose1[2] - pose2[2]

        return math.sqrt(
            dx * dx + dy * dy + dz * dz
        )

    def rotation_distance(self, pose1, pose2):
        pose1 = self._normalize_pose(pose1)
        pose2 = self._normalize_pose(pose2)

        rotation1 = self._rotvec_to_matrix(
            pose1[3:6]
        )
        rotation2 = self._rotvec_to_matrix(
            pose2[3:6]
        )

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
        cos_angle = max(
            -1.0,
            min(1.0, cos_angle),
        )

        return math.acos(cos_angle)

    @staticmethod
    def _rotvec_to_matrix(rotvec):
        angle = math.sqrt(
            sum(value * value for value in rotvec)
        )

        if angle < 1e-12:
            return [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]

        x, y, z = [
            value / angle
            for value in rotvec
        ]

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

    def joint_distance(self, joints1, joints2):
        joints1 = self._normalize_joints(joints1)
        joints2 = self._normalize_joints(joints2)

        return max(
            abs(joints1[index] - joints2[index])
            for index in range(self.ARM_DOF)
        )

    def wait_until_pose_reached(
        self,
        target_pose,
        timeout=ARM_WAIT_TIMEOUT,
        position_tolerance=ARM_POSE_TOLERANCE,
        rotation_tolerance=ARM_ROTATION_TOLERANCE,
        command_id=None,
    ):
        target_pose = self._normalize_pose(target_pose)

        timeout = float(timeout)
        position_tolerance = float(position_tolerance)
        rotation_tolerance = float(rotation_tolerance)

        if timeout <= 0:
            raise ValueError("timeout 必須大於 0")

        if position_tolerance <= 0:
            raise ValueError(
                "position_tolerance 必須大於 0"
            )

        if rotation_tolerance <= 0:
            raise ValueError(
                "rotation_tolerance 必須大於 0"
            )

        start_time = time.monotonic()
        current_pose = None

        while time.monotonic() - start_time < timeout:
            if (
                command_id is not None
                and not self._check_command_id(command_id)
            ):
                logger.info(
                    "[UR7e] pose wait interrupted "
                    "command_id=%s",
                    command_id,
                )
                return False

            try:
                state = self._get_receive_state()
                current_pose = state["pose"]
            except Exception as exc:
                logger.warning(
                    "[UR7e] wait pose state unavailable: %s",
                    exc,
                )
                time.sleep(0.1)
                continue

            if (
                not isinstance(current_pose, (list, tuple))
                or len(current_pose) != 6
            ):
                time.sleep(0.1)
                continue

            current_pose = self._normalize_pose(
                current_pose
            )

            position_error = self.pose_distance(
                current_pose,
                target_pose,
            )

            rotation_error = self.rotation_distance(
                current_pose,
                target_pose,
            )

            if (
                position_error <= position_tolerance
                and rotation_error <= rotation_tolerance
            ):
                return True

            time.sleep(0.1)

        logger.warning(
            "[UR7e] wait_until_pose_reached timeout"
        )
        logger.warning(
            "[UR7e] target pose: %s",
            target_pose,
        )
        logger.warning(
            "[UR7e] current pose: %s",
            current_pose,
        )

        if current_pose is not None:
            logger.warning(
                "[UR7e] position error: %s",
                self.pose_distance(
                    current_pose,
                    target_pose,
                ),
            )
            logger.warning(
                "[UR7e] rotation error: %s",
                self.rotation_distance(
                    current_pose,
                    target_pose,
                ),
            )

        return False

    def wait_until_joints_reached(
        self,
        target_joints,
        timeout=ARM_WAIT_TIMEOUT,
        tolerance=ARM_JOINT_TOLERANCE,
        command_id=None,
    ):
        target_joints = self._normalize_joints(
            target_joints
        )

        timeout = float(timeout)
        tolerance = float(tolerance)

        if timeout <= 0:
            raise ValueError("timeout 必須大於 0")

        if tolerance <= 0:
            raise ValueError("tolerance 必須大於 0")

        start_time = time.monotonic()
        current_joints = None

        while time.monotonic() - start_time < timeout:
            if (
                command_id is not None
                and not self._check_command_id(command_id)
            ):
                logger.info(
                    "[UR7e] joints wait interrupted "
                    "command_id=%s",
                    command_id,
                )
                return False

            try:
                state = self._get_receive_state()
                current_joints = state["joints"]
            except Exception as exc:
                logger.warning(
                    "[UR7e] wait joints state unavailable: %s",
                    exc,
                )
                time.sleep(0.1)
                continue

            if (
                not isinstance(current_joints, (list, tuple))
                or len(current_joints) != self.ARM_DOF
            ):
                time.sleep(0.1)
                continue

            current_joints = self._normalize_joints(
                current_joints
            )

            error_value = self.joint_distance(
                current_joints,
                target_joints,
            )

            if error_value <= tolerance:
                return True

            time.sleep(0.1)

        logger.warning(
            "[UR7e] wait_until_joints_reached timeout"
        )
        logger.warning(
            "[UR7e] target joints: %s",
            target_joints,
        )
        logger.warning(
            "[UR7e] current joints: %s",
            current_joints,
        )

        if current_joints is not None:
            logger.warning(
                "[UR7e] joint error: %s",
                self.joint_distance(
                    current_joints,
                    target_joints,
                ),
            )

        return False

    def shutdown(self):
        """
        正常關閉 UR5 Driver。

        關閉順序：
        1. 停止 Receive monitor
        2. 等待 Receive thread 完整結束
        3. Receive monitor 自己 disconnect RTDEReceiveInterface
        4. 關閉 RTDEControlInterface

        此 function 可重複呼叫。
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return True

            logger.info(
                "[UR7e] shutting down driver: %s",
                self.ip,
            )

            try:
                # 先停止 Receive monitor。
                #
                # _stop_receive_monitor() 會：
                #   set stop_event
                #   join thread
                #
                # monitor 的 finally 區塊會自行 disconnect Receive。
                self._stop_receive_monitor()

                # Receive 完全結束後才關閉 Control。
                with self._motion_lock:
                    self._disconnect_control()

                self._reset_receive_state(
                    error="UR5 driver shutdown"
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