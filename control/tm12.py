import logging
import math
import threading
import time

from control.tm_protocol import (
    TMProtocol,
    TMProtocolError,
)


logger = logging.getLogger(__name__)


DEFAULT_SPEED = 0.10
DEFAULT_ACCELERATION = 0.20
DEFAULT_TRAJECTORY_DT = 0.10
ARM_WAIT_TIMEOUT = 20.0
ARM_POSE_TOLERANCE = 0.005       # m
ARM_ROTATION_TOLERANCE = 0.03    # rad
ARM_JOINT_TOLERANCE = 0.003      # rad

DEFAULT_JOG_CYCLE_TIME = 0.05
DEFAULT_JOG_LINEAR_SPEED = 0.05       # m/s
DEFAULT_JOG_ANGULAR_SPEED = 0.10      # rad/s
JOG_THREAD_JOIN_TIMEOUT = 2.0         # s

# Techman TM Expression uses mm and degrees for Cartesian values.
M_TO_MM = 1000.0
RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0


class tm12Driver:
    """
    Techman TM12 driver for the Robot Brain generic arm interface.

    Transport:
      - SVR / Ethernet Slave : TCP 5891 (state)
      - SCT / Listen Node    : TCP 5890 (motion)

    IMPORTANT:
      1. TMflow Ethernet Slave must be enabled for state reads.
      2. A TMflow project containing a Listen Node must be running for motion.
      3. Public Robot Brain units remain identical to the UR drivers:
           pose   = [x, y, z, rx, ry, rz] in metres / radians
           joints = radians
      4. TMflow protocol values are converted internally to mm / degrees.

    This driver uses the pure-Python TMProtocol transport. Jog remains the verified
    small-step Cartesian PTP implementation; freedrive is intentionally not
    faked and remains unsupported until a verified TM-native hand-guiding path
    is available.
    """

    ARM_DOF = 6

    DRIVER_METADATA = {
        "name": "tm12",
        "manufacturer": "Techman Robot",
        "model": "TM12",
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
            "stop",
            "reconnect",
        ],
        "unsupported_capabilities": [
            "freedrive",
        ],
    }

    def __init__(self, ip, connect_timeout=2.0):
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError("ip must be a non-empty string")

        self.ip = ip.strip()
        self.connect_timeout = float(connect_timeout)
        if self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be > 0")

        self._tm = TMProtocol(
            ip=self.ip,
            connect_timeout=self.connect_timeout,
        )

        self._lock = threading.RLock()
        self._motion_lock = threading.RLock()

        # Jog lifecycle. start_arm_jog() starts one background loop and
        # stop_arm_jog() is the only normal stop condition.
        self._jog_lock = threading.RLock()
        self._jog_thread = None
        self._jog_stop_event = threading.Event()
        self._jog_direction = None

        self._last_error = None

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _number(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _normalize_pose(cls, pose):
        if not isinstance(pose, (list, tuple)) or len(pose) != 6:
            raise ValueError("pose must contain 6 values")
        return [cls._number(f"pose[{i}]", v) for i, v in enumerate(pose)]

    @classmethod
    def _normalize_joints(cls, joints):
        if not isinstance(joints, (list, tuple)) or len(joints) != cls.ARM_DOF:
            raise ValueError(f"joints must contain {cls.ARM_DOF} values")
        return [cls._number(f"joints[{i}]", v) for i, v in enumerate(joints)]

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "on", "yes", "run", "running"}
        return None

    @staticmethod
    def _extract_value(value):
        """Normalize scalar/vector values returned by the TM protocol layer."""
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, dict):
            for key in ("value", "data", "result"):
                if key in value:
                    return value[key]
        return value

    @classmethod
    def _parse_vector(cls, value, expected=6):
        value = cls._extract_value(value)
        if isinstance(value, str):
            text = value.strip().strip("{}[]()")
            parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
            value = parts
        if not isinstance(value, (list, tuple)) or len(value) != expected:
            raise RuntimeError(f"TM value is not a {expected}-element vector: {value!r}")
        return [float(v) for v in value]

    @staticmethod
    def _pose_to_tm(pose):
        pose = tm12Driver._normalize_pose(pose)
        return [
            pose[0] * M_TO_MM,
            pose[1] * M_TO_MM,
            pose[2] * M_TO_MM,
            pose[3] * RAD_TO_DEG,
            pose[4] * RAD_TO_DEG,
            pose[5] * RAD_TO_DEG,
        ]

    @staticmethod
    def _pose_from_tm(pose):
        pose = tm12Driver._parse_vector(pose, 6)
        return [
            pose[0] / M_TO_MM,
            pose[1] / M_TO_MM,
            pose[2] / M_TO_MM,
            pose[3] * DEG_TO_RAD,
            pose[4] * DEG_TO_RAD,
            pose[5] * DEG_TO_RAD,
        ]

    @staticmethod
    def _joints_to_tm(joints):
        return [v * RAD_TO_DEG for v in tm12Driver._normalize_joints(joints)]

    @staticmethod
    def _joints_from_tm(joints):
        joints = tm12Driver._parse_vector(joints, 6)
        return [v * DEG_TO_RAD for v in joints]

    @staticmethod
    def _tm_speed(speed):
        """
        Preserve the existing Robot Brain convention where 0.10 means a low
        normalized speed. TM PTP accepts a normalized velocity
        value (the upstream example uses 0.10). Clamp to (0, 1].
        """
        speed = float(speed if speed is not None else DEFAULT_SPEED)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be > 0")
        return min(speed, 1.0)

    @staticmethod
    def _tm_acceleration_ms(acceleration):
        """
        TM PTP acceleration parameter is time-to-top-speed in ms.
        Robot Brain currently supplies acceleration in UR-style units, so use
        a conservative conversion: 0.20 -> 200 ms, minimum 100 ms.
        """
        acceleration = float(
            acceleration if acceleration is not None else DEFAULT_ACCELERATION
        )
        if not math.isfinite(acceleration) or acceleration <= 0:
            raise ValueError("acceleration must be > 0")
        return max(100, int(round(acceleration * 1000.0)))

    def _svr_get(self, item):
        return self._extract_value(
            self._tm.get_value(item)
        )

    def _sct_call(self, function_name, *args):
        return self._tm.call(
            function_name,
            *args,
        )

    def _safe_svr_get(self, item, default=None):
        try:
            return self._svr_get(item)
        except Exception:
            logger.debug("[TM12] SVR item unavailable: %s", item, exc_info=True)
            return default

    # ========================================================
    # Connection lifecycle
    # ========================================================

    def reconnect_arm(self):
        # A successful TMSVR query is the health-check operation.
        self._svr_get("Robot_Model")
        self._last_error = None
        return True

    # ========================================================
    # State
    # ========================================================

    def get_arm_pose(self):
        return self._pose_from_tm(
            self._svr_get("Coord_Robot_Tool")
        )

    def get_arm_joints(self):
        return self._joints_from_tm(
            self._svr_get("Joint_Angle")
        )

    def get_arm_status(self):
        try:
            pose = self.get_arm_pose()
            joints = self.get_arm_joints()
            connected = True
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            return {
                "connected": False,
                "ready": False,
                "moving": None,
                "protective_stop": None,
                "emergency_stop": None,
                "fault": None,
                "program_running": None,
                "arm_mode": None,
                "safety_mode": None,
                "pose": None,
                "joints": None,
                "receive_error": self._last_error,
            }

        robot_link = self._as_bool(self._safe_svr_get("Robot_Link", True))
        robot_error = self._as_bool(self._safe_svr_get("Robot_Error", False))
        project_run = self._as_bool(self._safe_svr_get("Project_Run", None))
        project_pause = self._as_bool(self._safe_svr_get("Project_Pause", None))

        # Not every TMflow data table exposes dedicated E-stop/protective-stop
        # variables. Keep unknown states as None instead of inventing them.
        emergency_stop = self._as_bool(self._safe_svr_get("ESTOP", None))
        protective_stop = self._as_bool(self._safe_svr_get("Safeguard_A", None))

        ready = bool(
            connected
            and robot_link is not False
            and robot_error is not True
            and emergency_stop is not True
            and protective_stop is not True
        )

        return {
            "connected": connected,
            "ready": ready,
            "moving": None,
            "protective_stop": protective_stop,
            "emergency_stop": emergency_stop,
            "fault": robot_error,
            "program_running": project_run,
            "arm_mode": "TMflow",
            "safety_mode": None,
            "pose": pose,
            "joints": joints,
            "project_paused": project_pause,
            "receive_error": None,
        }

    # ========================================================
    # Motion
    # ========================================================

    def move_arm_joints(
        self,
        joints,
        speed=None,
        acceleration=None,
        wait=True,
    ):
        joints = self._normalize_joints(joints)
        if not isinstance(wait, bool):
            raise ValueError("wait must be bool")

        target_deg = self._joints_to_tm(joints)
        velocity = self._tm_speed(speed)
        accel_ms = self._tm_acceleration_ms(acceleration)

        with self._motion_lock:
            self._sct_call(
                "PTP",
                "JPP",
                target_deg,
                int(round(100.0 * velocity)),
                accel_ms,
                0,
                True,
            )

        if not wait:
            return True
        return self._wait_until_joints_reached(joints)

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
        target = self._normalize_pose([x, y, z, rx, ry, rz])
        if not isinstance(wait, bool):
            raise ValueError("wait must be bool")

        target_tm = self._pose_to_tm(target)
        velocity = self._tm_speed(speed)
        accel_ms = self._tm_acceleration_ms(acceleration)

        with self._motion_lock:
            self._sct_call(
                "PTP",
                "CPP",
                target_tm,
                int(round(100.0 * velocity)),
                accel_ms,
                0,
                True,
            )

        if not wait:
            return True
        return self._wait_until_pose_reached(target)

    def move_arm_joint_trajectory(
        self,
        joint_trajectory,
        dt=None,
        speed=None,
        acceleration=None,
        wait=True,
        move_to_start=False,
        move_to_start_speed=None,
        move_to_start_acceleration=None,
        lookahead_time=None,
        gain=None,
    ):
        if dt is None:
            dt = DEFAULT_TRAJECTORY_DT
        dt = self._number("dt", dt)
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if not isinstance(joint_trajectory, (list, tuple)) or not joint_trajectory:
            raise ValueError("joint_trajectory must not be empty")

        points = []
        for index, sample in enumerate(joint_trajectory):
            if isinstance(sample, dict) and "q_arm" in sample:
                q = sample["q_arm"]
            elif (
                isinstance(sample, (list, tuple))
                and len(sample) == 2
                and isinstance(sample[1], (list, tuple))
            ):
                q = sample[1]
            else:
                q = sample
            points.append(self._normalize_joints(q))

        # TM Listen PTP is command/queue based, not a UR servoJ equivalent.
        # Execute discrete points conservatively. For high-rate VLA streaming,
        # implement a separate RTRS-backed driver path instead of pretending
        # this is a realtime servo trajectory.
        if move_to_start:
            self.move_arm_joints(
                points[0],
                speed=move_to_start_speed or speed,
                acceleration=move_to_start_acceleration or acceleration,
                wait=True,
            )

        for point in points:
            self.move_arm_joints(
                point,
                speed=speed,
                acceleration=acceleration,
                wait=True,
            )

        return True if not wait else self._wait_until_joints_reached(points[-1])

    def move_arm_pose_trajectory(
        self,
        pose_trajectory,
        dt=None,
        speed=None,
        acceleration=None,
        wait=True,
        move_to_start=False,
        move_to_start_speed=None,
        move_to_start_acceleration=None,
        lookahead_time=None,
        gain=None,
    ):
        if dt is None:
            dt = DEFAULT_TRAJECTORY_DT
        dt = self._number("dt", dt)
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if not isinstance(pose_trajectory, (list, tuple)) or not pose_trajectory:
            raise ValueError("pose_trajectory must not be empty")

        points = []
        for sample in pose_trajectory:
            if isinstance(sample, dict) and "pose" in sample:
                pose = sample["pose"]
            elif (
                isinstance(sample, (list, tuple))
                and len(sample) == 2
                and isinstance(sample[1], (list, tuple))
            ):
                pose = sample[1]
            else:
                pose = sample
            points.append(self._normalize_pose(pose))

        if move_to_start:
            self.move_arm_pose(
                *points[0],
                speed=move_to_start_speed or speed,
                acceleration=move_to_start_acceleration or acceleration,
                wait=True,
            )

        for pose in points:
            self.move_arm_pose(
                *pose,
                speed=speed,
                acceleration=acceleration,
                wait=True,
            )

        return True if not wait else self._wait_until_pose_reached(points[-1])

    # ============================================================
    # Jog
    # ============================================================

    @staticmethod
    def _apply_jog_step(
        pose,
        direction,
        linear_step,
        angular_step,
    ):
        target = list(pose)

        if direction == "x+":
            target[0] += linear_step

        elif direction == "x-":
            target[0] -= linear_step

        elif direction == "y+":
            target[1] += linear_step

        elif direction == "y-":
            target[1] -= linear_step

        elif direction == "z+":
            target[2] += linear_step

        elif direction == "z-":
            target[2] -= linear_step

        elif direction == "rx+":
            target[3] += angular_step

        elif direction == "rx-":
            target[3] -= angular_step

        elif direction == "ry+":
            target[4] += angular_step

        elif direction == "ry-":
            target[4] -= angular_step

        elif direction == "rz+":
            target[5] += angular_step

        elif direction == "rz-":
            target[5] -= angular_step

        else:
            raise ValueError(
                f"invalid jog direction: {direction}"
            )

        return target


    def _jog_loop(
        self,
        direction,
        linear_speed,
        angular_speed,
        acceleration,
    ):
        """
        Continuous Cartesian jog for TM12.

        Implementation:
            - Start from current TCP pose.
            - Continuously advance the commanded target pose.
            - Send short blended PTP commands without waiting for each
            intermediate target to be reached.
            - stop_arm_jog() terminates generation and clears the TM
            motion buffer.

        Public units:
            linear_speed  : m/s
            angular_speed : rad/s
        """

        cycle_time = DEFAULT_JOG_CYCLE_TIME

        linear_step = (
            linear_speed
            * cycle_time
        )

        angular_step = (
            angular_speed
            * cycle_time
        )

        try:
            target_pose = (
                self.get_arm_pose()
            )

            #
            # PTP execution velocity.
            #
            # Robot Brain speed convention:
            #   0.0 ~ 1.0
            #
            # For Cartesian directions use linear_speed.
            # For rotational directions use angular_speed.
            #
            if direction.startswith("r"):
                ptp_speed = self._tm_speed(
                    angular_speed
                )
            else:
                ptp_speed = self._tm_speed(
                    linear_speed
                )

            velocity_percent = max(
                1,
                min(
                    100,
                    int(
                        round(
                            ptp_speed * 100.0
                        )
                    ),
                ),
            )

            accel_ms = (
                self._tm_acceleration_ms(
                    acceleration
                )
            )

            while not self._jog_stop_event.is_set():

                cycle_started = (
                    time.monotonic()
                )

                #
                # Advance from the previous commanded target,
                # not from the measured pose.
                #
                # This avoids:
                #   actual -> target
                #   actual -> target
                #
                # network/state-update jitter.
                #
                target_pose = (
                    self._apply_jog_step(
                        target_pose,
                        direction,
                        linear_step,
                        angular_step,
                    )
                )

                target_tm = (
                    self._pose_to_tm(
                        target_pose
                    )
                )

                #
                # TM PTP:
                #
                # PTP(
                #     "CPP",
                #     target,
                #     velocity,
                #     acceleration_time,
                #     blend_percentage,
                #     precise_positioning
                # )
                #
                # Jog uses full blending because intermediate points
                # are not stopping points.
                #
                with self._motion_lock:
                    self._tm.call(
                        "PTP",
                        "CPP",
                        target_tm,
                        velocity_percent,
                        accel_ms,
                        100,
                        False,
                    )

                elapsed = (
                    time.monotonic()
                    - cycle_started
                )

                remaining = (
                    cycle_time
                    - elapsed
                )

                if remaining > 0:
                    self._jog_stop_event.wait(
                        remaining
                    )

        except Exception as exc:
            self._last_error = str(
                exc
            )

            logger.exception(
                "[TM12] jog failed: "
                "direction=%s",
                direction,
            )

        finally:
            with self._jog_lock:
                self._jog_direction = None


    def start_arm_jog(
        self,
        direction,
        linear_speed=None,
        angular_speed=None,
        acceleration=None,
        timeout=None,
    ):
        """
        Start continuous Cartesian jog.

        Movement continues until stop_arm_jog() is called.

        Repeated start_arm_jog() calls for the same direction are
        idempotent, allowing the web UI to repeatedly send start
        commands while a button is held.

        timeout is accepted for compatibility with the generic
        Robot Brain ARM contract but does not automatically stop
        the TM12 jog.
        """

        direction = (
            str(direction)
            .strip()
            .lower()
        )

        valid_directions = {
            "x+",
            "x-",
            "y+",
            "y-",
            "z+",
            "z-",
            "rx+",
            "rx-",
            "ry+",
            "ry-",
            "rz+",
            "rz-",
        }

        if direction not in valid_directions:
            raise ValueError(
                f"invalid jog direction: "
                f"{direction}"
            )

        if linear_speed is None:
            linear_speed = (
                DEFAULT_JOG_LINEAR_SPEED
            )

        if angular_speed is None:
            angular_speed = (
                DEFAULT_JOG_ANGULAR_SPEED
            )

        if acceleration is None:
            acceleration = (
                DEFAULT_ACCELERATION
            )

        linear_speed = self._number(
            "linear_speed",
            linear_speed,
        )

        angular_speed = self._number(
            "angular_speed",
            angular_speed,
        )

        acceleration = self._number(
            "acceleration",
            acceleration,
        )

        if linear_speed <= 0:
            raise ValueError(
                "linear_speed must be > 0"
            )

        if angular_speed <= 0:
            raise ValueError(
                "angular_speed must be > 0"
            )

        if acceleration <= 0:
            raise ValueError(
                "acceleration must be > 0"
            )

        #
        # timeout remains part of the generic Robot Brain API,
        # but TM12 Jog is explicitly stopped by stop_arm_jog().
        #
        if timeout is not None:
            timeout = self._number(
                "timeout",
                timeout,
            )

            if timeout <= 0:
                raise ValueError(
                    "timeout must be > 0"
                )

        #
        # --------------------------------------------------------
        # Existing jog lifecycle
        # --------------------------------------------------------
        #
        with self._jog_lock:

            current_thread = (
                self._jog_thread
            )

            #
            # Frontend may send start every 100 ms.
            #
            # Same direction already running:
            # do nothing.
            #
            if (
                current_thread is not None
                and current_thread.is_alive()
                and self._jog_direction
                    == direction
                and not
                    self._jog_stop_event.is_set()
            ):
                return True

            #
            # Different direction:
            # terminate current jog first.
            #
            if (
                current_thread is not None
                and current_thread.is_alive()
            ):
                self._jog_stop_event.set()

        if (
            current_thread is not None
            and current_thread.is_alive()
            and current_thread
                is not threading.current_thread()
        ):
            current_thread.join(
                timeout=
                    JOG_THREAD_JOIN_TIMEOUT
            )

            if current_thread.is_alive():
                raise RuntimeError(
                    "previous TM12 jog "
                    "thread did not stop"
                )

            #
            # Remove pending commands from previous direction.
            #
            try:
                with self._motion_lock:
                    self._tm.stop_motion()

            except Exception as exc:
                self._last_error = str(
                    exc
                )

                raise RuntimeError(
                    "failed to stop previous "
                    f"TM12 jog: {exc}"
                ) from exc

        #
        # --------------------------------------------------------
        # Start new jog
        # --------------------------------------------------------
        #
        with self._jog_lock:

            self._jog_stop_event.clear()

            self._jog_direction = (
                direction
            )

            self._last_error = None

            thread = threading.Thread(
                target=
                    self._jog_loop,
                args=(
                    direction,
                    linear_speed,
                    angular_speed,
                    acceleration,
                ),
                name=(
                    f"TM12Jog-"
                    f"{self.ip}-"
                    f"{direction}"
                ),
                daemon=True,
            )

            self._jog_thread = thread

            thread.start()

        return True


    def stop_arm_jog(
        self,
    ):
        """
        Stop continuous TM12 Jog.

        1. Stop generating new PTP targets.
        2. Clear queued TM motion commands.
        3. Wait for Jog worker termination.
        """

        with self._jog_lock:

            self._jog_stop_event.set()

            thread = (
                self._jog_thread
            )

        #
        # Immediately stop queued movement.
        #
        try:
            with self._motion_lock:
                self._tm.stop_motion()

        except Exception as exc:
            self._last_error = str(
                exc
            )

            logger.exception(
                "[TM12] failed to clear "
                "jog motion buffer"
            )

            return False

        if (
            thread is not None
            and thread.is_alive()
            and thread
                is not threading.current_thread()
        ):
            thread.join(
                timeout=
                    JOG_THREAD_JOIN_TIMEOUT
            )

            if thread.is_alive():

                self._last_error = (
                    "TM12 jog thread "
                    "did not stop within "
                    f"{JOG_THREAD_JOIN_TIMEOUT:.1f}s"
                )

                return False

        with self._jog_lock:

            if (
                self._jog_thread
                is thread
            ):
                self._jog_thread = None

            self._jog_direction = None

        return True


    def start_arm_freedrive(self):
        raise NotImplementedError(
            "TM12 freedrive/direct-teach is not mapped to the generic driver yet"
        )

    def stop_arm_freedrive(self):
        return True

    def stop_arm(self, acceleration=None):
        """Stop software jog and pause the active TMflow project."""
        self.stop_arm_jog()

        with self._motion_lock:
            try:
                self._tm.pause_project()
                return True
            except Exception as exc:
                self._last_error = str(exc)
                raise RuntimeError(f"TM12 stop/pause failed: {exc}") from exc

    # ========================================================
    # Reached verification
    # ========================================================

    @staticmethod
    def _rotation_distance(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3, 6)))

    def _wait_until_pose_reached(self, target, timeout=ARM_WAIT_TIMEOUT):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            actual = self.get_arm_pose()
            xyz_error = math.sqrt(sum((actual[i] - target[i]) ** 2 for i in range(3)))
            rot_error = self._rotation_distance(actual, target)
            if xyz_error <= ARM_POSE_TOLERANCE and rot_error <= ARM_ROTATION_TOLERANCE:
                return True
            time.sleep(0.05)
        return False

    def _wait_until_joints_reached(self, target, timeout=ARM_WAIT_TIMEOUT):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            actual = self.get_arm_joints()
            error = max(abs(a - b) for a, b in zip(actual, target))
            if error <= ARM_JOINT_TOLERANCE:
                return True
            time.sleep(0.05)
        return False

    def shutdown(self):
        self.stop_arm_jog()
        self._tm.close()
        return True
