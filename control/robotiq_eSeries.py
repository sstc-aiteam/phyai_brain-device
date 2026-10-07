import logging
import math
import threading
import time

import rtde_receive

from control.robotiq_preamble import ROBOTIQ_PREAMBLE


logger = logging.getLogger(__name__)


MIN_RAW = 0
MAX_RAW = 255

DEFAULT_TIMEOUT = 5.0
DEFAULT_STATUS_CACHE_TTL = 0.15
DEFAULT_REGISTER_BASE = 36


class RobotiqESeriesDriver:
    """
    Robotiq 2F gripper controlled by URScript.

    Public gripper contract:
        position: 0.0 = fully open, 1.0 = fully closed
        speed:    0.0 ~ 1.0
        force:    0.0 ~ 1.0
    """

    DRIVER_METADATA = {
        "name": "robotiq_urscript",
        "manufacturer": "Robotiq",
        "model": "2F Gripper",
        "interface": "urscript_via_arm_rtde_gateway",
        "capabilities": {
            "status": True,
            "position": True,
            "speed": True,
            "force": True,
            "object_detection": True,
            "open": True,
            "close": True,
            "stop": True,
        },
    }

    def __init__(
        self,
        host,
        port=None,
        timeout=DEFAULT_TIMEOUT,
        auto_activate=True,
        arm_name=None,
        register_base=DEFAULT_REGISTER_BASE,
        status_cache_ttl=DEFAULT_STATUS_CACHE_TTL,
    ):
        """
        Parameters
        ----------
        host:
            UR controller IP.

        port:
            Accepted for config compatibility only.  URScript mode does not
            connect from Python to Robotiq port 63352.

        arm_name:
            Optional explicit arm name from config.ARMS.  If omitted, this
            driver resolves the arm by matching config.ARMS[*].kwargs.ip to
            host.

        register_base:
            Four consecutive RTDE output integer registers are used:
                base + 0 : actual gripper position [0..255]
                base + 1 : activated [0/1]
                base + 2 : OBJ status [0..3]
                base + 3 : FLT code
        """
        _ = port

        self._host = str(host).strip()
        if not self._host:
            raise ValueError("host must not be empty")

        self._timeout = self._normalize_positive_number(
            "timeout",
            timeout,
        )

        self._auto_activate = bool(auto_activate)

        self._arm_name = (
            None
            if arm_name is None
            else str(arm_name).strip().lower()
        )

        self._register_base = int(register_base)
        if self._register_base < 0:
            raise ValueError("register_base must be >= 0")

        self.REG_POSITION = self._register_base
        self.REG_ACTIVATED = self._register_base + 1
        self.REG_OBJECT_STATUS = self._register_base + 2
        self.REG_FAULT_CODE = self._register_base + 3

        self._status_cache_ttl = self._normalize_nonnegative_number(
            "status_cache_ttl",
            status_cache_ttl,
        )

        self._lock = threading.RLock()
        self._receive_lock = threading.RLock()

        self._rtde_r = None
        self._arm_driver = None
        self._resolved_arm_name = None

        self._requested_position = None
        self._speed = None
        self._force = None

        self._status_cache = None
        self._status_cache_time = None

        if self._auto_activate:
            try:
                self.activate_gripper(
                    timeout=self._timeout
                )
            except Exception:
                # Keep application startup alive when robot/gripper is
                # temporarily unavailable.
                logger.exception(
                    "[Robotiq URScript] auto activation failed: host=%s",
                    self._host,
                )

    # ========================================================
    # Validation / conversion
    # ========================================================

    @staticmethod
    def _normalize_ratio(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a number"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0.0 and 1.0"
            )

        return value

    @staticmethod
    def _normalize_positive_number(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a number"
            ) from exc

        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be greater than 0"
            )

        return value

    @staticmethod
    def _normalize_nonnegative_number(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a number"
            ) from exc

        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be >= 0"
            )

        return value

    @staticmethod
    def _ratio_to_raw(value):
        return int(
            round(
                float(value)
                * MAX_RAW
            )
        )

    @staticmethod
    def _raw_to_ratio(value):
        return float(value) / float(MAX_RAW)

    # ========================================================
    # Arm resolution / shared RTDE control owner
    # ========================================================

    def _resolve_arm_driver(self):
        """
        Resolve the existing arm driver managed by control.loader.

        The gripper does NOT create RTDEControlInterface itself.
        """
        with self._lock:
            if self._arm_driver is not None:
                return self._arm_driver

            # Lazy imports avoid import cycles during loader startup.
            import config
            from control import loader

            arm_name = self._arm_name

            if arm_name is None:
                matched = []

                for name, arm_config in config.ARMS.items():
                    kwargs = arm_config.get(
                        "kwargs",
                        {},
                    )

                    ip = kwargs.get("ip")

                    if (
                        ip is not None
                        and str(ip).strip() == self._host
                    ):
                        matched.append(name)

                if len(matched) == 1:
                    arm_name = matched[0]

                elif not matched:
                    raise RuntimeError(
                        "Unable to resolve arm for Robotiq URScript driver: "
                        f"no config.ARMS entry uses IP {self._host}. "
                        "Pass arm_name explicitly in gripper kwargs."
                    )

                else:
                    raise RuntimeError(
                        "Unable to resolve arm for Robotiq URScript driver: "
                        f"multiple arms use IP {self._host}: {matched}. "
                        "Pass arm_name explicitly in gripper kwargs."
                    )

            arm = loader.get_arm_driver(
                arm_name
            )

            required = (
                "_motion_lock",
                "_motion_mode",
                "_call_control",
                "_ensure_control_ready",
            )

            missing = [
                name
                for name in required
                if not hasattr(arm, name)
            ]

            if missing:
                raise RuntimeError(
                    f"arm driver '{arm_name}' does not expose the "
                    "URScript coordination hooks required by "
                    f"robotiq_urscript: {', '.join(missing)}"
                )

            self._arm_driver = arm
            self._resolved_arm_name = arm_name

            return arm

    # ========================================================
    # Receive-only RTDE status channel
    # ========================================================

    def _get_receive(self):
        with self._receive_lock:
            if self._rtde_r is not None:
                try:
                    if self._rtde_r.isConnected():
                        return self._rtde_r
                except Exception:
                    pass

                try:
                    self._rtde_r.disconnect()
                except Exception:
                    pass

                self._rtde_r = None

            self._rtde_r = (
                rtde_receive
                .RTDEReceiveInterface(
                    self._host,
                    -1.0,
                    [],
                    False,
                    True,
                )
            )

            return self._rtde_r

    def _disconnect_receive(self):
        with self._receive_lock:
            rtde_r = self._rtde_r
            self._rtde_r = None

            if rtde_r is not None:
                try:
                    rtde_r.disconnect()
                except Exception:
                    logger.debug(
                        "[Robotiq URScript] RTDE receive disconnect failed",
                        exc_info=True,
                    )

        return True

    def reconnect_gripper(self):
        """
        Rebuild only the gripper's receive-only RTDE channel.

        Arm RTDE Control remains owned by the Arm Driver.
        """
        self._disconnect_receive()
        self._arm_driver = None
        self._resolved_arm_name = None

        self._resolve_arm_driver()
        self._get_receive()

        return True

    # ========================================================
    # RTDE recovery
    # ========================================================

    def _recover_arm_control(self, arm):
        """
        Restore ur_rtde's default control script after custom URScript.

        First try the existing control object.  If that fails, delegate the
        full reconnect to the Arm Driver.
        """
        try:
            arm._ensure_control_ready()
            return True

        except Exception as first_exc:
            logger.warning(
                "[Robotiq URScript] control script recovery failed; "
                "trying arm reconnect: %s",
                first_exc,
            )

        reconnect = getattr(
            arm,
            "reconnect_arm",
            None,
        )

        if not callable(reconnect):
            raise RuntimeError(
                "Arm RTDE control script could not be recovered and "
                "arm driver does not support reconnect_arm()"
            )

        result = reconnect()

        if not result:
            raise RuntimeError(
                "Arm reconnect returned False after Robotiq URScript"
            )

        arm._ensure_control_ready()
        return True

    # ========================================================
    # URScript transaction
    # ========================================================

    def _execute_script(
        self,
        function_name,
        body,
        restore_freedrive=True,
    ):
        """
        Execute a Robotiq custom script through the existing Arm Driver.

        Transaction:
            acquire arm motion lock
            -> reject active normal motion
            -> temporarily leave Freedrive if needed
            -> sendCustomScriptFunction()
            -> recover RTDE control script
            -> restore Freedrive if it was active
        """
        if (
            not isinstance(function_name, str)
            or not function_name.strip()
        ):
            raise ValueError(
                "function_name must be a non-empty string"
            )

        if not isinstance(body, str) or not body.strip():
            raise ValueError(
                "body must be a non-empty string"
            )

        arm = self._resolve_arm_driver()

        script = (
            ROBOTIQ_PREAMBLE
            + "\n"
            + body.strip()
            + "\n"
        )

        with self._lock:
            with arm._motion_lock:
                previous_mode = arm._motion_mode

                # Freedrive can be deterministically restored.
                was_freedrive = (
                    previous_mode == "freedrive"
                )

                # Normal motion cannot be safely resumed after a custom
                # primary URScript replaces the RTDE control program.
                if (
                    previous_mode is not None
                    and not was_freedrive
                ):
                    raise RuntimeError(
                        "Robotiq URScript command rejected because arm "
                        f"motion is active: {previous_mode}. "
                        "Wait for the arm motion to finish or stop it first."
                    )

                script_error = None
                recovery_error = None
                freedrive_restore_error = None

                if was_freedrive:
                    stop_freedrive = getattr(
                        arm,
                        "stop_arm_freedrive",
                        None,
                    )

                    if not callable(stop_freedrive):
                        raise RuntimeError(
                            "Arm is in Freedrive but driver cannot "
                            "stop_arm_freedrive()"
                        )

                    if not stop_freedrive():
                        raise RuntimeError(
                            "Unable to leave Freedrive before "
                            "Robotiq URScript"
                        )

                    # Let controller finish the mode transition.
                    time.sleep(0.05)

                try:
                    result = arm._call_control(
                        "sendCustomScriptFunction",
                        function_name.strip(),
                        script,
                    )

                    if not bool(result):
                        raise RuntimeError(
                            "sendCustomScriptFunction returned False: "
                            f"{function_name}"
                        )

                except Exception as exc:
                    script_error = exc

                finally:
                    # A custom URScript program can stop the default ur_rtde
                    # control program.  Recover it regardless of command result.
                    try:
                        self._recover_arm_control(
                            arm
                        )
                    except Exception as exc:
                        recovery_error = exc

                    if (
                        was_freedrive
                        and restore_freedrive
                        and recovery_error is None
                    ):
                        try:
                            start_freedrive = getattr(
                                arm,
                                "start_arm_freedrive",
                                None,
                            )

                            if not callable(
                                start_freedrive
                            ):
                                raise RuntimeError(
                                    "Arm driver cannot "
                                    "start_arm_freedrive()"
                                )

                            if not start_freedrive():
                                raise RuntimeError(
                                    "start_arm_freedrive() "
                                    "returned False"
                                )

                        except Exception as exc:
                            freedrive_restore_error = exc

                if recovery_error is not None:
                    raise RuntimeError(
                        "Robotiq URScript finished/failed, but RTDE "
                        "control recovery also failed"
                    ) from recovery_error

                if freedrive_restore_error is not None:
                    raise RuntimeError(
                        "Robotiq URScript completed and RTDE control was "
                        "recovered, but Freedrive could not be restored"
                    ) from freedrive_restore_error

                if script_error is not None:
                    raise script_error

                return True

    # ========================================================
    # Status register helpers
    # ========================================================

    def _status_write_body(self):
        return f"""
rq_status_pos = rq_current_pos()
rq_status_obj = rq_get_var("OBJ")
rq_status_flt = rq_get_var("FLT")

if rq_is_gripper_activated():
    rq_status_act = 1
else:
    rq_status_act = 0
end

write_output_integer_register(
    {self.REG_POSITION},
    rq_status_pos
)

write_output_integer_register(
    {self.REG_ACTIVATED},
    rq_status_act
)

write_output_integer_register(
    {self.REG_OBJECT_STATUS},
    rq_status_obj
)

write_output_integer_register(
    {self.REG_FAULT_CODE},
    rq_status_flt
)
"""

    def _refresh_status_registers(self):
        self._execute_script(
            "robotiq_read_status",
            self._status_write_body(),
            restore_freedrive=True,
        )

    def _read_status_registers(self):
        rtde_r = self._get_receive()

        position = int(
            rtde_r.getOutputIntRegister(
                self.REG_POSITION
            )
        )

        activated = bool(
            int(
                rtde_r.getOutputIntRegister(
                    self.REG_ACTIVATED
                )
            )
        )

        object_status = int(
            rtde_r.getOutputIntRegister(
                self.REG_OBJECT_STATUS
            )
        )

        fault_code = int(
            rtde_r.getOutputIntRegister(
                self.REG_FAULT_CODE
            )
        )

        # Sanity-check register content.  This also helps detect wrong/conflicting
        # register allocation.
        if not MIN_RAW <= position <= MAX_RAW:
            raise RuntimeError(
                f"invalid Robotiq position register value: {position}"
            )

        if object_status not in (0, 1, 2, 3):
            raise RuntimeError(
                f"invalid Robotiq OBJ register value: {object_status}"
            )

        return (
            position,
            activated,
            object_status,
            fault_code,
        )

    @staticmethod
    def _decode_object_status(object_status):
        return {
            0: "moving",
            1: "object_detected_opening",
            2: "object_detected_closing",
            3: "position_reached",
        }.get(
            object_status,
            "unknown",
        )

    def _cache_status(self, status):
        self._status_cache = dict(status)
        self._status_cache_time = time.monotonic()

    def _get_cached_status(self):
        if (
            self._status_cache is None
            or self._status_cache_time is None
        ):
            return None

        age = (
            time.monotonic()
            - self._status_cache_time
        )

        if age > self._status_cache_ttl:
            return None

        return dict(
            self._status_cache
        )

    # ========================================================
    # Activation
    # ========================================================

    def activate_gripper(
        self,
        timeout=DEFAULT_TIMEOUT,
    ):
        timeout = self._normalize_positive_number(
            "timeout",
            timeout,
        )

        # rq_activate_and_wait() contains its own bounded wait.  Keep timeout
        # argument for common driver compatibility.
        _ = timeout

        body = (
            """
rq_reset()
rq_activate_and_wait()
"""
            + self._status_write_body()
        )

        self._execute_script(
            "robotiq_activate",
            body,
            restore_freedrive=True,
        )

        # Populate Python-side cache immediately.
        try:
            status = self._status_from_registers(
                refresh=False
            )
            self._cache_status(
                status
            )
        except Exception:
            pass

        return True

    # ========================================================
    # Public status
    # ========================================================

    def _status_from_registers(
        self,
        refresh=True,
    ):
        if refresh:
            self._refresh_status_registers()

        (
            raw_position,
            activated,
            object_status,
            fault_code,
        ) = self._read_status_registers()

        fault = (
            fault_code != 0
        )

        status = {
            # Generic gripper_service contract
            "connected": True,
            "ready": (
                activated
                and not fault
            ),
            "moving": (
                object_status == 0
            ),
            "position": self._raw_to_ratio(
                raw_position
            ),
            "object_detected": (
                object_status in (1, 2)
            ),
            "fault": fault,

            # Robotiq diagnostics
            "activated": activated,
            "requested_position":
                self._requested_position,
            "speed":
                self._speed,
            "force":
                self._force,
            "object_status":
                object_status,
            "object_status_name":
                self._decode_object_status(
                    object_status
                ),
            "fault_code":
                fault_code,
            "arm_name":
                self._resolved_arm_name,
        }

        return status

    def get_gripper_status(self):
        cached = self._get_cached_status()

        if cached is not None:
            return cached

        try:
            status = self._status_from_registers(
                refresh=True
            )

            self._cache_status(
                status
            )

            return status

        except Exception as exc:
            logger.warning(
                "[Robotiq URScript] status unavailable: %s",
                exc,
            )

            self._disconnect_receive()

            return {
                "connected": False,
                "ready": False,
                "moving": None,
                "position": None,
                "object_detected": None,
                "fault": None,

                "activated": False,
                "requested_position":
                    self._requested_position,
                "speed":
                    self._speed,
                "force":
                    self._force,
                "object_status":
                    None,
                "object_status_name":
                    "unavailable",
                "fault_code":
                    None,
                "arm_name":
                    self._resolved_arm_name,
            }

    # ========================================================
    # Motion
    # ========================================================

    def move_gripper(
        self,
        position,
        speed=1.0,
        force=0.6,
        wait=True,
        timeout=DEFAULT_TIMEOUT,
    ):
        position = self._normalize_ratio(
            position,
            "position",
        )

        speed = self._normalize_ratio(
            speed,
            "speed",
        )

        force = self._normalize_ratio(
            force,
            "force",
        )

        if not isinstance(wait, bool):
            raise ValueError(
                "wait must be bool"
            )

        timeout = self._normalize_positive_number(
            "timeout",
            timeout,
        )

        raw_position = self._ratio_to_raw(
            position
        )

        raw_speed = self._ratio_to_raw(
            speed
        )

        raw_force = self._ratio_to_raw(
            force
        )

        self._requested_position = position
        self._speed = speed
        self._force = force

        if wait:
            move_line = (
                f"rq_move_and_wait({raw_position})"
            )
        else:
            move_line = (
                f"rq_move({raw_position})"
            )

        body = f"""
if not rq_is_gripper_activated():
    rq_activate_and_wait()
end

rq_set_speed({raw_speed})
rq_set_force({raw_force})
{move_line}
""" + self._status_write_body()

        self._execute_script(
            "robotiq_move",
            body,
            restore_freedrive=True,
        )

        # sendCustomScriptFunction is synchronous for this transaction.
        # timeout is retained for the common gripper interface.  The bounded
        # wait inside rq_move_and_wait() is implemented by the preamble.
        _ = timeout

        try:
            status = self._status_from_registers(
                refresh=False
            )
            self._cache_status(
                status
            )
        except Exception:
            self._status_cache = None
            self._status_cache_time = None

        return True

    def open_gripper(
        self,
        speed=1.0,
        force=0.6,
        wait=True,
        timeout=DEFAULT_TIMEOUT,
    ):
        return self.move_gripper(
            position=0.0,
            speed=speed,
            force=force,
            wait=wait,
            timeout=timeout,
        )

    def close_gripper(
        self,
        speed=1.0,
        force=0.6,
        wait=True,
        timeout=DEFAULT_TIMEOUT,
    ):
        return self.move_gripper(
            position=1.0,
            speed=speed,
            force=force,
            wait=wait,
            timeout=timeout,
        )

    def stop_gripper(self):
        body = """
rq_stop()
""" + self._status_write_body()

        self._execute_script(
            "robotiq_stop",
            body,
            restore_freedrive=True,
        )

        try:
            status = self._status_from_registers(
                refresh=False
            )
            self._cache_status(
                status
            )
        except Exception:
            self._status_cache = None
            self._status_cache_time = None

        return True

    # ========================================================
    # Lifecycle
    # ========================================================

    def shutdown(self):
        self._disconnect_receive()
        self._arm_driver = None
        self._resolved_arm_name = None
        return True