import math
import socket
import threading
import time


DEFAULT_PORT = 63352
DEFAULT_TIMEOUT = 1.0

MIN_POSITION = 0
MAX_POSITION = 255
MIN_SPEED = 0
MAX_SPEED = 255
MIN_FORCE = 0
MAX_FORCE = 255



class RobotiqDriver:

    DRIVER_METADATA = {
        "name": "robotiq",
        "manufacturer": "Robotiq",
        "model": "2F Gripper",
        "interface": "urcap_socket",
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
        port=DEFAULT_PORT,
        timeout=DEFAULT_TIMEOUT,
        auto_activate=True,
    ):
        self._host = str(host)
        self._port = int(port)
        self._timeout = float(timeout)
        self._auto_activate = bool(auto_activate)

        self._lock = threading.RLock()
        self._activated = False

        if self._auto_activate:
            try:
                self.activate_gripper()
            except Exception:
                self._activated = False

    @staticmethod
    def _normalize_int(
        value,
        minimum,
        maximum,
        name,
    ):
        try:
            value = int(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} must be an integer"
            ) from exc

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between "
                f"{minimum} and {maximum}"
            )

        return value

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
    def _ratio_to_raw(value):
        return int(round(float(value) * MAX_POSITION))

    @staticmethod
    def _raw_to_ratio(value):
        return float(value) / float(MAX_POSITION)

    def _communicate(
        self,
        command,
    ):
        payload = (
            command.strip()
            + "\n"
        ).encode("ascii")

        with self._lock:
            try:
                with socket.create_connection(
                    (
                        self._host,
                        self._port,
                    ),
                    timeout=self._timeout,
                ) as sock:

                    sock.settimeout(
                        self._timeout
                    )

                    sock.sendall(
                        payload
                    )

                    data = b""

                    while True:

                        chunk = sock.recv(
                            1024
                        )

                        if not chunk:
                            break

                        data += chunk

                        # GET response:
                        #   b"STA 3\n"
                        if data.endswith(
                            b"\n"
                        ):
                            break

                        # SET response:
                        #   b"ack"
                        if (
                            data.strip()
                            .lower()
                            == b"ack"
                        ):
                            break

            except OSError as exc:
                raise ConnectionError(
                    f"Failed to communicate "
                    f"with gripper at "
                    f"{self._host}:"
                    f"{self._port}: "
                    f"{exc}"
                ) from exc

        response = (
            data
            .decode(
                "ascii",
                errors="replace",
            )
            .strip()
        )

        if not response:
            raise RuntimeError(
                "Empty response from gripper"
            )

        return response

    def _get_variable(
        self,
        variable,
    ):
        variable = (
            str(variable)
            .strip()
            .upper()
        )

        response = (
            self._communicate(
                f"GET {variable}"
            )
        )

        parts = response.split()

        if len(parts) < 2:
            raise RuntimeError(
                f"Invalid response: "
                f"{response}"
            )

        if parts[0].upper() != variable:
            raise RuntimeError(
                f"Unexpected response: "
                f"{response}"
            )

        try:
            return int(parts[1])
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid integer response: "
                f"{response}"
            ) from exc

    def _set_variables(
        self,
        **variables,
    ):
        tokens = ["SET"]

        for key, value in (
            variables.items()
        ):
            tokens.extend([
                str(key).upper(),
                str(int(value)),
            ])

        response = (
            self._communicate(
                " ".join(tokens)
            )
        )

        if response.upper() != "ACK":
            raise RuntimeError(
                f"Gripper command failed: "
                f"{response}"
            )

        return True

    def _check_connection(
        self,
    ):
        try:
            self._get_variable(
                "STA"
            )
            return True
        except Exception:
            return False

    def activate_gripper(
        self,
        timeout=5.0,
    ):
        self._set_variables(
            ACT=1,
            GTO=1,
        )

        deadline = (
            time.monotonic()
            + float(timeout)
        )

        while time.monotonic() < deadline:
            status = (
                self._get_variable(
                    "STA"
                )
            )

            if status == 3:
                self._activated = True
                return True

            time.sleep(
                0.05
            )

        self._activated = False

        raise TimeoutError(
            "Gripper activation timeout"
        )

    @staticmethod
    def _decode_object_status(
        object_status,
    ):
        return {
            0: "moving",
            1: "object_detected_opening",
            2: "object_detected_closing",
            3: "position_reached",
        }.get(
            object_status,
            "unknown",
        )

    def get_gripper_status(
        self,
    ):
        connected = self._check_connection()

        if not connected:
            self._activated = False
            return {
                "connected": False,
                "ready": False,
                "moving": None,
                "position": None,
                "object_detected": None,
                "fault": None,
                "activated": False,
                "requested_position": None,
                "speed": None,
                "force": None,
                "object_status": None,
                "object_status_name": "unavailable",
                "fault_code": None,
                "gripper_status": None,
            }

        gripper_status = self._get_variable("STA")
        raw_position = self._get_variable("POS")
        raw_requested_position = self._get_variable("PRE")
        object_status = self._get_variable("OBJ")
        fault_code = self._get_variable("FLT")

        raw_speed = None
        raw_force = None

        try:
            raw_speed = self._get_variable("SPE")
        except Exception:
            pass

        try:
            raw_force = self._get_variable("FOR")
        except Exception:
            pass

        activated = gripper_status == 3
        moving = object_status == 0
        object_detected = object_status in (1, 2)
        fault = fault_code != 0
        ready = connected and activated and not fault

        self._activated = activated

        return {
            "connected": True,
            "ready": ready,
            "moving": moving,
            "position": self._raw_to_ratio(raw_position),
            "object_detected": object_detected,
            "fault": fault,
            "activated": activated,
            "requested_position": self._raw_to_ratio(
                raw_requested_position
            ),
            "speed": (
                self._raw_to_ratio(raw_speed)
                if raw_speed is not None
                else None
            ),
            "force": (
                self._raw_to_ratio(raw_force)
                if raw_force is not None
                else None
            ),
            "object_status": object_status,
            "object_status_name": self._decode_object_status(
                object_status
            ),
            "fault_code": fault_code,
            "gripper_status": gripper_status,
        }

    def move_gripper(
        self,
        position,
        speed=1.0,
        force=0.6,
        wait=True,
        timeout=5.0,
    ):
        position = self._normalize_ratio(position, "position")
        speed = self._normalize_ratio(speed, "speed")
        force = self._normalize_ratio(force, "force")

        if not isinstance(wait, bool):
            raise ValueError("wait must be bool")

        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "timeout must be a number"
            ) from exc

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0"
            )

        raw_position = self._ratio_to_raw(position)
        raw_speed = self._ratio_to_raw(speed)
        raw_force = self._ratio_to_raw(force)

        gripper_status = self._get_variable("STA")

        if gripper_status != 3:
            self.activate_gripper()
        else:
            self._activated = True

        self._set_variables(
            POS=raw_position,
            SPE=raw_speed,
            FOR=raw_force,
            GTO=1,
        )

        if not wait:
            return True

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            object_status = self._get_variable("OBJ")

            if object_status in (1, 2, 3):
                return True

            time.sleep(0.03)

        raise TimeoutError(
            "Gripper move timeout"
        )

    def open_gripper(
        self,
        speed=1.0,
        force=0.6,
        wait=True,
        timeout=5.0,
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
        timeout=5.0,
    ):
        return self.move_gripper(
            position=1.0,
            speed=speed,
            force=force,
            wait=wait,
            timeout=timeout,
        )

    def stop_gripper(
        self,
    ):
        return self._set_variables(
            GTO=0
        )