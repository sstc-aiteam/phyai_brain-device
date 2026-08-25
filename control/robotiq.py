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
        connected = (
            self._check_connection()
        )

        if not connected:
            return {
                "connected": False,
                "activated": False,
                "position": None,
                "requested_position": None,
                "speed": None,
                "force": None,
                "object_detected": False,
                "object_status": None,
                "object_status_name":
                    "unavailable",
                "fault_code": None,
                "gripper_status": None,
            }

        gripper_status = (
            self._get_variable(
                "STA"
            )
        )

        position = (
            self._get_variable(
                "POS"
            )
        )

        requested_position = (
            self._get_variable(
                "PRE"
            )
        )

        object_status = (
            self._get_variable(
                "OBJ"
            )
        )

        fault_code = (
            self._get_variable(
                "FLT"
            )
        )

        speed = None
        force = None

        try:
            speed = self._get_variable(
                "SPE"
            )
        except Exception:
            pass

        try:
            force = self._get_variable(
                "FOR"
            )
        except Exception:
            pass

        activated = (
            gripper_status == 3
        )

        self._activated = activated

        return {
            "connected": True,
            "activated": activated,
            "position": position,
            "requested_position":
                requested_position,
            "speed": speed,
            "force": force,
            "object_detected":
                object_status in (
                    1,
                    2,
                ),
            "object_status":
                object_status,
            "object_status_name":
                self._decode_object_status(
                    object_status
                ),
            "fault_code":
                fault_code,
            "gripper_status":
                gripper_status,
        }

    def move_gripper(
        self,
        position,
        speed=255,
        force=150,
        wait=True,
        timeout=5.0,
    ):
        position = (
            self._normalize_int(
                position,
                MIN_POSITION,
                MAX_POSITION,
                "position",
            )
        )

        speed = (
            self._normalize_int(
                speed,
                MIN_SPEED,
                MAX_SPEED,
                "speed",
            )
        )

        force = (
            self._normalize_int(
                force,
                MIN_FORCE,
                MAX_FORCE,
                "force",
            )
        )

        gripper_status = (
            self._get_variable(
                "STA"
            )
        )

        if gripper_status != 3:
            self.activate_gripper()

        else:
            self._activated = True

        self._set_variables(
            POS=position,
            SPE=speed,
            FOR=force,
            GTO=1,
        )

        if not wait:
            return True

        deadline = (
            time.monotonic()
            + float(timeout)
        )

        while time.monotonic() < deadline:
            object_status = (
                self._get_variable(
                    "OBJ"
                )
            )

            if object_status in (
                1,
                2,
                3,
            ):
                return True

            time.sleep(
                0.03
            )

        raise TimeoutError(
            "Gripper move timeout"
        )

    def open_gripper(
        self,
        speed=255,
        force=150,
        wait=True,
        timeout=5.0,
    ):
        return self.move_gripper(
            position=MIN_POSITION,
            speed=speed,
            force=force,
            wait=wait,
            timeout=timeout,
        )

    def close_gripper(
        self,
        speed=255,
        force=150,
        wait=True,
        timeout=5.0,
    ):
        return self.move_gripper(
            position=MAX_POSITION,
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