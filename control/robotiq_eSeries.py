import socket
import struct
import threading

from control.robotiq_preamble import (
    ROBOTIQ_PREAMBLE,
)


MIN_POSITION = 0
MAX_POSITION = 255

MIN_SPEED = 0
MAX_SPEED = 255

MIN_FORCE = 0
MAX_FORCE = 255

DEFAULT_SCRIPT_PORT = 30002
DEFAULT_TIMEOUT = 5.0


class RobotiqESeriesDriver:

    DRIVER_METADATA = {
        "name": "robotiq_eseries",
        "manufacturer": "Robotiq",
        "model": "2F Gripper",
        "robot_family": "Universal Robots e-Series",
        "interface": "urscript_tcp_30002_callback",
        "capabilities": {
            "status": True,
            "position": True,
            "speed": True,
            "force": True,
            "object_detection": True,
            "fault_code": True,
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
        callback_host=None,
        script_port=DEFAULT_SCRIPT_PORT,
    ):
        _ = port

        self._host = str(host)
        self._timeout = float(timeout)
        self._auto_activate = bool(auto_activate)

        self._callback_host = (
            str(callback_host)
            if callback_host
            else None
        )

        self._script_port = int(
            script_port
        )

        self._lock = (
            threading.RLock()
        )

        if self._auto_activate:
            try:
                self.activate_gripper()

            except Exception:
                pass


    # ========================================================
    # Validation
    # ========================================================

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


    # ========================================================
    # Connection
    # ========================================================

    def _check_connection(
        self,
    ):
        try:
            with socket.create_connection(
                (
                    self._host,
                    self._script_port,
                ),
                timeout=self._timeout,
            ):
                return True

        except OSError:
            return False


    def _is_connected(
        self,
    ):
        return self._check_connection()


    def reconnect_gripper(
        self,
    ):
        return self._check_connection()


    # ========================================================
    # URScript
    # ========================================================

    @staticmethod
    def _indent_script(
        text,
    ):
        return "\n".join(
            (
                "  " + line
                if line.strip()
                else ""
            )
            for line
            in text.splitlines()
        )


    def _build_script(
        self,
        function_name,
        body,
    ):
        script_body = (
            ROBOTIQ_PREAMBLE.rstrip()
            + "\n"
            + body.strip()
        )

        return (
            f"def {function_name}():\n"
            f"{self._indent_script(script_body)}\n"
            "end\n"
        )


    def _send_script(
        self,
        function_name,
        body,
    ):
        script = self._build_script(
            function_name,
            body,
        )

        payload = script.encode(
            "utf-8"
        )

        with self._lock:

            try:
                with socket.create_connection(
                    (
                        self._host,
                        self._script_port,
                    ),
                    timeout=self._timeout,
                ) as sock:

                    sock.settimeout(
                        0.25
                    )

                    sock.sendall(
                        payload
                    )

                    try:
                        sock.recv(
                            4096
                        )

                    except socket.timeout:
                        pass

            except OSError as exc:
                raise ConnectionError(
                    f"Failed to send URScript to "
                    f"{self._host}:"
                    f"{self._script_port}: "
                    f"{exc}"
                ) from exc

        return True


    # ========================================================
    # Callback status
    # ========================================================

    def _detect_callback_host(
        self,
    ):
        if self._callback_host:
            return self._callback_host

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        try:
            sock.connect(
                (
                    self._host,
                    self._script_port,
                )
            )

            local_ip = (
                sock.getsockname()[0]
            )

        finally:
            sock.close()

        if not local_ip:
            raise RuntimeError(
                "Unable to determine callback host IP"
            )

        return local_ip


    @staticmethod
    def _recv_exact(
        conn,
        size,
    ):
        data = b""

        while len(data) < size:

            chunk = conn.recv(
                size - len(data)
            )

            if not chunk:
                raise ConnectionError(
                    "Status callback closed before "
                    "all data arrived"
                )

            data += chunk

        return data


    def _read_raw_status(
        self,
    ):
        callback_host = (
            self._detect_callback_host()
        )

        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server.settimeout(
            self._timeout
        )

        payload = None

        try:
            server.bind(
                (
                    callback_host,
                    0,
                )
            )

            server.listen(
                1
            )

            callback_port = (
                server.getsockname()[1]
            )

            body = f"""
rq_sta = rq_get_var("STA")
rq_pos = rq_get_var("POS")
rq_pre = rq_get_var("PRE")
rq_spe = rq_get_var("SPE")
rq_for = rq_get_var("FOR")
rq_obj = rq_get_var("OBJ")
rq_flt = rq_get_var("FLT")

rq_callback_connected = socket_open(
    "{callback_host}",
    {callback_port},
    "rq_status_callback"
)

if rq_callback_connected:

    socket_send_int(
        rq_sta,
        "rq_status_callback"
    )

    socket_send_int(
        rq_pos,
        "rq_status_callback"
    )

    socket_send_int(
        rq_pre,
        "rq_status_callback"
    )

    socket_send_int(
        rq_spe,
        "rq_status_callback"
    )

    socket_send_int(
        rq_for,
        "rq_status_callback"
    )

    socket_send_int(
        rq_obj,
        "rq_status_callback"
    )

    socket_send_int(
        rq_flt,
        "rq_status_callback"
    )

    socket_close(
        "rq_status_callback"
    )

end
"""

            self._send_script(
                "robotiq_eseries_read_status",
                body,
            )

            conn, _ = (
                server.accept()
            )

            with conn:

                conn.settimeout(
                    self._timeout
                )

                payload = (
                    self._recv_exact(
                        conn,
                        28,
                    )
                )

        finally:
            server.close()

        if payload is None:
            raise RuntimeError(
                "Robotiq status callback returned no data"
            )

        (
            sta,
            pos,
            pre,
            spe,
            force,
            obj,
            flt,
        ) = struct.unpack(
            "!7i",
            payload,
        )

        return {
            "STA": int(sta),
            "POS": int(pos),
            "PRE": int(pre),
            "SPE": int(spe),
            "FOR": int(force),
            "OBJ": int(obj),
            "FLT": int(flt),
        }


    # ========================================================
    # Activation
    # ========================================================

    def activate_gripper(
        self,
        timeout=5.0,
    ):
        _ = timeout

        self._send_script(
            "robotiq_eseries_activate",
            """
rq_reset()
rq_activate_and_wait()
""",
        )

        return True


    # ========================================================
    # Status
    # ========================================================

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


    @staticmethod
    def _unavailable_status(
        connected,
        error=None,
    ):
        result = {
            "connected":
                bool(connected),

            "activated":
                False,

            "position":
                None,

            "requested_position":
                None,

            "speed":
                None,

            "force":
                None,

            "object_detected":
                False,

            "object_status":
                None,

            "object_status_name":
                "unavailable",

            "fault_code":
                None,

            "gripper_status":
                None,
        }

        if error is not None:
            result["error"] = str(
                error
            )

            result["error_type"] = (
                type(error).__name__
            )

        return result


    def get_gripper_status(
        self,
    ):
        if not self._is_connected():
            return self._unavailable_status(
                connected=False
            )

        try:
            raw = (
                self._read_raw_status()
            )

        except Exception as exc:
            return self._unavailable_status(
                connected=
                    self._is_connected(),

                error=
                    exc,
            )

        gripper_status = (
            raw["STA"]
        )

        object_status = (
            raw["OBJ"]
        )

        return {
            "connected":
                True,

            "activated":
                gripper_status == 3,

            "position":
                raw["POS"],

            "requested_position":
                raw["PRE"],

            "speed":
                raw["SPE"],

            "force":
                raw["FOR"],

            "object_detected":
                object_status
                in (
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
                raw["FLT"],

            "gripper_status":
                gripper_status,
        }


    # ========================================================
    # Motion
    # ========================================================

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

        _ = timeout

        move_command = (
            f"rq_move_and_wait({position})"
            if wait
            else f"rq_move({position})"
        )

        body = f"""
rq_set_speed({speed})
rq_set_force({force})
{move_command}
"""

        self._send_script(
            "robotiq_eseries_move",
            body,
        )

        return True


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
        self._send_script(
            "robotiq_eseries_stop",
            """
rq_stop()
""",
        )

        return True