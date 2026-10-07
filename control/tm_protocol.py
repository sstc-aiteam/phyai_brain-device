from __future__ import annotations

"""
Pure-Python Techman Robot TMSCT / TMSVR protocol client.

No ROS dependency.
No techmanpy dependency.
Python standard library only.

Wire framing:
    $HEADER,LENGTH,DATA,*CHECKSUM\r\n

Ports:
    TMSCT / Listen Node      : TCP 5890
    TMSVR / Ethernet Slave   : TCP 5891

Robot Brain integration:
    This module is a low-level protocol helper.  It is NOT an ARM driver by
    itself.  control.tm12.tm12Driver should continue satisfying the loader
    contract and import TMProtocol internally.
"""

import ast
import json
import math
import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable


TMSCT_PORT = 5890
TMSVR_PORT = 5891

DEFAULT_CONNECT_TIMEOUT = 2.0
DEFAULT_IO_TIMEOUT = 3.0
DEFAULT_MAX_PACKET_BYTES = 4 * 1024 * 1024

_HEADER_RE = re.compile(r"^[A-Z0-9]+$")
_TRANSACTION_ID_RE = re.compile(r"^[A-Za-z0-9]+$")


class TMProtocolError(RuntimeError):
    pass


class TMConnectionError(TMProtocolError):
    pass


class TMPacketError(TMProtocolError):
    pass


class TMControllerError(TMProtocolError):
    def __init__(
        self,
        message: str,
        *,
        header: str | None = None,
        data: bytes | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.header = header
        self.data = data
        self.code = code


class TMTimeoutError(TMProtocolError):
    pass


@dataclass(frozen=True)
class TMPacket:
    header: str
    data: bytes
    checksum: int
    raw: bytes

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


def _xor_checksum(payload: bytes) -> int:
    value = 0
    for byte in payload:
        value ^= byte
    return value


def encode_packet(
    header: str,
    data: str | bytes,
) -> bytes:
    if not isinstance(header, str) or not header:
        raise ValueError("header must be a non-empty string")

    header = header.strip().upper()

    if not _HEADER_RE.fullmatch(header):
        raise ValueError(
            "header must contain only uppercase letters and digits"
        )

    if isinstance(data, str):
        data_bytes = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        data_bytes = bytes(data)
    else:
        raise TypeError("data must be str or bytes-like")

    prefix = (
        b"$"
        + header.encode("ascii")
        + b","
        + str(len(data_bytes)).encode("ascii")
        + b","
        + data_bytes
        + b","
    )

    checksum = _xor_checksum(prefix[1:])

    return (
        prefix
        + b"*"
        + f"{checksum:02X}".encode("ascii")
        + b"\r\n"
    )


def decode_packet(raw: bytes) -> TMPacket:
    raw = bytes(raw)

    if not raw.startswith(b"$"):
        raise TMPacketError("packet does not start with '$'")

    if not raw.endswith(b"\r\n"):
        raise TMPacketError("packet does not end with CRLF")

    star_index = raw.rfind(b"*")

    if star_index < 0:
        raise TMPacketError("packet checksum marker '*' is missing")

    checksum_text = raw[star_index + 1:star_index + 3]

    try:
        expected_checksum = int(checksum_text.decode("ascii"), 16)
    except Exception as exc:
        raise TMPacketError(
            f"invalid checksum field: {checksum_text!r}"
        ) from exc

    checksum_payload = raw[1:star_index]
    actual_checksum = _xor_checksum(checksum_payload)

    if actual_checksum != expected_checksum:
        raise TMPacketError(
            "checksum mismatch: "
            f"expected 0x{expected_checksum:02X}, "
            f"calculated 0x{actual_checksum:02X}"
        )

    body = raw[1:star_index]

    first_comma = body.find(b",")
    second_comma = body.find(b",", first_comma + 1)

    if first_comma < 0 or second_comma < 0:
        raise TMPacketError("invalid TM packet header")

    header = body[:first_comma].decode("ascii")

    try:
        declared_length = int(
            body[first_comma + 1:second_comma].decode("ascii")
        )
    except Exception as exc:
        raise TMPacketError("invalid packet length") from exc

    data_with_trailing_comma = body[second_comma + 1:]

    if not data_with_trailing_comma.endswith(b","):
        raise TMPacketError(
            "packet data is missing trailing protocol comma"
        )

    data = data_with_trailing_comma[:-1]

    if len(data) != declared_length:
        raise TMPacketError(
            "packet length mismatch: "
            f"declared={declared_length}, actual={len(data)}"
        )

    return TMPacket(
        header=header,
        data=data,
        checksum=expected_checksum,
        raw=raw,
    )


class TMPacketStream:
    def __init__(
        self,
        max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
    ):
        self._buffer = bytearray()
        self.max_packet_bytes = int(max_packet_bytes)

    def feed(self, data: bytes) -> list[TMPacket]:
        if not data:
            return []

        self._buffer.extend(data)

        if len(self._buffer) > self.max_packet_bytes:
            raise TMPacketError(
                "receive buffer exceeded maximum packet size"
            )

        packets = []

        while True:
            start = self._buffer.find(b"$")

            if start < 0:
                self._buffer.clear()
                break

            if start > 0:
                del self._buffer[:start]

            end = self._buffer.find(b"\r\n")

            if end < 0:
                break

            raw = bytes(self._buffer[:end + 2])
            del self._buffer[:end + 2]

            packets.append(decode_packet(raw))

        return packets


class _TMChannel:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        io_timeout: float = DEFAULT_IO_TIMEOUT,
        max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
    ):
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string")

        self.host = host.strip()
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self.io_timeout = float(io_timeout)
        self.max_packet_bytes = int(max_packet_bytes)

        self._sock = None
        self._stream = TMPacketStream(self.max_packet_bytes)
        self._rx_queue = []
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> bool:
        with self._lock:
            if self._sock is not None:
                return True

            try:
                sock = socket.create_connection(
                    (self.host, self.port),
                    timeout=self.connect_timeout,
                )
                sock.settimeout(self.io_timeout)
            except OSError as exc:
                raise TMConnectionError(
                    f"failed to connect {self.host}:{self.port}: {exc}"
                ) from exc

            self._sock = sock
            self._stream = TMPacketStream(self.max_packet_bytes)
            self._rx_queue.clear()

            return True

    def close(self) -> None:
        with self._lock:
            sock = self._sock
            self._sock = None

            if sock is None:
                return

            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                sock.close()
            except OSError:
                pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise TMConnectionError("channel is not connected")
        return self._sock

    def send_packet(
        self,
        header: str,
        data: str | bytes,
    ) -> bytes:
        raw = encode_packet(header, data)
        sock = self._require_socket()

        try:
            sock.sendall(raw)
        except OSError as exc:
            self.close()
            raise TMConnectionError(f"send failed: {exc}") from exc

        return raw

    def _recv_more(self, timeout: float) -> None:
        sock = self._require_socket()
        sock.settimeout(timeout)

        try:
            chunk = sock.recv(65536)
        except socket.timeout as exc:
            raise TMTimeoutError(
                "timed out waiting for TM response"
            ) from exc
        except OSError as exc:
            self.close()
            raise TMConnectionError(f"receive failed: {exc}") from exc

        if not chunk:
            self.close()
            raise TMConnectionError(
                "TM controller closed the connection"
            )

        self._rx_queue.extend(self._stream.feed(chunk))

    def receive_packet(
        self,
        *,
        timeout: float | None = None,
    ) -> TMPacket:
        timeout = self.io_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + timeout

        while True:
            if self._rx_queue:
                return self._rx_queue.pop(0)

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                raise TMTimeoutError(
                    "timed out waiting for TM packet"
                )

            self._recv_more(remaining)

    def wait_for(
        self,
        predicate,
        *,
        timeout: float | None = None,
    ) -> TMPacket:
        timeout = self.io_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + timeout
        deferred = []

        try:
            while True:
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise TMTimeoutError(
                        "timed out waiting for matching TM packet"
                    )

                packet = self.receive_packet(timeout=remaining)

                if predicate(packet):
                    return packet

                deferred.append(packet)
        finally:
            if deferred:
                self._rx_queue = deferred + self._rx_queue


@dataclass(frozen=True)
class TMSCTResponse:
    transaction_id: str
    result: str
    ok: bool
    packet: TMPacket


def _validate_transaction_id(transaction_id: str) -> str:
    if not isinstance(transaction_id, str):
        raise TypeError("transaction_id must be str")

    transaction_id = transaction_id.strip()

    if not transaction_id:
        raise ValueError("transaction_id must not be empty")

    if not _TRANSACTION_ID_RE.fullmatch(transaction_id):
        raise ValueError(
            "transaction_id must contain only A-Z, a-z and 0-9"
        )

    return transaction_id


def _format_tm_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if value is None:
        return "none"

    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("TM numeric values must be finite")
        return str(value)

    if isinstance(value, (list, tuple)):
        return (
            "{"
            + ",".join(_format_tm_value(item) for item in value)
            + "}"
        )

    raise TypeError(
        f"unsupported TM value type: {type(value).__name__}"
    )


def build_tm_function(name: str, *args: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("function name must be non-empty")

    name = name.strip()

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid TM function name: {name!r}")

    return (
        name
        + "("
        + ",".join(_format_tm_value(arg) for arg in args)
        + ")"
    )


class TMSCTClient(_TMChannel):
    def __init__(
        self,
        host: str,
        *,
        port: int = TMSCT_PORT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        io_timeout: float = DEFAULT_IO_TIMEOUT,
    ):
        super().__init__(
            host,
            port,
            connect_timeout=connect_timeout,
            io_timeout=io_timeout,
        )

    @staticmethod
    def _parse_response(packet: TMPacket) -> TMSCTResponse:
        if packet.header == "CPERR":
            raise TMControllerError(
                f"TM protocol error: {packet.text}",
                header=packet.header,
                data=packet.data,
                code=packet.text,
            )

        if packet.header != "TMSCT":
            raise TMPacketError(
                f"expected TMSCT, got {packet.header}"
            )

        try:
            transaction_id, result = packet.text.split(",", 1)
        except ValueError as exc:
            raise TMPacketError(
                f"invalid TMSCT response: {packet.text!r}"
            ) from exc

        result_text = result.strip()
        ok = (
            result_text.upper() == "OK"
            or result_text.upper().startswith("OK;")
        )

        return TMSCTResponse(
            transaction_id=transaction_id,
            result=result,
            ok=ok,
            packet=packet,
        )

    def send_script(
        self,
        script: str,
        *,
        transaction_id: str = "SCT1",
        timeout: float | None = None,
        require_ok: bool = True,
    ) -> TMSCTResponse:
        transaction_id = _validate_transaction_id(transaction_id)

        if not isinstance(script, str) or not script.strip():
            raise ValueError("script must be a non-empty string")

        with self._lock:
            self.connect()

            self.send_packet(
                "TMSCT",
                transaction_id + "," + script,
            )

            def matches(packet: TMPacket) -> bool:
                if packet.header == "CPERR":
                    return True

                return (
                    packet.header == "TMSCT"
                    and packet.data.startswith(
                        (transaction_id + ",").encode("utf-8")
                    )
                )

            packet = self.wait_for(matches, timeout=timeout)
            response = self._parse_response(packet)

            if require_ok and not response.ok:
                raise TMControllerError(
                    "TMscript rejected by controller: "
                    f"id={response.transaction_id!r}, "
                    f"result={response.result!r}",
                    header="TMSCT",
                    data=packet.data,
                )

            return response

    def call(
        self,
        function_name: str,
        *args: Any,
        transaction_id: str = "SCT1",
        timeout: float | None = None,
        require_ok: bool = True,
    ) -> TMSCTResponse:
        return self.send_script(
            build_tm_function(function_name, *args),
            transaction_id=transaction_id,
            timeout=timeout,
            require_ok=require_ok,
        )


@dataclass(frozen=True)
class TMSVRResponse:
    transaction_id: str
    mode: int
    content: bytes
    packet: TMPacket

    @property
    def content_text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


def _split_tmsvr_data(data: bytes) -> tuple[str, int, bytes]:
    first = data.find(b",")
    second = data.find(b",", first + 1)

    if first <= 0 or second < 0:
        raise TMPacketError(f"invalid TMSVR data: {data!r}")

    try:
        transaction_id = data[:first].decode("ascii")
        mode = int(data[first + 1:second].decode("ascii"))
    except Exception as exc:
        raise TMPacketError(
            f"invalid TMSVR response: {data!r}"
        ) from exc

    return transaction_id, mode, data[second + 1:]


def _parse_tm_scalar(text: str) -> Any:
    text = text.strip()

    if not text:
        return ""

    lowered = text.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    candidate = text

    if candidate.startswith("{") and candidate.endswith("}"):
        candidate = "[" + candidate[1:-1] + "]"

    try:
        return ast.literal_eval(candidate)
    except Exception:
        pass

    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
    except Exception:
        pass

    try:
        return float(text)
    except Exception:
        return text


def parse_tmsvr_string_items(content: str) -> dict[str, Any]:
    result = {}

    for line in content.splitlines():
        line = line.strip()

        if not line:
            continue

        if "=" not in line:
            result[line] = None
            continue

        name, value = line.split("=", 1)
        result[name.strip()] = _parse_tm_scalar(value)

    return result


class TMSVRClient(_TMChannel):
    def __init__(
        self,
        host: str,
        *,
        port: int = TMSVR_PORT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        io_timeout: float = DEFAULT_IO_TIMEOUT,
    ):
        super().__init__(
            host,
            port,
            connect_timeout=connect_timeout,
            io_timeout=io_timeout,
        )

    @staticmethod
    def _parse_response(packet: TMPacket) -> TMSVRResponse:
        if packet.header == "CPERR":
            raise TMControllerError(
                f"TM protocol error: {packet.text}",
                header=packet.header,
                data=packet.data,
                code=packet.text,
            )

        if packet.header != "TMSVR":
            raise TMPacketError(
                f"expected TMSVR, got {packet.header}"
            )

        transaction_id, mode, content = _split_tmsvr_data(
            packet.data
        )

        return TMSVRResponse(
            transaction_id=transaction_id,
            mode=mode,
            content=content,
            packet=packet,
        )

    def _request(
        self,
        transaction_id: str,
        mode: int,
        content: str | bytes,
        *,
        timeout: float | None = None,
    ) -> TMSVRResponse:
        transaction_id = _validate_transaction_id(transaction_id)

        content_bytes = (
            content.encode("utf-8")
            if isinstance(content, str)
            else bytes(content)
        )

        with self._lock:
            self.connect()

            data = (
                transaction_id.encode("ascii")
                + b","
                + str(int(mode)).encode("ascii")
                + b","
                + content_bytes
            )

            self.send_packet("TMSVR", data)

            def matches(packet: TMPacket) -> bool:
                if packet.header == "CPERR":
                    return True

                return (
                    packet.header == "TMSVR"
                    and packet.data.startswith(
                        (transaction_id + ",").encode("ascii")
                    )
                )

            packet = self.wait_for(matches, timeout=timeout)
            return self._parse_response(packet)

    def read_items(
        self,
        items: Iterable[str],
        *,
        transaction_id: str = "Q1",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        normalized = []

        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "TMSVR item names must be non-empty strings"
                )
            normalized.append(item.strip())

        if not normalized:
            raise ValueError("items must not be empty")

        response = self._request(
            transaction_id,
            12,
            "\r\n".join(normalized),
            timeout=timeout,
        )

        if response.mode == 0:
            text = response.content_text

            if not text.startswith("00,"):
                raise TMControllerError(
                    f"TMSVR read rejected: {text}",
                    header="TMSVR",
                    data=response.packet.data,
                    code=text.split(",", 1)[0] if text else None,
                )

            return {}

        if response.mode != 12:
            raise TMPacketError(
                f"unexpected TMSVR mode for request-read: "
                f"{response.mode}"
            )

        return parse_tmsvr_string_items(response.content_text)

    def get_value(
        self,
        item: str,
        *,
        transaction_id: str = "Q1",
        timeout: float | None = None,
    ) -> Any:
        values = self.read_items(
            [item],
            transaction_id=transaction_id,
            timeout=timeout,
        )

        if item in values:
            return values[item]

        if len(values) == 1:
            return next(iter(values.values()))

        raise TMProtocolError(
            f"TMSVR response did not contain item {item!r}: {values!r}"
        )

    def write_items(
        self,
        values: dict[str, Any],
        *,
        transaction_id: str = "W1",
        timeout: float | None = None,
    ) -> bool:
        if not isinstance(values, dict) or not values:
            raise ValueError("values must be a non-empty dict")

        lines = []

        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "TMSVR item names must be non-empty strings"
                )

            lines.append(
                name.strip()
                + "="
                + _format_tm_value(value)
            )

        response = self._request(
            transaction_id,
            2,
            "\r\n".join(lines),
            timeout=timeout,
        )

        if response.mode != 0:
            raise TMPacketError(
                f"expected TMSVR mode 0 write response, "
                f"got {response.mode}"
            )

        text = response.content_text

        if not text.startswith("00,"):
            raise TMControllerError(
                f"TMSVR write rejected: {text}",
                header="TMSVR",
                data=response.packet.data,
                code=text.split(",", 1)[0] if text else None,
            )

        return True

    def set_value(
        self,
        item: str,
        value: Any,
        *,
        transaction_id: str = "W1",
        timeout: float | None = None,
    ) -> bool:
        return self.write_items(
            {item: value},
            transaction_id=transaction_id,
            timeout=timeout,
        )


class TMProtocol:
    """
    Unified façade for control/tm12.py.

    This class deliberately stays below the Robot Brain ARM driver contract.
    `tm12Driver` remains the object loaded by loader.py.
    """

    def __init__(
        self,
        ip: str,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        io_timeout: float = DEFAULT_IO_TIMEOUT,
    ):
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError("ip must be a non-empty string")

        self.ip = ip.strip()
        self.connect_timeout = float(connect_timeout)
        self.io_timeout = float(io_timeout)

        self.sct = TMSCTClient(
            self.ip,
            connect_timeout=self.connect_timeout,
            io_timeout=self.io_timeout,
        )

        self.svr = TMSVRClient(
            self.ip,
            connect_timeout=self.connect_timeout,
            io_timeout=self.io_timeout,
        )

        self._id_lock = threading.Lock()
        self._id_counter = 0

    def close(self) -> None:
        self.sct.close()
        self.svr.close()

    def _next_id(self, prefix: str) -> str:
        clean_prefix = re.sub(r"[^A-Za-z0-9]", "", prefix) or "TM"

        with self._id_lock:
            self._id_counter = (self._id_counter + 1) % 1_000_000
            return f"{clean_prefix}{self._id_counter}"

    def send_script(
        self,
        script: str,
        *,
        transaction_id: str | None = None,
        timeout: float | None = None,
        require_ok: bool = True,
    ) -> TMSCTResponse:
        return self.sct.send_script(
            script,
            transaction_id=transaction_id or self._next_id("S"),
            timeout=timeout,
            require_ok=require_ok,
        )

    def call(
        self,
        function_name: str,
        *args: Any,
        transaction_id: str | None = None,
        timeout: float | None = None,
        require_ok: bool = True,
    ) -> TMSCTResponse:
        return self.sct.call(
            function_name,
            *args,
            transaction_id=transaction_id or self._next_id("S"),
            timeout=timeout,
            require_ok=require_ok,
        )

    def get_value(self, item: str) -> Any:
        return self.svr.get_value(
            item,
            transaction_id=self._next_id("Q"),
        )

    def get_values(self, items: Iterable[str]) -> dict[str, Any]:
        return self.svr.read_items(
            items,
            transaction_id=self._next_id("Q"),
        )

    def set_value(self, item: str, value: Any) -> bool:
        return self.svr.set_value(
            item,
            value,
            transaction_id=self._next_id("W"),
        )

    def set_values(self, values: dict[str, Any]) -> bool:
        return self.svr.write_items(
            values,
            transaction_id=self._next_id("W"),
        )

    def stop_motion(
        self,
        level: int | None = None,
    ) -> TMSCTResponse:
        if level is None:
            return self.call("StopAndClearBuffer")

        return self.call(
            "StopAndClearBuffer",
            int(level),
        )

    def pause_project(self) -> TMSCTResponse:
        return self.call("Pause")

    def resume_project(self) -> TMSCTResponse:
        return self.call("Resume")

    def health_check(self) -> dict[str, Any]:
        result = {
            "svr": False,
            "sct": False,
            "robot_model": None,
            "svr_error": None,
            "sct_error": None,
        }

        try:
            result["robot_model"] = self.get_value("Robot_Model")
            result["svr"] = True
        except Exception as exc:
            result["svr_error"] = str(exc)

        try:
            self.sct.connect()
            result["sct"] = True
        except Exception as exc:
            result["sct_error"] = str(exc)

        return result


def _self_test() -> None:
    assert encode_packet(
        "TMSTA",
        "00",
    ) == b"$TMSTA,2,00,*41\r\n"

    assert encode_packet(
        "TMSCT",
        "1,var_i++",
    ) == b"$TMSCT,9,1,var_i++,*06\r\n"

    packet = encode_packet(
        "TMSCT",
        "1,var_i++",
    )

    decoded = decode_packet(packet)

    assert decoded.header == "TMSCT"
    assert decoded.data == b"1,var_i++"

    assert build_tm_function(
        "PTP",
        "CPP",
        [1, 2, 3, 4, 5, 6],
        10,
        200,
        0,
        False,
    ) == 'PTP("CPP",{1,2,3,4,5,6},10,200,0,false)'


if __name__ == "__main__":
    _self_test()
    print("tm_protocol self-test: OK")
