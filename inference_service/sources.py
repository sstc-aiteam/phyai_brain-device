from __future__ import annotations

from typing import Protocol, Sequence

import requests


class PoseSource(Protocol):
    def read(self) -> Sequence[float]: ...
    def close(self) -> None: ...


class ImageSource(Protocol):
    def read_jpeg(self) -> bytes: ...
    def close(self) -> None: ...


class URTCPPoseSource:
    """Owns the RTDE receive connection from this .99 host to the .76 robot."""

    def __init__(self, robot_ip: str):
        try:
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise RuntimeError("ur-rtde is required for live TCP pose capture") from exc
        self._receiver = RTDEReceiveInterface(robot_ip)

    def read(self) -> Sequence[float]:
        return self._receiver.getActualTCPPose()

    def close(self) -> None:
        disconnect = getattr(self._receiver, "disconnect", None)
        if callable(disconnect):
            disconnect()


class HTTPJPEGSource:
    """Reads the latest JPEG from the existing brain-device camera endpoint."""

    def __init__(self, url: str, timeout: float):
        self._url = url
        self._timeout = timeout
        self._session = requests.Session()

    def read_jpeg(self) -> bytes:
        response = self._session.get(self._url, timeout=self._timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "image/jpeg" not in content_type:
            raise ValueError(f"image source returned unsupported Content-Type: {content_type!r}")
        if not response.content:
            raise ValueError("image source returned an empty JPEG")
        return response.content

    def close(self) -> None:
        self._session.close()
