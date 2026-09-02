from __future__ import annotations

import itertools
from typing import Any

import cv2
import numpy as np
import zmq

from .models import InferenceAction, RobotState
from .serialization import pack_message, unpack_message


class InferenceClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        gripper_position: float = 0.0,
        instruction: str = "open the left drawer",
    ):
        self._host = host
        self._port = port
        self._timeout_ms = max(1, int(timeout * 1000))
        self._gripper_position = float(gripper_position)
        self._instruction = instruction
        self._sequence = itertools.count(1)
        self._context = zmq.Context()
        self._socket = None
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def infer(self, jpeg: bytes, state: RobotState) -> InferenceAction:
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode camera JPEG")
        if bgr.shape != (480, 640, 3):
            raise ValueError(f"camera image must have shape (480, 640, 3), got {bgr.shape}")

        request = {
            "endpoint": "get_action",
            "data": {
                # Keep the transport language-neutral. The server converts these
                # native values into NumPy after decoding the JPEG.
                "tcp_pose": [*state.eef_position, *state.eef_rotation],
                "gripper_position": self._gripper_position,
                "image": jpeg,
                "instruction": self._instruction,
                "sequence_id": next(self._sequence),
            },
        }
        try:
            self._socket.send(pack_message(request))
            payload: Any = unpack_message(self._socket.recv())
        except zmq.Again:
            # REQ sockets cannot send again after a timed-out request.
            self._connect()
            raise TimeoutError("GR00T inference request timed out")
        if not isinstance(payload, dict):
            raise ValueError("GR00T inference response must be an object")
        if "error" in payload:
            raise RuntimeError(f"GR00T server error: {payload['error']}")
        return InferenceAction.from_response(payload)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._context.term()
