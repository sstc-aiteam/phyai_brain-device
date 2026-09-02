from __future__ import annotations

import json
import logging
import signal
import threading
import time
from collections.abc import Callable

from .client import InferenceClient
from .models import InferenceAction, RobotState
from .sources import ImageSource, PoseSource

logger = logging.getLogger(__name__)


class LiveInferenceService:
    def __init__(
        self,
        pose_source: PoseSource,
        image_source: ImageSource,
        client: InferenceClient,
        hz: float,
        on_action: Callable[[InferenceAction], None] | None = None,
    ):
        self._pose_source = pose_source
        self._image_source = image_source
        self._client = client
        self._period = 1.0 / hz
        self._on_action = on_action or self._print_action
        self._stop = threading.Event()

    @staticmethod
    def _print_action(action: InferenceAction) -> None:
        print(json.dumps(action.as_dict(), ensure_ascii=False), flush=True)

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> InferenceAction:
        # Capture both observations in the same cycle, then infer.
        jpeg = self._image_source.read_jpeg()
        state = RobotState.from_tcp_pose(self._pose_source.read())
        action = self._client.infer(jpeg, state)
        self._on_action(action)
        return action

    def run_forever(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.run_once()
            except Exception:
                logger.exception("live inference cycle failed")
            remaining = self._period - (time.monotonic() - started)
            self._stop.wait(max(0.0, remaining))

    def close(self) -> None:
        self._client.close()
        self._image_source.close()
        self._pose_source.close()

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
