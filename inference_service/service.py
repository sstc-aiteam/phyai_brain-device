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
        self._camera_was_running: bool | None = None
        self._camera_retry_period = max(1.0, self._period)
        self._last_error: tuple[type[Exception], str] | None = None

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
                if not self._image_source.is_camera_running():
                    if self._camera_was_running is not False:
                        logger.warning("camera is not running; please start the camera")
                    self._camera_was_running = False
                    remaining = self._camera_retry_period - (time.monotonic() - started)
                    self._stop.wait(max(0.0, remaining))
                    continue

                if self._camera_was_running is False:
                    logger.info("camera is running; inference resumed")
                self._camera_was_running = True
                self.run_once()
                if self._last_error is not None:
                    logger.info("live inference recovered")
                    self._last_error = None
            except Exception as exc:
                error = (type(exc), str(exc))
                if error != self._last_error:
                    logger.exception("live inference cycle failed; suppressing repeats")
                    self._last_error = error
            remaining = self._period - (time.monotonic() - started)
            self._stop.wait(max(0.0, remaining))

    def close(self) -> None:
        self._client.close()
        self._image_source.close()
        self._pose_source.close()

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
