from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import requests

from .models import InferenceAction


class HTTPArmExecutor:
    """Safely converts one relative policy action into one blocking moveL."""

    def __init__(
        self,
        api_url: str,
        arm_name: str,
        timeout: float,
        action_frame: str,
        max_translation_delta: float,
        max_rotation_delta: float,
        speed: float,
        acceleration: float,
    ):
        self._api_url = api_url.rstrip("/")
        self._arm_name = arm_name
        self._timeout = timeout
        self._action_frame = action_frame
        self._max_translation_delta = max_translation_delta
        self._max_rotation_delta = max_rotation_delta
        self._speed = speed
        self._acceleration = acceleration
        self._session = requests.Session()

    def _get(self, path: str) -> dict[str, Any]:
        response = self._session.get(
            f"{self._api_url}{path}",
            params={"arm_name": self._arm_name},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("result"):
            raise RuntimeError(f"arm API request failed: {payload}")
        return payload

    @staticmethod
    def _single_arm(payload: dict[str, Any]) -> dict[str, Any]:
        arms = payload.get("data", {}).get("arms", [])
        if not isinstance(arms, list) or len(arms) != 1:
            raise RuntimeError("arm API did not return exactly one connected arm")
        return arms[0]

    def _current_pose(self) -> list[float]:
        status_entry = self._single_arm(self._get("/api/arm/get_arm_status"))
        status = status_entry.get("status", {})
        if status.get("is_emergency_stopped"):
            raise RuntimeError("refusing inference motion: arm is emergency-stopped")
        if status.get("is_protective_stopped"):
            raise RuntimeError("refusing inference motion: arm is protective-stopped")
        if not status.get("data_valid", False) or status.get("data_stale", True):
            raise RuntimeError("refusing inference motion: arm state is invalid or stale")
        pose = status.get("pose")
        if not isinstance(pose, list) or len(pose) != 6:
            raise RuntimeError("arm status did not contain a valid TCP pose")
        return [float(value) for value in pose]

    def _target_pose(self, current_pose: list[float], action: InferenceAction) -> list[float]:
        translation = np.asarray(action.relative_position, dtype=np.float64)
        rotation_delta = np.asarray(action.relative_rotation, dtype=np.float64)
        translation_norm = float(np.linalg.norm(translation))
        rotation_norm = float(np.linalg.norm(rotation_delta))
        if not math.isfinite(translation_norm) or translation_norm > self._max_translation_delta:
            raise RuntimeError(
                f"refusing inference motion: translation delta {translation_norm:.6f} m "
                f"exceeds {self._max_translation_delta:.6f} m"
            )
        if not math.isfinite(rotation_norm) or rotation_norm > self._max_rotation_delta:
            raise RuntimeError(
                f"refusing inference motion: rotation delta {rotation_norm:.6f} rad "
                f"exceeds {self._max_rotation_delta:.6f} rad"
            )

        current = np.asarray(current_pose, dtype=np.float64)
        current_rotation, _ = cv2.Rodrigues(current[3:])
        delta_rotation, _ = cv2.Rodrigues(rotation_delta)
        if self._action_frame == "tool":
            target_position = current[:3] + current_rotation @ translation
            target_rotation = current_rotation @ delta_rotation
        else:
            target_position = current[:3] + translation
            target_rotation = delta_rotation @ current_rotation
        target_rotvec, _ = cv2.Rodrigues(target_rotation)
        return [*target_position.tolist(), *target_rotvec.reshape(3).tolist()]

    def __call__(self, action: InferenceAction) -> None:
        target_pose = self._target_pose(self._current_pose(), action)
        command = {
            "arm_name": self._arm_name,
            "pose": target_pose,
            "speed": self._speed,
            "acceleration": self._acceleration,
            "wait": True,
        }
        response = self._session.post(
            f"{self._api_url}/api/arm/move_arm_pose",
            json=command,
            timeout=max(self._timeout, 30.0),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("result"):
            raise RuntimeError(f"arm rejected inference motion: {payload}")

    def close(self) -> None:
        self._session.close()
