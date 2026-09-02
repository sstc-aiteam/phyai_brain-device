from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _vector3(name: str, value: Any) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite numbers")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class RobotState:
    eef_position: tuple[float, float, float]
    eef_rotation: tuple[float, float, float]

    @classmethod
    def from_tcp_pose(cls, pose: Sequence[float]) -> "RobotState":
        if len(pose) != 6:
            raise ValueError("getActualTCPPose() must return [x,y,z,rx,ry,rz]")
        return cls(
            eef_position=_vector3("state.eef_position", pose[:3]),
            eef_rotation=_vector3("state.eef_rotation", pose[3:]),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "eef_position": list(self.eef_position),
            "eef_rotation": list(self.eef_rotation),
        }


@dataclass(frozen=True)
class InferenceAction:
    relative_position: tuple[float, float, float]
    relative_rotation: tuple[float, float, float]

    @classmethod
    def from_response(cls, payload: Mapping[str, Any]) -> "InferenceAction":
        # Also allow a server to wrap its result in {"action": {...}}.
        body = payload.get("action", payload)
        if not isinstance(body, Mapping):
            raise ValueError("inference response action must be an object")
        position = next((body[key] for key in (
            "relative_position", "action.eef_position_delta", "action.relative_position",
            "action.eef_position",
        ) if key in body), None)
        rotation = next((body[key] for key in (
            "relative_rotation", "action.eef_rotation_delta", "action.relative_rotation",
            "action.eef_rotation",
        ) if key in body), None)
        return cls(
            relative_position=_vector3("relative_position", position),
            relative_rotation=_vector3("relative_rotation", rotation),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "relative_position": list(self.relative_position),
            "relative_rotation": list(self.relative_rotation),
        }
