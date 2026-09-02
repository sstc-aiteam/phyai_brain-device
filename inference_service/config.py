from __future__ import annotations

import os
from dataclasses import dataclass


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    robot_ip: str = "192.168.50.76"
    image_url: str = "http://127.0.0.1:5001/api/vision/camera_rgb?camera_name=left"
    inference_host: str = "192.168.50.215"
    inference_port: int = 5555
    hz: float = 10.0
    request_timeout: float = 5.0
    jpeg_quality: int = 90
    gripper_position: float = 0.0
    instruction: str = "open drawer"
    execute_actions: bool = False
    arm_api_url: str = "http://127.0.0.1:5001"
    arm_name: str = "left"
    action_frame: str = "tool"
    max_translation_delta: float = 0.025
    max_rotation_delta: float = 0.05
    motion_speed: float = 0.05
    motion_acceleration: float = 0.1

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            robot_ip=os.getenv("INFERENCE_ROBOT_IP", cls.robot_ip),
            image_url=os.getenv("INFERENCE_IMAGE_URL", cls.image_url),
            inference_host=os.getenv("INFERENCE_HOST", cls.inference_host),
            inference_port=int(os.getenv("INFERENCE_PORT", str(cls.inference_port))),
            hz=_float_env("INFERENCE_HZ", cls.hz),
            request_timeout=_float_env("INFERENCE_TIMEOUT", cls.request_timeout),
            jpeg_quality=int(os.getenv("INFERENCE_JPEG_QUALITY", str(cls.jpeg_quality))),
            gripper_position=_float_env("INFERENCE_GRIPPER_POSITION", cls.gripper_position),
            instruction=os.getenv("INFERENCE_INSTRUCTION", cls.instruction),
            execute_actions=_bool_env("INFERENCE_EXECUTE_ACTIONS", cls.execute_actions),
            arm_api_url=os.getenv("INFERENCE_ARM_API_URL", cls.arm_api_url),
            arm_name=os.getenv("INFERENCE_ARM_NAME", cls.arm_name),
            action_frame=os.getenv("INFERENCE_ACTION_FRAME", cls.action_frame).lower(),
            max_translation_delta=_float_env(
                "INFERENCE_MAX_TRANSLATION_DELTA", cls.max_translation_delta
            ),
            max_rotation_delta=_float_env(
                "INFERENCE_MAX_ROTATION_DELTA", cls.max_rotation_delta
            ),
            motion_speed=_float_env("INFERENCE_MOTION_SPEED", cls.motion_speed),
            motion_acceleration=_float_env(
                "INFERENCE_MOTION_ACCELERATION", cls.motion_acceleration
            ),
        )
        if settings.hz <= 0:
            raise ValueError("INFERENCE_HZ must be greater than zero")
        if not 1 <= settings.jpeg_quality <= 100:
            raise ValueError("INFERENCE_JPEG_QUALITY must be between 1 and 100")
        if settings.action_frame not in {"tool", "base"}:
            raise ValueError("INFERENCE_ACTION_FRAME must be tool or base")
        if settings.max_translation_delta <= 0 or settings.max_rotation_delta <= 0:
            raise ValueError("inference action limits must be greater than zero")
        return settings
