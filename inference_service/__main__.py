import logging

from .client import InferenceClient
from .config import Settings
from .executor import HTTPArmExecutor
from .service import LiveInferenceService
from .sources import HTTPJPEGSource, URTCPPoseSource


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.from_env()
    executor = None
    if settings.execute_actions:
        executor = HTTPArmExecutor(
            settings.arm_api_url,
            settings.arm_name,
            settings.request_timeout,
            settings.action_frame,
            settings.max_translation_delta,
            settings.max_rotation_delta,
            settings.motion_speed,
            settings.motion_acceleration,
        )
        logging.warning("LIVE ARM EXECUTION ENABLED for arm=%s", settings.arm_name)

    service = LiveInferenceService(
        pose_source=URTCPPoseSource(settings.robot_ip),
        image_source=HTTPJPEGSource(settings.image_url, settings.request_timeout),
        client=InferenceClient(
            settings.inference_host,
            settings.inference_port,
            settings.request_timeout,
            settings.gripper_position,
            settings.instruction,
        ),
        hz=settings.hz,
        on_action=executor,
    )
    service.install_signal_handlers()
    try:
        service.run_forever()
    finally:
        service.close()
        if executor is not None:
            executor.close()


if __name__ == "__main__":
    main()
