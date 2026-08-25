import config
from control.loader import (get_gripper_driver, get_gripper_driver_name)
from utils.response import (success, error)

MODULE = "gripper"

def _normalize_gripper_name(
    gripper_name,
):
    if not isinstance(
        gripper_name,
        str,
    ) or not gripper_name.strip():
        raise ValueError(
            "gripper_name must be "
            "a non-empty string"
        )

    gripper_name = (
        gripper_name
        .strip()
        .lower()
    )

    if gripper_name not in (
        config.GRIPPERS
    ):
        raise ValueError(
            f"Unsupported gripper: "
            f"{gripper_name}. "
            f"Supported grippers: "
            f"{', '.join(sorted(config.GRIPPERS))}"
        )

    return gripper_name


def _get_gripper_context(
    gripper_name,
):
    gripper_name = (
        _normalize_gripper_name(
            gripper_name
        )
    )

    gripper_config = (
        config.GRIPPERS[
            gripper_name
        ]
    )

    gripper = (
        get_gripper_driver(
            gripper_name
        )
    )

    driver = (
        get_gripper_driver_name(
            gripper_name
        )
    )

    return (
        gripper_name,
        gripper,
        driver,
        gripper_config,
    )


def _service_error(
    action,
    exc,
    driver=None,
):
    return error(
        MODULE,
        action,
        error=exc,
        driver=driver,
        error_type=
            type(exc).__name__,
    )


def _get_target_gripper_names(
    gripper_name=None,
):
    if gripper_name is None:
        return list(
            config.GRIPPERS.keys()
        )

    return [
        _normalize_gripper_name(
            gripper_name
        )
    ]


def _resolve_motion(
    gripper_config,
    speed=None,
    force=None,
    wait=None,
    timeout=None,
):
    motion = (
        gripper_config.get(
            "motion",
            {},
        )
    )

    if speed is None:
        speed = motion.get(
            "speed",
            255,
        )

    if force is None:
        force = motion.get(
            "force",
            150,
        )

    if wait is None:
        wait = motion.get(
            "wait",
            True,
        )

    if timeout is None:
        timeout = motion.get(
            "timeout",
            5.0,
        )

    return (
        speed,
        force,
        wait,
        timeout,
    )


def get_gripper_status(
    gripper_name=None,
):
    action = "get_gripper_status"

    try:
        grippers = []

        for name in (
            _get_target_gripper_names(
                gripper_name
            )
        ):
            driver = None

            try:
                (
                    current_name,
                    gripper,
                    driver,
                    _,
                ) = (
                    _get_gripper_context(
                        name
                    )
                )

                status = (
                    gripper
                    .get_gripper_status()
                )

                grippers.append({
                    "gripper_name":
                        current_name,
                    "driver":
                        driver,
                    **status,
                })

            except Exception as exc:
                if gripper_name is not None:
                    return (
                        _service_error(
                            action,
                            exc,
                            driver,
                        )
                    )

                grippers.append({
                    "gripper_name":
                        name,
                    "driver":
                        driver,
                    "connected":
                        False,
                    "error":
                        str(exc),
                })

        return success(
            MODULE,
            action,
            result=True,
            data={
                "grippers":
                    grippers,
            },
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
        )


def get_gripper_status_value(
    gripper_name,
):
    (
        _,
        gripper,
        _,
        _,
    ) = (
        _get_gripper_context(
            gripper_name
        )
    )

    return (
        gripper
        .get_gripper_status()
    )


def move_gripper(
    gripper_name,
    position,
    speed=None,
    force=None,
    wait=None,
    timeout=None,
):
    action = "move_gripper"
    driver = None

    try:
        (
            gripper_name,
            gripper,
            driver,
            gripper_config,
        ) = (
            _get_gripper_context(
                gripper_name
            )
        )

        (
            speed,
            force,
            wait,
            timeout,
        ) = (
            _resolve_motion(
                gripper_config,
                speed,
                force,
                wait,
                timeout,
            )
        )

        result = (
            gripper.move_gripper(
                position=position,
                speed=speed,
                force=force,
                wait=wait,
                timeout=timeout,
            )
        )

        return success(
            MODULE,
            action,
            result=bool(
                result
            ),
            data={
                "gripper_name":
                    gripper_name,
                "position":
                    int(position),
                "speed":
                    int(speed),
                "force":
                    int(force),
            },
            driver=driver,
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )


def open_gripper(
    gripper_name,
    speed=None,
    force=None,
    wait=None,
    timeout=None,
):
    action = "open_gripper"
    driver = None

    try:
        (
            gripper_name,
            gripper,
            driver,
            gripper_config,
        ) = (
            _get_gripper_context(
                gripper_name
            )
        )

        (
            speed,
            force,
            wait,
            timeout,
        ) = (
            _resolve_motion(
                gripper_config,
                speed,
                force,
                wait,
                timeout,
            )
        )

        result = (
            gripper.open_gripper(
                speed=speed,
                force=force,
                wait=wait,
                timeout=timeout,
            )
        )

        return success(
            MODULE,
            action,
            result=bool(
                result
            ),
            data={
                "gripper_name":
                    gripper_name,
                "position":
                    0,
                "speed":
                    int(speed),
                "force":
                    int(force),
            },
            driver=driver,
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )


def close_gripper(
    gripper_name,
    speed=None,
    force=None,
    wait=None,
    timeout=None,
):
    action = "close_gripper"
    driver = None

    try:
        (
            gripper_name,
            gripper,
            driver,
            gripper_config,
        ) = (
            _get_gripper_context(
                gripper_name
            )
        )

        (
            speed,
            force,
            wait,
            timeout,
        ) = (
            _resolve_motion(
                gripper_config,
                speed,
                force,
                wait,
                timeout,
            )
        )

        result = (
            gripper.close_gripper(
                speed=speed,
                force=force,
                wait=wait,
                timeout=timeout,
            )
        )

        return success(
            MODULE,
            action,
            result=bool(
                result
            ),
            data={
                "gripper_name":
                    gripper_name,
                "position":
                    255,
                "speed":
                    int(speed),
                "force":
                    int(force),
            },
            driver=driver,
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )


def stop_gripper(
    gripper_name,
):
    action = "stop_gripper"
    driver = None

    try:
        (
            gripper_name,
            gripper,
            driver,
            _,
        ) = (
            _get_gripper_context(
                gripper_name
            )
        )

        result = (
            gripper.stop_gripper()
        )

        return success(
            MODULE,
            action,
            result=bool(
                result
            ),
            data={
                "gripper_name":
                    gripper_name,
            },
            driver=driver,
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )