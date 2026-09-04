import math
import config
from control import loader
from utils import response

MODULE = "gripper"

# ============================================================
# Gripper Context
# ============================================================

def _normalize_gripper_name(
    gripper_name,
):
    if not isinstance(
        gripper_name,
        str,
    ):
        raise ValueError(
            "gripper_name must be a string"
        )

    gripper_name = (
        gripper_name
        .strip()
        .lower()
    )

    if not gripper_name:
        raise ValueError(
            "gripper_name must not be empty"
        )

    if gripper_name not in config.GRIPPERS:
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
        loader.get_gripper_driver(
            gripper_name
        )
    )

    driver = (
        loader.get_gripper_driver_name(
            gripper_name
        )
    )

    return (
        gripper_name,
        gripper,
        driver,
        gripper_config,
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


# ============================================================
# Validation Helpers
# ============================================================

def _normalize_number(
    name,
    value,
):
    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} 必須是數值"
        ) from exc

    if not math.isfinite(
        number
    ):
        raise ValueError(
            f"{name} 必須是有限數值"
        )

    return number


def _normalize_ratio(
    name,
    value,
):
    value = (
        _normalize_number(
            name,
            value,
        )
    )

    if not (
        0.0
        <= value
        <= 1.0
    ):
        raise ValueError(
            f"{name} 必須介於 "
            f"0.0 ~ 1.0"
        )

    return value


# ============================================================
# Config Helpers
# ============================================================

def _get_motion_params(
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
            "speed"
        )

    if force is None:
        force = motion.get(
            "force"
        )

    if wait is None:
        wait = motion.get(
            "wait",
            True,
        )

    if timeout is None:
        timeout = motion.get(
            "timeout"
        )

    if speed is None:
        raise ValueError(
            "motion.speed is not configured"
        )

    if force is None:
        raise ValueError(
            "motion.force is not configured"
        )

    if timeout is None:
        raise ValueError(
            "motion.timeout is not configured"
        )

    speed = _normalize_ratio(
        "speed",
        speed,
    )

    force = _normalize_ratio(
        "force",
        force,
    )

    if not isinstance(
        wait,
        bool,
    ):
        raise ValueError(
            "wait must be bool"
        )

    timeout = (
        _normalize_number(
            "timeout",
            timeout,
        )
    )

    if timeout <= 0:
        raise ValueError(
            "timeout 必須大於 0"
        )

    return (
        speed,
        force,
        wait,
        timeout,
    )


# ============================================================
# GET STATUS
# ============================================================

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

                raw_status = (
                    gripper
                    .get_gripper_status()
                )

                if not isinstance(
                    raw_status,
                    dict,
                ):
                    raise RuntimeError(
                        f"gripper driver "
                        f"'{driver}' "
                        f"get_gripper_status() "
                        f"must return dict"
                    )

                position = (
                    raw_status.get(
                        "position"
                    )
                )

                if position is not None:
                    position = (
                        _normalize_ratio(
                            "position",
                            position,
                        )
                    )

                status = {
                    "connected":
                        bool(
                            raw_status.get(
                                "connected",
                                False,
                            )
                        ),

                    "ready":
                        raw_status.get(
                            "ready"
                        ),

                    "moving":
                        raw_status.get(
                            "moving"
                        ),

                    "position":
                        position,

                    "object_detected":
                        raw_status.get(
                            "object_detected"
                        ),

                    "fault":
                        raw_status.get(
                            "fault"
                        ),
                }

                grippers.append(
                    {
                        "gripper_name":
                            current_name,

                        "driver":
                            driver,

                        "status":
                            status,
                    }
                )

            except Exception:
                grippers.append(
                    {
                        "gripper_name":
                            name,

                        "driver":
                            driver,

                        "status": {
                            "connected":
                                False,

                            "ready":
                                None,

                            "moving":
                                None,

                            "position":
                                None,

                            "object_detected":
                                None,

                            "fault":
                                None,
                        },
                    }
                )

        return response.success(
            MODULE,
            action,
            result=True,
            data={
                "grippers":
                    grippers,
            },
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# MOVE
# ============================================================

def move_gripper(
    gripper_name,
    position,
    speed=None,
    force=None,
    wait=None,
    timeout=None,
):
    """
    position:
        0.0 = fully open
        1.0 = fully closed

    speed:
        0.0 ~ 1.0

    force:
        0.0 ~ 1.0
    """

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

        position = (
            _normalize_ratio(
                "position",
                position,
            )
        )

        (
            speed,
            force,
            wait,
            timeout,
        ) = (
            _get_motion_params(
                gripper_config,
                speed,
                force,
                wait,
                timeout,
            )
        )

        if not callable(
            getattr(
                gripper,
                "move_gripper",
                None,
            )
        ):
            raise NotImplementedError(
                f"gripper driver "
                f"'{driver}' "
                f"does not support "
                f"move_gripper"
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

        return response.success(
            MODULE,
            action,
            result=bool(
                result
            ),
            data={
                "gripper_name":
                    gripper_name,

                "target_position":
                    position,

                "speed":
                    speed,

                "force":
                    force,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# OPEN
# ============================================================

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
            _get_motion_params(
                gripper_config,
                speed,
                force,
                wait,
                timeout,
            )
        )

        if not callable(
            getattr(
                gripper,
                "open_gripper",
                None,
            )
        ):
            raise NotImplementedError(
                f"gripper driver "
                f"'{driver}' "
                f"does not support "
                f"open_gripper"
            )

        result = (
            gripper.open_gripper(
                speed=speed,
                force=force,
                wait=wait,
                timeout=timeout,
            )
        )

        return response.success(
            MODULE,
            action,
            result=bool(
                result
            ),
            data={
                "gripper_name":
                    gripper_name,

                "target_position":
                    0.0,

                "speed":
                    speed,

                "force":
                    force,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# CLOSE
# ============================================================

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
            _get_motion_params(
                gripper_config,
                speed,
                force,
                wait,
                timeout,
            )
        )

        if not callable(
            getattr(
                gripper,
                "close_gripper",
                None,
            )
        ):
            raise NotImplementedError(
                f"gripper driver "
                f"'{driver}' "
                f"does not support "
                f"close_gripper"
            )

        result = (
            gripper.close_gripper(
                speed=speed,
                force=force,
                wait=wait,
                timeout=timeout,
            )
        )

        return response.success(
            MODULE,
            action,
            result=bool(
                result
            ),
            data={
                "gripper_name":
                    gripper_name,

                "target_position":
                    1.0,

                "speed":
                    speed,

                "force":
                    force,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# STOP
# ============================================================

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

        if not callable(
            getattr(
                gripper,
                "stop_gripper",
                None,
            )
        ):
            raise NotImplementedError(
                f"gripper driver "
                f"'{driver}' "
                f"does not support "
                f"stop_gripper"
            )

        result = (
            gripper.stop_gripper()
        )

        return response.success(
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
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )