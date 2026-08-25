import config

from control.loader import (
    get_camera_driver,
    get_camera_driver_name,
)

from utils.response import (
    success,
    error,
)


MODULE = "camera"


# ============================================================
# Camera Context
# ============================================================

def _normalize_camera_name(
    camera_name,
):
    if not isinstance(
        camera_name,
        str,
    ) or not camera_name.strip():
        raise ValueError(
            "camera_name must be a non-empty string"
        )

    camera_name = (
        camera_name
        .strip()
        .lower()
    )

    if camera_name not in config.CAMERAS:
        raise ValueError(
            f"Unsupported camera: "
            f"{camera_name}. "
            f"Supported cameras: "
            f"{', '.join(sorted(config.CAMERAS))}"
        )

    return camera_name


def _get_camera_context(
    camera_name,
):
    camera_name = (
        _normalize_camera_name(
            camera_name
        )
    )

    camera_config = (
        config.CAMERAS[
            camera_name
        ]
    )

    camera = (
        get_camera_driver(
            camera_name
        )
    )

    driver_name = (
        get_camera_driver_name(
            camera_name
        )
    )

    return (
        camera_name,
        camera,
        driver_name,
        camera_config,
    )


def _get_target_camera_names(
    camera_name=None,
):
    if camera_name is None:
        return list(
            config.CAMERAS.keys()
        )

    return [
        _normalize_camera_name(
            camera_name
        )
    ]


# ============================================================
# Helpers
# ============================================================

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


def _execution_result(
    action,
    result,
    camera_name,
    driver,
):
    if not isinstance(
        result,
        bool,
    ):
        raise RuntimeError(
            f"camera driver '{driver}' "
            f"must return bool, got "
            f"{type(result).__name__}"
        )

    return success(
        MODULE,
        action,
        result=result,
        data={
            "camera_name":
                camera_name,
        },
        driver=driver,
    )


def _get_stream_config(
    camera_config,
):
    stream = (
        camera_config.get(
            "stream",
            {},
        )
    )

    if not isinstance(
        stream,
        dict,
    ):
        raise ValueError(
            "camera stream config "
            "must be a dictionary"
        )

    return stream


def _get_capabilities(
    camera,
):
    metadata = getattr(
        camera,
        "DRIVER_METADATA",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        return {}

    capabilities = (
        metadata.get(
            "capabilities",
            {},
        )
    )

    if not isinstance(
        capabilities,
        dict,
    ):
        return {}

    return capabilities


def _require_capability(
    camera_name,
    camera,
    capability,
):
    capabilities = (
        _get_capabilities(
            camera
        )
    )

    if not capabilities.get(
        capability,
        False,
    ):
        raise NotImplementedError(
            f"Camera '{camera_name}' "
            f"does not support "
            f"{capability}"
        )


# ============================================================
# Status
# ============================================================

def get_camera_status(
    camera_name=None,
):
    action = "get_camera_status"

    try:
        cameras = []

        for name in (
            _get_target_camera_names(
                camera_name
            )
        ):
            try:
                (
                    current_name,
                    camera,
                    driver,
                    _,
                ) = (
                    _get_camera_context(
                        name
                    )
                )

                cameras.append({
                    "camera_name":
                        current_name,

                    "driver":
                        driver,

                    "status":
                        camera
                        .get_camera_status(),
                })

            except Exception as exc:
                if camera_name is not None:
                    return (
                        _service_error(
                            action,
                            exc,
                        )
                    )

                cameras.append({
                    "camera_name":
                        name,

                    "error":
                        str(exc),
                })

        return success(
            MODULE,
            action,
            result=True,
            data={
                "cameras":
                    cameras,
            },
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
        )


# ============================================================
# Lifecycle
# ============================================================

def start_camera(
    camera_name,
    width=None,
    height=None,
    fps=None,
    enable_color=None,
    enable_depth=None,
    align_to=None,
    frame_timeout_ms=None,
):
    action = "start_camera"
    driver = None

    try:
        (
            camera_name,
            camera,
            driver,
            camera_config,
        ) = (
            _get_camera_context(
                camera_name
            )
        )

        stream = (
            _get_stream_config(
                camera_config
            )
        )

        if width is None:
            width = stream.get(
                "width"
            )

        if height is None:
            height = stream.get(
                "height"
            )

        if fps is None:
            fps = stream.get(
                "fps"
            )

        if enable_color is None:
            enable_color = stream.get(
                "enable_color"
            )

        if enable_depth is None:
            enable_depth = stream.get(
                "enable_depth"
            )

        if align_to is None:
            align_to = stream.get(
                "align_to"
            )

        if frame_timeout_ms is None:
            frame_timeout_ms = (
                stream.get(
                    "frame_timeout_ms"
                )
            )

        result = (
            camera.start_camera(
                width=width,
                height=height,
                fps=fps,
                enable_color=
                    enable_color,
                enable_depth=
                    enable_depth,
                align_to=
                    align_to,
                frame_timeout_ms=
                    frame_timeout_ms,
            )
        )

        return _execution_result(
            action,
            result,
            camera_name,
            driver,
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )


def stop_camera(
    camera_name,
):
    action = "stop_camera"
    driver = None

    try:
        (
            camera_name,
            camera,
            driver,
            _,
        ) = (
            _get_camera_context(
                camera_name
            )
        )

        result = (
            camera.stop_camera()
        )

        return _execution_result(
            action,
            result,
            camera_name,
            driver,
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )


# ============================================================
# Internal Generic Camera APIs
# ============================================================

def get_frame(
    camera_name,
):
    (
        _,
        camera,
        _,
        _,
    ) = (
        _get_camera_context(
            camera_name
        )
    )

    return (
        camera.get_frame()
    )


def get_camera_capabilities(
    camera_name,
):
    (
        _,
        camera,
        _,
        _,
    ) = (
        _get_camera_context(
            camera_name
        )
    )

    return dict(
        _get_capabilities(
            camera
        )
    )


def get_distance_value(
    camera_name,
    x,
    y,
    frame=None,
):
    (
        camera_name,
        camera,
        _,
        _,
    ) = (
        _get_camera_context(
            camera_name
        )
    )

    _require_capability(
        camera_name,
        camera,
        "depth",
    )

    return float(
        camera.get_distance(
            x=x,
            y=y,
            frame=frame,
        )
    )


def deproject_pixel_to_point_value(
    camera_name,
    x,
    y,
    depth=None,
    frame=None,
):
    (
        camera_name,
        camera,
        _,
        _,
    ) = (
        _get_camera_context(
            camera_name
        )
    )

    _require_capability(
        camera_name,
        camera,
        "deprojection",
    )

    point = (
        camera
        .deproject_pixel_to_point(
            x=x,
            y=y,
            depth=depth,
            frame=frame,
        )
    )

    return [
        float(
            point[0]
        ),
        float(
            point[1]
        ),
        float(
            point[2]
        ),
    ]


def get_intrinsics_value(
    camera_name,
    frame=None,
):
    (
        camera_name,
        camera,
        _,
        _,
    ) = (
        _get_camera_context(
            camera_name
        )
    )

    _require_capability(
        camera_name,
        camera,
        "intrinsics",
    )

    intrinsics = (
        camera.get_intrinsics(
            frame=frame
        )
    )

    if not isinstance(
        intrinsics,
        dict,
    ):
        raise RuntimeError(
            "get_intrinsics() must "
            "return dictionary"
        )

    for key in (
        "fx",
        "fy",
        "cx",
        "cy",
    ):
        if key not in intrinsics:
            raise RuntimeError(
                f"intrinsics missing "
                f"'{key}'"
            )

    return intrinsics


def get_point_cloud_value(
    camera_name,
    frame=None,
):
    (
        camera_name,
        camera,
        _,
        _,
    ) = (
        _get_camera_context(
            camera_name
        )
    )

    _require_capability(
        camera_name,
        camera,
        "point_cloud",
    )

    point_cloud = (
        camera.get_point_cloud(
            frame=frame
        )
    )

    if point_cloud is None:
        raise RuntimeError(
            "point cloud unavailable"
        )

    return point_cloud


# ============================================================
# JSON-friendly APIs
# ============================================================

def get_distance(
    camera_name,
    x,
    y,
):
    action = "get_distance"
    driver = None

    try:
        (
            camera_name,
            _,
            driver,
            _,
        ) = (
            _get_camera_context(
                camera_name
            )
        )

        distance = (
            get_distance_value(
                camera_name,
                x,
                y,
            )
        )

        return success(
            MODULE,
            action,
            result=True,
            data={
                "camera_name":
                    camera_name,

                "distance_m":
                    distance,
            },
            driver=driver,
        )

    except NotImplementedError as exc:
        return error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                "DepthNotSupported",
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )


def deproject_pixel_to_point(
    camera_name,
    x,
    y,
    depth=None,
):
    action = (
        "deproject_pixel_to_point"
    )

    driver = None

    try:
        (
            camera_name,
            _,
            driver,
            _,
        ) = (
            _get_camera_context(
                camera_name
            )
        )

        point = (
            deproject_pixel_to_point_value(
                camera_name=
                    camera_name,
                x=x,
                y=y,
                depth=depth,
            )
        )

        return success(
            MODULE,
            action,
            result=True,
            data={
                "camera_name":
                    camera_name,

                "camera_xyz":
                    point,
            },
            driver=driver,
        )

    except NotImplementedError as exc:
        return error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                "DeprojectionNotSupported",
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )


def get_intrinsics(
    camera_name,
):
    action = "get_intrinsics"
    driver = None

    try:
        (
            camera_name,
            _,
            driver,
            _,
        ) = (
            _get_camera_context(
                camera_name
            )
        )

        intrinsics = (
            get_intrinsics_value(
                camera_name
            )
        )

        return success(
            MODULE,
            action,
            result=True,
            data={
                "camera_name":
                    camera_name,

                "intrinsics":
                    intrinsics,
            },
            driver=driver,
        )

    except NotImplementedError as exc:
        return error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                "IntrinsicsNotSupported",
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
            driver,
        )