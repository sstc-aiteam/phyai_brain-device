import config

from control.loader import (
    get_camera_driver,
    get_camera_driver_name,
)

from utils import response


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
    ):
        raise ValueError(
            "camera_name must be a string"
        )

    camera_name = (
        camera_name
        .strip()
        .lower()
    )

    if not camera_name:
        raise ValueError(
            "camera_name must not be empty"
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

    driver = (
        get_camera_driver_name(
            camera_name
        )
    )

    return (
        camera_name,
        camera,
        driver,
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
# Config
# ============================================================

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


# ============================================================
# Capabilities
# ============================================================

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

    return dict(
        capabilities
    )


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
# GET STATUS
# ============================================================

def get_camera_status(
    camera_name=None,
):
    """
    Standard camera status schema:

    {
        "connected": bool,
        "running": bool | None,
        "width": int | None,
        "height": int | None,
        "fps": float | int | None,
        "color_enabled": bool | None,
        "depth_enabled": bool | None,
        "fault": bool | None
    }

    回傳 config.CAMERAS 中所有已設定的 camera。

    Camera 無法建立、連線或取得狀態時，
    仍保留該 camera 並標記 connected=False。
    """

    action = "get_camera_status"

    try:
        cameras = []

        for name in (
            _get_target_camera_names(
                camera_name
            )
        ):
            driver = None

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

                raw_status = (
                    camera
                    .get_camera_status()
                )

                if not isinstance(
                    raw_status,
                    dict,
                ):
                    raise RuntimeError(
                        f"camera driver "
                        f"'{driver}' "
                        f"get_camera_status() "
                        f"must return dict"
                    )

                status = {
                    "connected":
                        bool(
                            raw_status.get(
                                "connected",
                                False,
                            )
                        ),

                    "running":
                        raw_status.get(
                            "running"
                        ),

                    "width":
                        raw_status.get(
                            "width"
                        ),

                    "height":
                        raw_status.get(
                            "height"
                        ),

                    "fps":
                        raw_status.get(
                            "fps"
                        ),

                    "color_enabled":
                        raw_status.get(
                            "color_enabled"
                        ),

                    "depth_enabled":
                        raw_status.get(
                            "depth_enabled"
                        ),

                    "fault":
                        raw_status.get(
                            "fault"
                        ),
                }

                cameras.append(
                    {
                        "camera_name":
                            current_name,

                        "driver":
                            driver,

                        "status":
                            status,
                    }
                )

            except Exception:
                cameras.append(
                    {
                        "camera_name":
                            name,

                        "driver":
                            driver,

                        "status": {
                            "connected":
                                False,

                            "running":
                                None,

                            "width":
                                None,

                            "height":
                                None,

                            "fps":
                                None,

                            "color_enabled":
                                None,

                            "depth_enabled":
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
                "cameras":
                    cameras,
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
# START CAMERA
# ============================================================

def start_camera(
    camera_name,
    width=None,
    height=None,
    fps=None,
):
    """
    啟動指定 camera。

    Public generic parameters:
        width
        height
        fps

    其他 stream-specific 設定由 config.CAMERAS
    的 stream configuration 提供。
    """

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

        result = (
            camera.start_camera(
                width=width,
                height=height,
                fps=fps,
                enable_color=
                    stream.get(
                        "enable_color"
                    ),
                enable_depth=
                    stream.get(
                        "enable_depth"
                    ),
                align_to=
                    stream.get(
                        "align_to"
                    ),
                frame_timeout_ms=
                    stream.get(
                        "frame_timeout_ms"
                    ),
            )
        )

        if not isinstance(
            result,
            bool,
        ):
            raise RuntimeError(
                f"camera driver "
                f"'{driver}' "
                f"start_camera() "
                f"must return bool"
            )

        return response.success(
            MODULE,
            action,
            result=result,
            data={
                "camera_name":
                    camera_name,

                "width":
                    width,

                "height":
                    height,

                "fps":
                    fps,
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
# STOP CAMERA
# ============================================================

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

        if not isinstance(
            result,
            bool,
        ):
            raise RuntimeError(
                f"camera driver "
                f"'{driver}' "
                f"stop_camera() "
                f"must return bool"
            )

        return response.success(
            MODULE,
            action,
            result=result,
            data={
                "camera_name":
                    camera_name,
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
# GET FRAME
# Internal Python API
# ============================================================

def get_frame(
    camera_name,
):
    """
    Standard frame contract:

    {
        "timestamp": float | None,
        "color_image": numpy.ndarray | None,
        "depth_image": numpy.ndarray | None
    }

    此 function 給其他 service 直接使用，
    不包裝成 JSON response。
    """

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

    frame = (
        camera.get_frame()
    )

    if not isinstance(
        frame,
        dict,
    ):
        raise RuntimeError(
            "camera get_frame() "
            "must return dictionary"
        )

    return frame


# ============================================================
# GET CAPABILITIES
# Internal Python API
# ============================================================

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

    return (
        _get_capabilities(
            camera
        )
    )


# ============================================================
# DISTANCE
# Internal Python API
# ============================================================

def get_distance_value(
    camera_name,
    x,
    y,
    frame=None,
):
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

    _require_capability(
        camera_name,
        camera,
        "depth",
    )

    if not callable(
        getattr(
            camera,
            "get_distance",
            None,
        )
    ):
        raise NotImplementedError(
            f"camera driver "
            f"'{driver}' "
            f"does not support "
            f"get_distance"
        )

    distance = (
        camera.get_distance(
            x=x,
            y=y,
            frame=frame,
        )
    )

    return float(
        distance
    )


# ============================================================
# DISTANCE
# JSON-friendly API
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
            camera,
            driver,
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

        if not callable(
            getattr(
                camera,
                "get_distance",
                None,
            )
        ):
            raise NotImplementedError(
                f"camera driver "
                f"'{driver}' "
                f"does not support "
                f"get_distance"
            )

        distance = float(
            camera.get_distance(
                x=x,
                y=y,
                frame=None,
            )
        )

        return response.success(
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
# DEPROJECT
# Internal Python API
# ============================================================

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
        driver,
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

    if not callable(
        getattr(
            camera,
            "deproject_pixel_to_point",
            None,
        )
    ):
        raise NotImplementedError(
            f"camera driver "
            f"'{driver}' "
            f"does not support "
            f"deproject_pixel_to_point"
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

    if (
        not isinstance(
            point,
            (
                list,
                tuple,
            ),
        )
        or len(point) != 3
    ):
        raise RuntimeError(
            "deproject_pixel_to_point() "
            "must return [x, y, z]"
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


# ============================================================
# DEPROJECT
# JSON-friendly API
# ============================================================

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
            camera,
            driver,
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

        if not callable(
            getattr(
                camera,
                "deproject_pixel_to_point",
                None,
            )
        ):
            raise NotImplementedError(
                f"camera driver "
                f"'{driver}' "
                f"does not support "
                f"deproject_pixel_to_point"
            )

        point = (
            camera
            .deproject_pixel_to_point(
                x=x,
                y=y,
                depth=depth,
                frame=None,
            )
        )

        if (
            not isinstance(
                point,
                (
                    list,
                    tuple,
                ),
            )
            or len(point) != 3
        ):
            raise RuntimeError(
                "deproject_pixel_to_point() "
                "must return [x, y, z]"
            )

        camera_xyz = [
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

        return response.success(
            MODULE,
            action,
            result=True,
            data={
                "camera_name":
                    camera_name,

                "camera_xyz":
                    camera_xyz,
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
# INTRINSICS
# Internal Python API
# ============================================================

def get_intrinsics_value(
    camera_name,
    frame=None,
):
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

    _require_capability(
        camera_name,
        camera,
        "intrinsics",
    )

    if not callable(
        getattr(
            camera,
            "get_intrinsics",
            None,
        )
    ):
        raise NotImplementedError(
            f"camera driver "
            f"'{driver}' "
            f"does not support "
            f"get_intrinsics"
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


# ============================================================
# INTRINSICS
# JSON-friendly API
# ============================================================

def get_intrinsics(
    camera_name,
):
    action = "get_intrinsics"
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

        _require_capability(
            camera_name,
            camera,
            "intrinsics",
        )

        if not callable(
            getattr(
                camera,
                "get_intrinsics",
                None,
            )
        ):
            raise NotImplementedError(
                f"camera driver "
                f"'{driver}' "
                f"does not support "
                f"get_intrinsics"
            )

        intrinsics = (
            camera.get_intrinsics(
                frame=None
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

        return response.success(
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
# POINT CLOUD
# Internal Python API
# ============================================================

def get_point_cloud_value(
    camera_name,
    frame=None,
):
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

    _require_capability(
        camera_name,
        camera,
        "point_cloud",
    )

    if not callable(
        getattr(
            camera,
            "get_point_cloud",
            None,
        )
    ):
        raise NotImplementedError(
            f"camera driver "
            f"'{driver}' "
            f"does not support "
            f"get_point_cloud"
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