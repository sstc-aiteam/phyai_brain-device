import threading
import time
import cv2
import numpy as np
import config
from control.yolo_detector import (
    create_detector,
)

from services import (
    camera_service,
    coordinate_service,
)

from utils.response import (
    success,
    error,
)


MODULE = "vision"

_DETECTORS = {}
_DETECTOR_LOCK = threading.RLock()


# ============================================================
# Validation
# ============================================================
def _normalize_model_name(
    model_name,
):
    if not isinstance(
        model_name,
        str,
    ) or not model_name.strip():
        raise ValueError(
            "model_name must be a non-empty string"
        )

    model_name = (
        model_name
        .strip()
        .lower()
    )

    if model_name not in config.YOLO_MODELS:
        raise ValueError(
            f"Unsupported YOLO model: "
            f"{model_name}. "
            f"Supported models: "
            f"{', '.join(sorted(config.YOLO_MODELS))}"
        )

    return model_name


def _as_bool(
    value,
    default=False,
):
    if value is None:
        return default

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        return (
            value
            .strip()
            .lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

    return bool(
        value
    )


def _as_float(
    value,
    default,
):
    if value is None:
        return float(
            default
        )

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return float(
            default
        )


def _as_int(
    value,
    default,
):
    if value is None:
        return int(
            default
        )

    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return int(
            default
        )


# ============================================================
# YOLO
# ============================================================

def _get_yolo_detector(
    model_name,
):
    model_name = (
        _normalize_model_name(
            model_name
        )
    )

    with _DETECTOR_LOCK:
        detector = (
            _DETECTORS.get(
                model_name
            )
        )

        if detector is not None:
            return detector

        model_config = (
            config.YOLO_MODELS[
                model_name
            ]
        )

        driver = (
            model_config.get(
                "driver"
            )
        )

        if (
            not isinstance(
                driver,
                str,
            )
            or driver
            .strip()
            .lower()
            != "yolo"
        ):
            raise ValueError(
                f"Model '{model_name}' "
                "driver must be 'yolo'"
            )

        kwargs = dict(
            model_config.get(
                "kwargs",
                {},
            )
        )

        detector = (
            create_detector(
                **kwargs
            )
        )

        _DETECTORS[
            model_name
        ] = detector

        return detector


# ============================================================
# Camera Data
# ============================================================

def _get_camera_rgb(
    frame,
):
    if not isinstance(
        frame,
        dict,
    ):
        raise RuntimeError(
            "camera frame must be a dictionary"
        )

    camera_rgb = (
        frame.get(
            "color_image"
        )
    )

    if camera_rgb is None:
        raise RuntimeError(
            "camera color image unavailable"
        )

    if not isinstance(
        camera_rgb,
        np.ndarray,
    ):
        raise RuntimeError(
            "camera color image must be numpy.ndarray"
        )

    if (
        camera_rgb.ndim != 3
        or camera_rgb.shape[2] != 3
    ):
        raise RuntimeError(
            "camera_rgb must have shape [H, W, 3]"
        )

    return camera_rgb


def _get_intrinsics(
    camera_name,
    frame,
):
    try:
        return (
            camera_service
            .get_intrinsics_value(
                camera_name=
                    camera_name,
                frame=
                    frame,
            ),
            "available",
        )

    except NotImplementedError:
        return (
            None,
            "not_supported",
        )

    except Exception:
        return (
            None,
            "unavailable",
        )


def _get_point_cloud(
    camera_name,
    frame,
):
    try:
        return (
            camera_service
            .get_point_cloud_value(
                camera_name=
                    camera_name,
                frame=
                    frame,
            ),
            "available",
        )

    except NotImplementedError:
        return (
            None,
            "not_supported",
        )

    except Exception:
        return (
            None,
            "unavailable",
        )


# ============================================================
# Detection 3D Enrichment
# ============================================================

def _enrich_detection_3d(
    camera_name,
    frame,
    detection,
):
    detection = dict(
        detection
    )

    detection.update({
        "distance_m":
            None,

        "camera_xyz":
            None,

        "robot_xyz":
            None,

        "depth_status":
            "not_supported",

        "coordinate_status":
            "not_available",
    })

    center = (
        detection.get(
            "center"
        )
        or {}
    )

    if (
        "x" not in center
        or "y" not in center
    ):
        detection[
            "depth_status"
        ] = "invalid_center"

        return detection

    x = center["x"]
    y = center["y"]

    try:
        distance = (
            camera_service
            .get_distance_value(
                camera_name=
                    camera_name,
                x=x,
                y=y,
                frame=
                    frame,
            )
        )

        detection[
            "distance_m"
        ] = float(
            distance
        )

        detection[
            "depth_status"
        ] = "available"

    except NotImplementedError:
        return detection

    except Exception:
        detection[
            "depth_status"
        ] = "invalid_depth"

        return detection

    try:
        camera_xyz = (
            camera_service
            .deproject_pixel_to_point_value(
                camera_name=
                    camera_name,
                x=x,
                y=y,
                depth=
                    detection[
                        "distance_m"
                    ],
                frame=
                    frame,
            )
        )

        detection[
            "camera_xyz"
        ] = camera_xyz

    except NotImplementedError:
        detection[
            "coordinate_status"
        ] = (
            "deprojection_not_supported"
        )

        return detection

    except Exception:
        detection[
            "coordinate_status"
        ] = (
            "deprojection_failed"
        )

        return detection

    try:
        robot_xyz = (
            coordinate_service
            .camera_xyz_to_robot_xyz_value(
                camera_name=
                    camera_name,
                camera_xyz=
                    camera_xyz,
            )
        )

        detection[
            "robot_xyz"
        ] = robot_xyz

        detection[
            "coordinate_status"
        ] = "available"

    except Exception:
        detection[
            "coordinate_status"
        ] = (
            "robot_transform_failed"
        )

    return detection


# ============================================================
# Core Vision
# ============================================================

def capture_vision(
    camera_name,
    model_name=
        "object_detector",
    run_yolo=True,
    draw=True,
    include_intrinsics=True,
    include_point_cloud=False,
    include_robot_xyz=True,
):
    """
    取得指定 camera 的完整 vision observation。

    Camera name 的驗證、driver 取得與 frame abstraction
    全部由 camera_service 負責。
    """

    frame = (
        camera_service
        .get_frame(
            camera_name
        )
    )

    camera_rgb = (
        _get_camera_rgb(
            frame
        )
    )

    timestamp = (
        frame.get(
            "timestamp"
        )
    )

    camera_intrinsics = None
    intrinsics_status = (
        "not_requested"
    )

    if include_intrinsics:
        (
            camera_intrinsics,
            intrinsics_status,
        ) = (
            _get_intrinsics(
                camera_name,
                frame,
            )
        )

    point_cloud = None
    point_cloud_status = (
        "not_requested"
    )

    if include_point_cloud:
        (
            point_cloud,
            point_cloud_status,
        ) = (
            _get_point_cloud(
                camera_name,
                frame,
            )
        )

    annotated_frame = None
    detections = []

    if run_yolo:
        detector = (
            _get_yolo_detector(
                model_name
            )
        )

        (
            annotated_frame,
            raw_detections,
        ) = (
            detector.predict(
                color_image=
                    camera_rgb,
                draw=
                    draw,
            )
        )

        if include_robot_xyz:
            detections = [
                _enrich_detection_3d(
                    camera_name=
                        camera_name,
                    frame=
                        frame,
                    detection=
                        detection,
                )
                for detection
                in raw_detections
            ]

        else:
            detections = [
                dict(
                    detection
                )
                for detection
                in raw_detections
            ]

    return {
        "camera_name":
            camera_name,

        "timestamp":
            timestamp,

        "camera_rgb":
            camera_rgb,

        "camera_intrinsics":
            camera_intrinsics,

        "intrinsics_status":
            intrinsics_status,

        "point_cloud":
            point_cloud,

        "point_cloud_status":
            point_cloud_status,

        "annotated_frame":
            annotated_frame,

        "detections":
            detections,
    }


# ============================================================
# Internal APIs
# ============================================================

def get_vla_observation(
    camera_name,
    model_name=
        "object_detector",
    run_yolo=True,
    include_point_cloud=True,
):
    result = (
        capture_vision(
            camera_name=
                camera_name,
            model_name=
                model_name,
            run_yolo=
                run_yolo,
            draw=False,
            include_intrinsics=True,
            include_point_cloud=
                include_point_cloud,
            include_robot_xyz=True,
        )
    )

    camera_rgb = (
        result[
            "camera_rgb"
        ]
    )

    return {
        "camera_name":
            result[
                "camera_name"
            ],

        "timestamp":
            result[
                "timestamp"
            ],

        "camera_rgb":
            camera_rgb,

        "camera_rgb_shape":
            list(
                camera_rgb.shape
            ),

        "camera_intrinsics":
            result[
                "camera_intrinsics"
            ],

        "intrinsics_status":
            result[
                "intrinsics_status"
            ],

        "point_cloud":
            result[
                "point_cloud"
            ],

        "point_cloud_status":
            result[
                "point_cloud_status"
            ],

        "yolo_detections":
            result[
                "detections"
            ],
    }


def detect_objects_value(
    camera_name,
    model_name=
        "object_detector",
    draw=True,
    include_robot_xyz=True,
):
    result = (
        capture_vision(
            camera_name=
                camera_name,
            model_name=
                model_name,
            run_yolo=True,
            draw=
                draw,
            include_intrinsics=False,
            include_point_cloud=False,
            include_robot_xyz=
                include_robot_xyz,
        )
    )

    return {
        "camera_name":
            result[
                "camera_name"
            ],

        "timestamp":
            result[
                "timestamp"
            ],

        "camera_rgb":
            result[
                "camera_rgb"
            ],

        "annotated_frame":
            result[
                "annotated_frame"
            ],

        "detections":
            result[
                "detections"
            ],
    }


# ============================================================
# Route-facing Data Preparation
# ============================================================

def _encode_jpeg(
    image,
    quality=90,
):
    if image is None:
        raise RuntimeError(
            "image unavailable"
        )

    success_flag, encoded = (
        cv2.imencode(
            ".jpg",
            image,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                int(
                    quality
                ),
            ],
        )
    )

    if not success_flag:
        raise RuntimeError(
            "failed to encode JPEG"
        )

    return (
        encoded
        .tobytes()
    )


def _make_mjpeg_chunk(
    jpeg_bytes,
):
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + (
            f"Content-Length: "
            f"{len(jpeg_bytes)}\r\n\r\n"
        ).encode(
            "utf-8"
        )
        + jpeg_bytes
        + b"\r\n"
    )


def _point_cloud_summary(
    point_cloud,
):
    if point_cloud is None:
        return None

    if not isinstance(
        point_cloud,
        np.ndarray,
    ):
        return {
            "type":
                type(
                    point_cloud
                ).__name__,
        }

    return {
        "shape":
            list(
                point_cloud.shape
            ),

        "dtype":
            str(
                point_cloud.dtype
            ),

        "point_count":
            int(
                np.prod(
                    point_cloud.shape[:-1]
                )
            )
            if (
                point_cloud.ndim >= 2
                and point_cloud.shape[-1]
                == 3
            )
            else None,
    }


def get_camera_rgb_jpeg(
    camera_name,
    quality=90,
):
    quality = max(
        1,
        min(
            100,
            _as_int(
                quality,
                90,
            ),
        ),
    )

    result = (
        capture_vision(
            camera_name=
                camera_name,
            run_yolo=False,
            draw=False,
            include_intrinsics=False,
            include_point_cloud=False,
            include_robot_xyz=False,
        )
    )

    return (
        _encode_jpeg(
            result[
                "camera_rgb"
            ],
            quality=
                quality,
        )
    )


def get_yolo_image_jpeg(
    camera_name,
    model_name=
        "object_detector",
    include_robot_xyz=True,
    quality=90,
):
    include_robot_xyz = (
        _as_bool(
            include_robot_xyz,
            True,
        )
    )

    quality = max(
        1,
        min(
            100,
            _as_int(
                quality,
                90,
            ),
        ),
    )

    result = (
        detect_objects_value(
            camera_name=
                camera_name,
            model_name=
                model_name,
            draw=True,
            include_robot_xyz=
                include_robot_xyz,
        )
    )

    return (
        _encode_jpeg(
            result[
                "annotated_frame"
            ],
            quality=
                quality,
        )
    )


# ============================================================
# MJPEG Stream
# ============================================================

def camera_stream(
    camera_name,
    interval_sec=0.03,
    quality=85,
):
    """
    原始 RGB MJPEG stream generator。

    Camera name validation 由 camera_service 負責。

    Routes:
        Response(
            vision_service.camera_stream(...),
            mimetype=
                "multipart/x-mixed-replace; "
                "boundary=frame",
        )
    """

    interval_sec = max(
        0.0,
        _as_float(
            interval_sec,
            0.03,
        ),
    )

    quality = max(
        1,
        min(
            100,
            _as_int(
                quality,
                85,
            ),
        ),
    )

    def _generator():
        while True:
            jpeg = (
                get_camera_rgb_jpeg(
                    camera_name=
                        camera_name,
                    quality=
                        quality,
                )
            )

            yield (
                _make_mjpeg_chunk(
                    jpeg
                )
            )

            if interval_sec > 0:
                time.sleep(
                    interval_sec
                )

    return _generator()


def yolo_stream(
    camera_name,
    model_name=
        "object_detector",
    include_robot_xyz=True,
    interval_sec=0.10,
    quality=85,
):
    """
    含 YOLO 標註結果的 MJPEG stream generator。

    Camera name validation 由 camera_service 負責。
    Model validation 由 vision_service 負責。

    Routes:
        Response(
            vision_service.yolo_stream(...),
            mimetype=
                "multipart/x-mixed-replace; "
                "boundary=frame",
        )
    """

    model_name = (
        _normalize_model_name(
            model_name
        )
    )

    include_robot_xyz = (
        _as_bool(
            include_robot_xyz,
            True,
        )
    )

    interval_sec = max(
        0.0,
        _as_float(
            interval_sec,
            0.10,
        ),
    )

    quality = max(
        1,
        min(
            100,
            _as_int(
                quality,
                85,
            ),
        ),
    )

    def _generator():
        while True:
            jpeg = (
                get_yolo_image_jpeg(
                    camera_name=
                        camera_name,
                    model_name=
                        model_name,
                    include_robot_xyz=
                        include_robot_xyz,
                    quality=
                        quality,
                )
            )

            yield (
                _make_mjpeg_chunk(
                    jpeg
                )
            )

            if interval_sec > 0:
                time.sleep(
                    interval_sec
                )

    return _generator()


# ============================================================
# VLA Summary
# ============================================================

def get_vla_observation_summary(
    camera_name,
    model_name=
        "object_detector",
    run_yolo=True,
    include_point_cloud=True,
):
    run_yolo = (
        _as_bool(
            run_yolo,
            True,
        )
    )

    include_point_cloud = (
        _as_bool(
            include_point_cloud,
            True,
        )
    )

    result = (
        get_vla_observation(
            camera_name=
                camera_name,
            model_name=
                model_name,
            run_yolo=
                run_yolo,
            include_point_cloud=
                include_point_cloud,
        )
    )

    camera_rgb = (
        result.get(
            "camera_rgb"
        )
    )

    rgb_summary = None

    if isinstance(
        camera_rgb,
        np.ndarray,
    ):
        rgb_summary = {
            "shape":
                list(
                    camera_rgb.shape
                ),

            "dtype":
                str(
                    camera_rgb.dtype
                ),
        }

    return success(
        MODULE,
        "get_vla_observation_summary",
        result=True,
        data={
            "camera_name":
                result.get(
                    "camera_name"
                ),

            "timestamp":
                result.get(
                    "timestamp"
                ),

            "camera_rgb":
                rgb_summary,

            "camera_intrinsics":
                result.get(
                    "camera_intrinsics"
                ),

            "intrinsics_status":
                result.get(
                    "intrinsics_status"
                ),

            "point_cloud":
                _point_cloud_summary(
                    result.get(
                        "point_cloud"
                    )
                ),

            "point_cloud_status":
                result.get(
                    "point_cloud_status"
                ),

            "yolo_detections":
                result.get(
                    "yolo_detections",
                    [],
                ),
        },
    )


# ============================================================
# DETECTIONS
# ============================================================

def get_detections(
    camera_name,
    model_name=
        "object_detector",
    include_robot_xyz=True,
):
    action = "get_detections"

    try:
        include_robot_xyz = (
            _as_bool(
                include_robot_xyz,
                True,
            )
        )

        result = (
            detect_objects_value(
                camera_name=
                    camera_name,
                model_name=
                    model_name,
                draw=False,
                include_robot_xyz=
                    include_robot_xyz,
            )
        )

        return success(
            MODULE,
            action,
            result=True,
            data={
                "camera_name":
                    result[
                        "camera_name"
                    ],

                "timestamp":
                    result[
                        "timestamp"
                    ],

                "detections":
                    result[
                        "detections"
                    ],
            },
        )

    except Exception as exc:
        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
        )