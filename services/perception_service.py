import time
import math
import cv2
import numpy as np
import config
from services import camera_service, coordinate_service, model_service
from utils import response


MODULE = "perception"

def _normalize_model_name(model_name):
    if not isinstance(model_name, str) or model_name.strip().lower() not in ("object_detector", "yolo"):
        raise ValueError("Supported detection models: object_detector, yolo")
    return "yolo"


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
# YOLO segmentation
# ============================================================

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
CENTER_COLOR = (0, 0, 255)
CENTER_RADIUS = 4
CENTER_CROSS_SIZE = 8
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 128, 0)
INFO_TEXT_COLOR = (0, 255, 255)
TEXT_SCALE = 0.42
TEXT_THICKNESS = 1
TEXT_PADDING = 4
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
ORIENTATION_AXIS_COLOR = (255, 128, 0)


def _normalize_mask_points(raw_points, frame_shape):
    """Clamp the Docker segmentation polygon to this camera frame's pixel grid."""
    height, width = frame_shape[:2]
    points = []
    for point in raw_points or []:
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            continue
        try:
            x, y = float(point["x"]), float(point["y"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        points.append({
            "x": max(0, min(int(round(x)), width - 1)),
            "y": max(0, min(int(round(y)), height - 1)),
        })
    return points


def _enrich_yolo_geometry(detection, frame_shape):
    """Restore the old contour center, minAreaRect yaw and PCA debug contract."""
    detection = dict(detection)
    height, width = frame_shape[:2]
    box = detection.get("box") or {}
    detection["box"] = {
        k: max(0, min(int(round(float(box[k]))), width - 1 if k in ("x1", "x2") else height - 1))
        for k in ("x1", "y1", "x2", "y2")
    }
    center = detection.get("center") or {}
    detection["center"] = {
        "x": int(round(float(center["x"]))),
        "y": int(round(float(center["y"]))),
    }
    points = _normalize_mask_points(detection.get("segmentation_points"), frame_shape)
    has_mask = len(points) >= 3
    detection["task"] = "segmentation" if has_mask else "detection"
    detection["segmentation_points"] = points if has_mask else []
    detection["yaw_deg"] = None
    detection["pca_status"] = None
    detection["pca_axis_ratio"] = None
    detection["pca_debug"] = None
    detection["orientation_debug"] = None

    if not has_mask:
        # The model's existing box-based center is retained for detection-only models.
        return detection

    center = _get_points_center(points)
    if center is not None:
        detection["center"] = {"x": int(center[0]), "y": int(center[1])}

    orientation = _calc_orientation_debug(
        points=points,
        frame_shape=frame_shape,
        class_name=detection.get("class_name", ""),
    )
    pca = orientation.get("pca_debug")
    detection["orientation_debug"] = orientation
    detection["pca_debug"] = pca
    detection["yaw_deg"] = orientation.get("yaw_deg")
    detection["pca_status"] = None if pca is None else pca.get("status")
    detection["pca_axis_ratio"] = None if pca is None else pca.get("axis_ratio")
    return detection


def _draw_yolo_image(camera_image, detections):
    """Draw the legacy contour, main-axis, center, class and yaw annotations."""
    annotated = camera_image.copy()
    for detection in detections:
        _draw_detection(annotated, detection)
    return annotated


def _get_points_center(points):
        """
        使用 segmentation polygon points 計算中心點。

        優先使用 cv2.moments 計算輪廓中心。
        如果 moments 失敗，改用所有點位的平均值。
        """

        if not points:
            return None

        contour = np.array(
            [[point["x"], point["y"]] for point in points],
            dtype=np.int32,
        )

        if len(contour) >= 3:
            moments = cv2.moments(contour)

            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                return center_x, center_y

        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]

        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))

        return center_x, center_y


def _calc_pca_debug_from_points(points):
        """
        使用 segmentation polygon points 計算 PCA 主軸、次軸與 yaw。
        """

        if not points or len(points) < 20:
            return {
                "yaw_deg": None,
                "axis_ratio": None,
                "status": "not_enough_points",
            }

        pts = np.array(
            [[point["x"], point["y"]] for point in points],
            dtype=np.float32,
        )

        if pts.ndim != 2 or pts.shape[1] != 2:
            return {
                "yaw_deg": None,
                "axis_ratio": None,
                "status": "invalid_points",
            }

        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

        if eigenvectors is None or eigenvalues is None:
            return {
                "yaw_deg": None,
                "axis_ratio": None,
                "status": "invalid_pca",
            }

        if len(eigenvectors) < 2 or len(eigenvalues) < 2:
            return {
                "yaw_deg": None,
                "axis_ratio": None,
                "status": "invalid_pca",
            }

        major_value = float(eigenvalues[0][0])
        minor_value = float(eigenvalues[1][0])

        if minor_value == 0.0:
            return {
                "yaw_deg": None,
                "axis_ratio": None,
                "status": "zero_minor_axis",
            }

        axis_ratio = major_value / minor_value
        major_vector = eigenvectors[0]
        minor_vector = eigenvectors[1]
        center_x = float(mean[0][0])
        center_y = float(mean[0][1])

        yaw_deg = math.degrees(math.atan2(float(major_vector[0]), float(major_vector[1])))

        if yaw_deg > 90.0:
            yaw_deg -= 180.0
        elif yaw_deg < -90.0:
            yaw_deg += 180.0

        major_half_length = max(24.0, math.sqrt(max(major_value, 0.0)) * 2.5)
        minor_half_length = max(12.0, math.sqrt(max(minor_value, 0.0)) * 2.5)

        status = "success"
        if axis_ratio < 1.4:
            status = "weak_principal_axis"

        return {
            "yaw_deg": float(yaw_deg) if status == "success" else None,
            "axis_ratio": float(axis_ratio),
            "status": status,
            "center": {
                "x": int(round(center_x)),
                "y": int(round(center_y)),
            },
            "major_axis": {
                "vx": float(major_vector[0]),
                "vy": float(major_vector[1]),
                "half_length": float(major_half_length),
            },
            "minor_axis": {
                "vx": float(minor_vector[0]),
                "vy": float(minor_vector[1]),
                "half_length": float(minor_half_length),
            },
        }


def _build_orientation_contour(points, frame_width, frame_height, class_name):
        """
        segmentation polygon -> binary mask -> morphology -> contour。
        對 bottle_alcohol_spray 額外只取下半部瓶身。
        """

        if not points:
            return None, "no_points", "full_mask"

        contour = np.array(
            [[point["x"], point["y"]] for point in points],
            dtype=np.int32,
        )

        if len(contour) < 3:
            return None, "not_enough_points", "full_mask"

        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [contour.reshape((-1, 1, 2))], 255)

        kernel_open = np.ones((3, 3), dtype=np.uint8)
        kernel_close = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        roi_mode = "full_mask"
        if class_name == "bottle_alcohol_spray":
            ys, xs = np.where(mask > 0)
            if len(xs) >= 10 and len(ys) >= 10:
                y_min = int(np.min(ys))
                y_max = int(np.max(ys))
                crop_start = int(round(y_min + (y_max - y_min) * 0.35))
                cropped_mask = np.zeros_like(mask)
                cropped_mask[crop_start:, :] = mask[crop_start:, :]

                if np.count_nonzero(cropped_mask) >= 30:
                    mask = cropped_mask
                    roi_mode = "lower_body_mask"

        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            return None, "no_contour_after_cleanup", roi_mode

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < 20.0:
            return None, "contour_too_small", roi_mode

        return largest, "success", roi_mode


def _calc_orientation_debug(points, frame_shape, class_name):
        """
        用清理後的 mask 輪廓計算 orientation。
        優先使用 minAreaRect 長邊方向，並保留 PCA debug 供比對。
        bottle_alcohol_spray 只用下半部瓶身，降低噴頭影響。
        """

        frame_height, frame_width = frame_shape[:2]
        contour, contour_status, roi_mode = _build_orientation_contour(
            points=points,
            frame_width=frame_width,
            frame_height=frame_height,
            class_name=class_name,
        )

        pca_debug = _calc_pca_debug_from_points(points)

        if contour is None or len(contour) < 5:
            return {
                "yaw_deg": None,
                "status": contour_status,
                "method": "min_area_rect",
                "roi_mode": roi_mode,
                "pca_debug": pca_debug,
            }

        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle_deg = rect

        if width <= 1.0 and height <= 1.0:
            return {
                "yaw_deg": None,
                "status": "degenerate_rect",
                "method": "min_area_rect",
                "roi_mode": roi_mode,
                "pca_debug": pca_debug,
            }

        use_width_axis = width >= height
        long_side = max(float(width), float(height))
        short_side = max(1e-6, min(float(width), float(height)))
        axis_ratio = long_side / short_side

        if use_width_axis:
            orientation_deg = float(angle_deg)
        else:
            orientation_deg = float(angle_deg) + 90.0

        while orientation_deg > 90.0:
            orientation_deg -= 180.0
        while orientation_deg < -90.0:
            orientation_deg += 180.0

        theta = math.radians(orientation_deg)
        axis_dx = math.cos(theta)
        axis_dy = math.sin(theta)
        half_length = max(24.0, long_side * 0.5)

        status = "success" if axis_ratio >= 1.2 else "weak_principal_axis"

        return {
            "yaw_deg": float(orientation_deg) if status == "success" else None,
            "status": status,
            "method": "min_area_rect",
            "roi_mode": roi_mode,
            "axis_ratio": float(axis_ratio),
            "center": {
                "x": int(round(float(cx))),
                "y": int(round(float(cy))),
            },
            "major_axis": {
                "vx": float(axis_dx),
                "vy": float(axis_dy),
                "half_length": float(half_length),
            },
            "box_points": [
                {"x": int(round(point[0])), "y": int(round(point[1]))}
                for point in cv2.boxPoints(rect)
            ],
            "pca_debug": pca_debug,
        }


def _draw_segmentation(frame, points):
        if not points:
            return

        contour = np.array(
            [[point["x"], point["y"]] for point in points],
            dtype=np.int32,
        )

        if len(contour) < 2:
            return

        contour = contour.reshape((-1, 1, 2))

        cv2.polylines(
            frame,
            [contour],
            isClosed=True,
            color=BOX_COLOR,
            thickness=BOX_THICKNESS,
        )


def _draw_orientation_axis(frame, orientation_debug):
        if not isinstance(orientation_debug, dict):
            return

        center = orientation_debug.get("center") or {}
        major_axis = orientation_debug.get("major_axis") or {}
        if not center or not major_axis:
            return

        cx = int(center.get("x", 0))
        cy = int(center.get("y", 0))
        vx = float(major_axis.get("vx", 0.0))
        vy = float(major_axis.get("vy", 0.0))
        half_length = float(major_axis.get("half_length", 0.0))

        start = (
            int(round(cx - vx * half_length)),
            int(round(cy - vy * half_length)),
        )
        end = (
            int(round(cx + vx * half_length)),
            int(round(cy + vy * half_length)),
        )

        cv2.line(frame, start, end, ORIENTATION_AXIS_COLOR, 3, cv2.LINE_AA)


def _draw_label_with_background(frame, text, x, y, bg_color):
        text_size, baseline = cv2.getTextSize(
            text,
            TEXT_FONT,
            TEXT_SCALE,
            TEXT_THICKNESS,
        )

        text_width, text_height = text_size

        x1 = max(x, 0)
        y1 = max(y - text_height - TEXT_PADDING * 2, 0)
        x2 = min(x + text_width + TEXT_PADDING * 2, frame.shape[1] - 1)
        y2 = min(y, frame.shape[0] - 1)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            bg_color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x1 + TEXT_PADDING, y2 - TEXT_PADDING),
            TEXT_FONT,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )


def _draw_detection(frame, detection):
        x1 = detection["box"]["x1"]
        y1 = detection["box"]["y1"]
        x2 = detection["box"]["x2"]
        y2 = detection["box"]["y2"]

        center_x = detection["center"]["x"]
        center_y = detection["center"]["y"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]
        task = detection.get("task", "detection")

        if task == "segmentation" and detection.get("segmentation_points"):
            _draw_segmentation(
                frame=frame,
                points=detection["segmentation_points"],
            )
            _draw_orientation_axis(
                frame=frame,
                orientation_debug=detection.get("orientation_debug"),
            )
        else:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                BOX_THICKNESS,
            )

        cv2.circle(
            frame,
            (center_x, center_y),
            CENTER_RADIUS,
            CENTER_COLOR,
            -1,
        )

        cv2.line(
            frame,
            (center_x - CENTER_CROSS_SIZE, center_y),
            (center_x + CENTER_CROSS_SIZE, center_y),
            CENTER_COLOR,
            1,
        )

        cv2.line(
            frame,
            (center_x, center_y - CENTER_CROSS_SIZE),
            (center_x, center_y + CENTER_CROSS_SIZE),
            CENTER_COLOR,
            1,
        )

        object_label = f"{class_name} {confidence:.2f} [{task}]"

        _draw_label_with_background(
            frame=frame,
            text=object_label,
            x=x1,
            y=y1,
            bg_color=TEXT_BG_COLOR,
        )

        center_text = f"center: ({center_x}, {center_y})"

        cv2.putText(
            frame,
            center_text,
            (x1, min(y2 + 18, frame.shape[0] - 8)),
            TEXT_FONT,
            TEXT_SCALE,
            INFO_TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )


        yaw_deg = detection.get("yaw_deg")
        if yaw_deg is not None:
            yaw_text = f"yaw_deg: {float(yaw_deg):.2f}"
            cv2.putText(
                frame,
                yaw_text,
                (x1, min(y2 + 36, frame.shape[0] - 8)),
                TEXT_FONT,
                TEXT_SCALE,
                INFO_TEXT_COLOR,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )
        elif detection.get("task") == "segmentation":
            cv2.putText(
                frame,
                "yaw_deg: N/A",
                (x1, min(y2 + 36, frame.shape[0] - 8)),
                TEXT_FONT,
                TEXT_SCALE,
                INFO_TEXT_COLOR,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

# ============================================================
# Detection 3D Enrichment
# ============================================================

OBJECT_CLOUD_MIN_POINTS = 20
OBJECT_CLOUD_MAX_POINTS = 5000
OBJECT_CLOUD_MAD_SCALE = 3.5
OBJECT_CLOUD_MIN_MAD_M = 0.003


def _build_segmentation_mask(
    frame,
    detection,
):
    depth_image = (
        frame.get("depth_image")
        if isinstance(frame, dict)
        else None
    )

    if not isinstance(depth_image, np.ndarray):
        raise RuntimeError(
            "depth_image unavailable for object point cloud"
        )

    if depth_image.ndim != 2:
        raise RuntimeError(
            "depth_image must have shape [H, W]"
        )

    points = (
        detection.get("segmentation_points")
        or []
    )

    if len(points) < 3:
        raise ValueError(
            "segmentation_points must contain at least 3 points"
        )

    polygon = []

    for point in points:
        if not isinstance(point, dict):
            continue

        if "x" not in point or "y" not in point:
            continue

        try:
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
        except (TypeError, ValueError):
            continue

        polygon.append([x, y])

    if len(polygon) < 3:
        raise ValueError(
            "segmentation polygon does not contain enough valid points"
        )

    height, width = depth_image.shape

    polygon = np.asarray(
        polygon,
        dtype=np.int32,
    )

    polygon[:, 0] = np.clip(
        polygon[:, 0],
        0,
        width - 1,
    )

    polygon[:, 1] = np.clip(
        polygon[:, 1],
        0,
        height - 1,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [polygon.reshape(-1, 1, 2)],
        1,
    )

    if not np.any(mask):
        raise RuntimeError(
            "segmentation mask is empty"
        )

    return mask


def _filter_object_point_cloud(
    points,
):
    """
    Remove invalid depth points and robustly reject Z outliers.

    The Z filtering is intentionally performed in camera coordinates before
    Camera -> Robot Base transformation.  Median/MAD is robust to isolated
    RealSense depth spikes and background leakage near mask boundaries.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(
            "object point cloud must have shape [N, 3]"
        )

    finite = np.all(
        np.isfinite(points),
        axis=1,
    )

    positive_depth = points[:, 2] > 0.0

    points = points[
        finite & positive_depth
    ]

    if len(points) < OBJECT_CLOUD_MIN_POINTS:
        raise RuntimeError(
            "not enough valid segmentation depth points"
        )

    z = points[:, 2]
    z_median = float(np.median(z))

    absolute_deviation = np.abs(
        z - z_median
    )

    mad = float(
        np.median(absolute_deviation)
    )

    threshold = max(
        OBJECT_CLOUD_MIN_MAD_M,
        OBJECT_CLOUD_MAD_SCALE * mad,
    )

    points = points[
        absolute_deviation <= threshold
    ]

    if len(points) < OBJECT_CLOUD_MIN_POINTS:
        raise RuntimeError(
            "not enough object points after depth outlier filtering"
        )

    if len(points) > OBJECT_CLOUD_MAX_POINTS:
        indices = np.linspace(
            0,
            len(points) - 1,
            OBJECT_CLOUD_MAX_POINTS,
            dtype=np.int64,
        )
        points = points[indices]

    return points


def _get_object_cloud_center(
    camera_name,
    frame,
    detection,
):

    mask = _build_segmentation_mask(
        frame=frame,
        detection=detection,
    )

    point_cloud = (
        camera_service
        .get_point_cloud_value(
            camera_name=camera_name,
            frame=frame,
        )
    )

    point_cloud = np.asarray(
        point_cloud,
        dtype=np.float64,
    )

    height, width = mask.shape

    if (
        point_cloud.ndim == 2
        and point_cloud.shape == (height * width, 3)
    ):
        point_cloud = point_cloud.reshape(
            height,
            width,
            3,
        )

    elif (
        point_cloud.ndim == 3
        and point_cloud.shape == (height, width, 3)
    ):
        pass

    else:
        raise RuntimeError(
            "point cloud shape does not match depth image: "
            f"cloud={point_cloud.shape}, depth={(height, width)}"
        )

    object_points = point_cloud[
        mask.astype(bool)
    ]

    object_points = _filter_object_point_cloud(
        object_points
    )

    center_xyz = np.median(
        object_points,
        axis=0,
    )

    return (
        [
            float(center_xyz[0]),
            float(center_xyz[1]),
            float(center_xyz[2]),
        ],
        int(len(object_points)),
        object_points.astype(float).tolist(),
    )


def _get_center_pixel_camera_xyz(
    camera_name,
    frame,
    detection,
):
    """Fallback for detections that do not provide segmentation points."""

    center = (
        detection.get("center")
        or {}
    )

    if "x" not in center or "y" not in center:
        raise ValueError(
            "detection center is unavailable"
        )

    x = center["x"]
    y = center["y"]

    distance = (
        camera_service
        .get_distance_value(
            camera_name=camera_name,
            x=x,
            y=y,
            frame=frame,
        )
    )

    camera_xyz = (
        camera_service
        .deproject_pixel_to_point_value(
            camera_name=camera_name,
            x=x,
            y=y,
            depth=distance,
            frame=frame,
        )
    )

    return (
        [
            float(camera_xyz[0]),
            float(camera_xyz[1]),
            float(camera_xyz[2]),
        ],
        float(distance),
    )


def _enrich_detection_3d(
    camera_name,
    frame,
    detection,
):
    detection = dict(
        detection
    )

    detection.update({
        "distance_m": None,
        "camera_xyz": None,
        "robot_xyz": None,
        "depth_status": "not_supported",
        "coordinate_status": "not_available",
        "depth_method": None,
        "object_point_count": None,
    })

    camera_xyz = None

    # Preferred path: segmentation region + aligned depth point cloud.
    if detection.get("segmentation_points"):
        try:
            (
                camera_xyz,
                object_point_count,
                segmentation_point_cloud,
            ) = _get_object_cloud_center(
                camera_name=camera_name,
                frame=frame,
                detection=detection,
            )

            detection["camera_xyz"] = camera_xyz
            detection["distance_m"] = float(
                camera_xyz[2]
            )
            detection["depth_status"] = "available"
            detection["depth_method"] = (
                "segmentation_point_cloud_median"
            )
            detection["object_point_count"] = (
                object_point_count
            )
            detection["segmentation_point_cloud"] = (
                segmentation_point_cloud
            )

        except NotImplementedError:
            # Camera has no point-cloud capability. Fall through to center pixel.
            camera_xyz = None

        except Exception:
            # Segmentation/cloud failure should not break backward compatibility.
            camera_xyz = None

    # Backward-compatible fallback: original single center-pixel depth method.
    if camera_xyz is None:
        try:
            (
                camera_xyz,
                distance,
            ) = _get_center_pixel_camera_xyz(
                camera_name=camera_name,
                frame=frame,
                detection=detection,
            )

            detection["camera_xyz"] = camera_xyz
            detection["distance_m"] = float(
                distance
            )
            detection["depth_status"] = "available"
            detection["depth_method"] = "center_pixel"

        except NotImplementedError:
            return detection

        except Exception:
            detection["depth_status"] = "invalid_depth"
            return detection

    try:
        robot_xyz = (
            coordinate_service
            .camera_xyz_to_robot_xyz_value(
                camera_name=camera_name,
                camera_xyz=camera_xyz,
            )
        )

        detection["robot_xyz"] = robot_xyz
        detection["coordinate_status"] = "available"

    except Exception:
        detection["coordinate_status"] = (
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
        _normalize_model_name(model_name)

        yolo_result = model_service.yolo.create_detector(
            camera_name=camera_name, frame=frame, draw=False,
        )
        raw_detections = yolo_result["detections"]
        detections = [
            _enrich_yolo_geometry(detection, camera_rgb.shape)
            for detection in raw_detections
        ]

        if include_robot_xyz:
            detections = [
                _enrich_detection_3d(
                    camera_name=camera_name,
                    frame=frame,
                    detection=detection,
                )
                for detection in detections
            ]

        if draw:
            annotated_frame = _draw_yolo_image(camera_rgb, detections)

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
def get_detections_value(
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


def get_camera_image(
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


def get_detection_image(
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
        get_detections_value(
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
# Stream
# ============================================================

def get_camera_stream(
    camera_name,
    interval_sec=0.03,
    quality=85,
):
    """
    原始 RGB MJPEG stream generator。
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
                get_camera_image(
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


def get_detection_stream(
    camera_name,
    model_name=
        "object_detector",
    include_robot_xyz=True,
    interval_sec=0.10,
    quality=85,
):
    """
    含 YOLO 標註結果的 MJPEG stream generator。
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
                get_detection_image(
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
            get_detections_value(
                camera_name=
                    camera_name,
                model_name=
                    model_name,
                draw=False,
                include_robot_xyz=
                    include_robot_xyz,
            )
        )

        return response.success(
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
        return response.error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
        )