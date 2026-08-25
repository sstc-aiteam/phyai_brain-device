import cv2
import math
import numpy as np
from ultralytics import YOLO


# ============================================================
# YOLO Defaults
# ============================================================

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.5
DEFAULT_DEVICE = "cpu"


# ============================================================
# Drawing Settings
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


class YoloDetector:

    def __init__(
        self,
        model_path,
        imgsz=None,
        conf=None,
        device=None,
    ):
        if not isinstance(
            model_path,
            str,
        ) or not model_path.strip():
            raise ValueError(
                "model_path 必須是非空字串"
            )

        model_path = (
            model_path
            .strip()
        )

        if imgsz is None:
            imgsz = DEFAULT_IMGSZ

        if conf is None:
            conf = DEFAULT_CONF

        if device is None:
            device = DEFAULT_DEVICE

        try:
            imgsz = int(
                imgsz
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "imgsz 必須是整數"
            ) from exc

        if imgsz <= 0:
            raise ValueError(
                "imgsz 必須大於 0"
            )

        try:
            conf = float(
                conf
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "conf 必須是數值"
            ) from exc

        if not (
            0.0
            <= conf
            <= 1.0
        ):
            raise ValueError(
                "conf 必須介於 0.0 ~ 1.0"
            )

        if not isinstance(
            device,
            str,
        ) or not device.strip():
            raise ValueError(
                "device 必須是非空字串"
            )

        device = (
            device
            .strip()
        )

        self.model_path = (
            model_path
        )

        self.imgsz = imgsz
        self.conf = conf
        self.device = device

        self.model = YOLO(
            self.model_path
        )

        if self.device != "cpu":
            self.model.to(
                self.device
            )

    def predict(self, color_image, draw=True):
        """
        對單張 color_image 做 YOLO 推論。

        只處理 2D vision inference，不處理 depth / distance / 3D coordinate。

        Returns:
            annotated_frame, detections

            Detection model 回傳範例：
            [
                {
                    "task": "detection",
                    "class_id": 0,
                    "class_name": "trash",
                    "confidence": 0.91,
                    "box": {
                        "x1": 100,
                        "y1": 120,
                        "x2": 220,
                        "y2": 300
                    },
                    "center": {
                        "x": 160,
                        "y": 210
                    },
                    "segmentation_points": [],
                    "yaw_deg": None
                }
            ]

            Segmentation model 回傳範例：
            [
                {
                    "task": "segmentation",
                    "class_id": 0,
                    "class_name": "trash",
                    "confidence": 0.91,
                    "box": {
                        "x1": 100,
                        "y1": 120,
                        "x2": 220,
                        "y2": 300
                    },
                    "center": {
                        "x": 158,
                        "y": 214
                    },
                    "segmentation_points": [
                        {"x": 120, "y": 130},
                        {"x": 135, "y": 125},
                        {"x": 180, "y": 140}
                    ],
                    "yaw_deg": 28.4
                }
            ]
        """

        if color_image is None:
            return None, []

        frame = color_image.copy()

        results = self.model.predict(
            source=color_image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        if not results:
            return frame, []

        result = results[0]

        annotated_frame, detections = self._parse_result(
            frame=frame,
            result=result,
            draw=draw,
        )

        return annotated_frame, detections

    def _parse_result(self, frame, result, draw=True):
        detections = []

        if result.boxes is None:
            return frame, detections

        boxes = result.boxes

        has_masks = (
            hasattr(result, "masks")
            and result.masks is not None
            and hasattr(result.masks, "xy")
            and result.masks.xy is not None
        )

        mask_polygons = result.masks.xy if has_masks else []

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(0, min(x2, frame.shape[1] - 1))
            y2 = max(0, min(y2, frame.shape[0] - 1))

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.model.names.get(cls_id, str(cls_id))

            task = "detection"
            segmentation_points = []

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if has_masks and index < len(mask_polygons):
                mask_points = mask_polygons[index]

                if mask_points is not None and len(mask_points) > 0:
                    task = "segmentation"

                    segmentation_points = self._convert_mask_points(
                        mask_points=mask_points,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )

                    mask_center = self._get_points_center(segmentation_points)

                    if mask_center is not None:
                        center_x, center_y = mask_center

            yaw_deg = None
            pca_debug = None
            orientation_debug = None

            if task == "segmentation" and segmentation_points:
                orientation_debug = self._calc_orientation_debug(
                    points=segmentation_points,
                    frame_shape=frame.shape,
                    class_name=class_name,
                )
                pca_debug = orientation_debug.get("pca_debug")
                yaw_deg = orientation_debug.get("yaw_deg")

            detection = {
                "task": task,
                "class_id": int(cls_id),
                "class_name": str(class_name),
                "confidence": float(conf),
                "box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
                "center": {
                    "x": int(center_x),
                    "y": int(center_y),
                },
                "segmentation_points": segmentation_points,
                "yaw_deg": yaw_deg,
                "pca_status": None if pca_debug is None else pca_debug.get("status"),
                "pca_axis_ratio": None if pca_debug is None else pca_debug.get("axis_ratio"),
                "pca_debug": pca_debug,
                "orientation_debug": orientation_debug,
            }

            detections.append(detection)

            if draw:
                self._draw_detection(
                    frame=frame,
                    detection=detection,
                )

        return frame, detections

    def _calc_pca_debug_from_points(self, points):
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

    def _calc_orientation_debug(self, points, frame_shape, class_name):
        """
        用清理後的 mask 輪廓計算 orientation。
        優先使用 minAreaRect 長邊方向，並保留 PCA debug 供比對。
        bottle_alcohol_spray 只用下半部瓶身，降低噴頭影響。
        """

        frame_height, frame_width = frame_shape[:2]
        contour, contour_status, roi_mode = self._build_orientation_contour(
            points=points,
            frame_width=frame_width,
            frame_height=frame_height,
            class_name=class_name,
        )

        pca_debug = self._calc_pca_debug_from_points(points)

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

    def _build_orientation_contour(self, points, frame_width, frame_height, class_name):
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

    def _convert_mask_points(self, mask_points, frame_width, frame_height):
        """
        將 YOLO segmentation mask polygon 轉成 JSON-friendly 格式。
        result.masks.xy[index] 通常是 Nx2 numpy array，座標為 image pixel。
        """

        points = []

        for point in mask_points:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))

            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))

            points.append({
                "x": x,
                "y": y,
            })

        return points

    def _get_points_center(self, points):
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

    def _draw_detection(self, frame, detection):
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
            self._draw_segmentation(
                frame=frame,
                points=detection["segmentation_points"],
            )
            self._draw_orientation_axis(
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

        self._draw_label_with_background(
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

    def _draw_segmentation(self, frame, points):
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

    def _draw_orientation_axis(self, frame, orientation_debug):
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

    def _draw_label_with_background(self, frame, text, x, y, bg_color):
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


# ============================================================
# Factory
# ============================================================

def create_detector(
    model_path,
    imgsz=None,
    conf=None,
    device=None,
):
    """
    建立 YoloDetector instance。

    建議由 vision_service 將
    config.YOLO_MODELS[...]["kwargs"]
    傳入此函式。
    """

    return YoloDetector(
        model_path=model_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )