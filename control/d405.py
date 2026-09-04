import logging
import math
import threading
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)

# ============================================================
# Driver Defaults
# ============================================================

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_ENABLE_COLOR = True
DEFAULT_ENABLE_DEPTH = True
DEFAULT_ALIGN_TO = "color"
DEFAULT_FRAME_TIMEOUT_MS = 3000


# ============================================================
# Hard Limits
# ============================================================

MIN_WIDTH = 1
MAX_WIDTH = 4096
MIN_HEIGHT = 1
MAX_HEIGHT = 2160
MIN_FPS = 1
MAX_FPS = 120
MIN_FRAME_TIMEOUT_MS = 100
MAX_FRAME_TIMEOUT_MS = 30000


# ============================================================
# D405 Driver
# ============================================================

class D405Driver:
    """
    Intel RealSense D405 camera driver。

    Driver 責任：
    - RealSense pipeline lifecycle
    - RGB / Depth stream
    - frame acquisition
    - pixel depth distance
    - pixel -> camera XYZ deprojection
    - camera intrinsics
    - point cloud

    Driver 不依賴 config.py。
    """

    DRIVER_METADATA = {
        "name": "d405",
        "manufacturer": "Intel RealSense",
        "model": "D405",

        "capabilities": {
            "color": True,
            "depth": True,
            "deprojection": True,
            "intrinsics": True,
            "point_cloud": True,
        },
    }

    def __init__(
        self,
        serial_number=None,
    ):
        if serial_number is not None:
            if (
                not isinstance(serial_number, str)
                or not serial_number.strip()
            ):
                raise ValueError(
                    "serial_number 必須是非空字串或 None"
                )

            serial_number = (
                serial_number
                .strip()
            )

        self.serial_number = (
            serial_number
        )

        self._lifecycle_lock = (
            threading.RLock()
        )

        self._frame_lock = (
            threading.RLock()
        )

        self._pipeline = None
        self._profile = None
        self._align = None

        self._running = False

        self._depth_scale = None

        self._width = None
        self._height = None
        self._fps = None

        self._enable_color = None
        self._enable_depth = None
        self._align_to = None

        self._frame_timeout_ms = None


    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _normalize_int(
        value,
        name,
        minimum,
        maximum,
    ):
        try:
            number = int(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} 必須是整數"
            ) from exc

        if not (
            minimum
            <= number
            <= maximum
        ):
            raise ValueError(
                f"{name} 超出允許範圍："
                f"{number}，"
                f"允許範圍 "
                f"{minimum} ~ {maximum}"
            )

        return number


    @staticmethod
    def _normalize_bool(
        value,
        name,
    ):
        if not isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{name} 必須是 bool"
            )

        return value


    @staticmethod
    def _normalize_align_to(
        align_to,
    ):
        if align_to is None:
            return None

        if not isinstance(
            align_to,
            str,
        ):
            raise ValueError(
                "align_to 必須是字串或 None"
            )

        align_to = (
            align_to
            .strip()
            .lower()
        )

        allowed = {
            "color",
            "depth",
            "none",
        }

        if align_to not in allowed:
            raise ValueError(
                "align_to 僅支援："
                "color、depth、none"
            )

        if align_to == "none":
            return None

        return align_to


    @staticmethod
    def _normalize_pixel(
        x,
        y,
    ):
        try:
            x = int(x)
            y = int(y)

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "pixel x、y 必須是整數"
            ) from exc

        if x < 0 or y < 0:
            raise ValueError(
                "pixel x、y 不可小於 0"
            )

        return x, y


    @staticmethod
    def _get_depth_frame_from_frame(
        frame,
    ):
        if frame is None:
            return None

        if not isinstance(
            frame,
            dict,
        ):
            raise ValueError(
                "frame 必須是 dictionary"
            )

        return frame.get(
            "depth_frame"
        )


    # ========================================================
    # Lifecycle
    # ========================================================

    def start_camera(
        self,
        width=None,
        height=None,
        fps=None,
        enable_color=None,
        enable_depth=None,
        align_to=None,
        frame_timeout_ms=None,
    ):
        """
        啟動 RealSense pipeline。

        上層沒有傳入參數時使用 Driver fallback defaults。
        """

        if width is None:
            width = DEFAULT_WIDTH

        if height is None:
            height = DEFAULT_HEIGHT

        if fps is None:
            fps = DEFAULT_FPS

        if enable_color is None:
            enable_color = (
                DEFAULT_ENABLE_COLOR
            )

        if enable_depth is None:
            enable_depth = (
                DEFAULT_ENABLE_DEPTH
            )

        if align_to is None:
            align_to = (
                DEFAULT_ALIGN_TO
            )

        if frame_timeout_ms is None:
            frame_timeout_ms = (
                DEFAULT_FRAME_TIMEOUT_MS
            )

        width = self._normalize_int(
            width,
            "width",
            MIN_WIDTH,
            MAX_WIDTH,
        )

        height = self._normalize_int(
            height,
            "height",
            MIN_HEIGHT,
            MAX_HEIGHT,
        )

        fps = self._normalize_int(
            fps,
            "fps",
            MIN_FPS,
            MAX_FPS,
        )

        enable_color = (
            self._normalize_bool(
                enable_color,
                "enable_color",
            )
        )

        enable_depth = (
            self._normalize_bool(
                enable_depth,
                "enable_depth",
            )
        )

        align_to = (
            self._normalize_align_to(
                align_to
            )
        )

        frame_timeout_ms = (
            self._normalize_int(
                frame_timeout_ms,
                "frame_timeout_ms",
                MIN_FRAME_TIMEOUT_MS,
                MAX_FRAME_TIMEOUT_MS,
            )
        )

        if (
            not enable_color
            and not enable_depth
        ):
            raise ValueError(
                "color 與 depth "
                "至少必須啟用一個"
            )

        if (
            align_to == "color"
            and not enable_color
        ):
            raise ValueError(
                "align_to=color 時 "
                "必須啟用 color stream"
            )

        if (
            align_to == "depth"
            and not enable_depth
        ):
            raise ValueError(
                "align_to=depth 時 "
                "必須啟用 depth stream"
            )

        with self._lifecycle_lock:
            if self._running:
                return True

            pipeline = rs.pipeline()
            rs_config = rs.config()

            if self.serial_number:
                rs_config.enable_device(
                    self.serial_number
                )

            if enable_depth:
                rs_config.enable_stream(
                    rs.stream.depth,
                    width,
                    height,
                    rs.format.z16,
                    fps,
                )

            if enable_color:
                rs_config.enable_stream(
                    rs.stream.color,
                    width,
                    height,
                    rs.format.bgr8,
                    fps,
                )

            try:
                profile = (
                    pipeline.start(
                        rs_config
                    )
                )

            except Exception:
                logger.exception(
                    "[D405] pipeline start failed"
                )
                raise

            try:
                depth_scale = None

                if enable_depth:
                    device = (
                        profile
                        .get_device()
                    )

                    depth_sensor = (
                        device
                        .first_depth_sensor()
                    )

                    depth_scale = float(
                        depth_sensor
                        .get_depth_scale()
                    )

                align = None

                if align_to == "color":
                    align = rs.align(
                        rs.stream.color
                    )

                elif align_to == "depth":
                    align = rs.align(
                        rs.stream.depth
                    )

            except Exception:
                try:
                    pipeline.stop()
                except Exception:
                    pass

                raise

            self._pipeline = pipeline
            self._profile = profile
            self._align = align

            self._depth_scale = (
                depth_scale
            )

            self._width = width
            self._height = height
            self._fps = fps

            self._enable_color = (
                enable_color
            )

            self._enable_depth = (
                enable_depth
            )

            self._align_to = (
                align_to
            )

            self._frame_timeout_ms = (
                frame_timeout_ms
            )

            self._running = True

            logger.info(
                "[D405] camera started "
                "serial=%s "
                "resolution=%sx%s "
                "fps=%s "
                "color=%s "
                "depth=%s "
                "align_to=%s",
                self.serial_number,
                width,
                height,
                fps,
                enable_color,
                enable_depth,
                align_to,
            )

        return True


    def stop_camera(
        self,
    ):
        """
        停止 RealSense pipeline。
        """

        with self._lifecycle_lock:
            pipeline = (
                self._pipeline
            )

            if not self._running:
                return True

            self._running = False

            self._pipeline = None
            self._profile = None
            self._align = None

            if pipeline is not None:
                try:
                    pipeline.stop()

                except Exception:
                    logger.exception(
                        "[D405] pipeline stop failed"
                    )
                    raise

            logger.info(
                "[D405] camera stopped "
                "serial=%s",
                self.serial_number,
            )

        return True


    # ========================================================
    # Status
    # ========================================================

    def get_camera_status(
        self,
    ):
        """
        取得 Driver 狀態。

        通用 camera_service contract：
            connected
            running
            width
            height
            fps
            color_enabled
            depth_enabled
            fault

        其他欄位為 D405-specific diagnostics。
        """

        with self._lifecycle_lock:
            connected = bool(
                self._running
                and self._pipeline is not None
            )

            return {
                # Generic camera-service contract
                "connected":
                    connected,

                "running":
                    self._running,

                "width":
                    self._width,

                "height":
                    self._height,

                "fps":
                    self._fps,

                "color_enabled":
                    self._enable_color,

                "depth_enabled":
                    self._enable_depth,

                "fault":
                    False,

                # D405-specific diagnostics
                "serial_number":
                    self.serial_number,

                "align_to":
                    self._align_to,

                "frame_timeout_ms":
                    self._frame_timeout_ms,

                "depth_scale":
                    self._depth_scale,

                "capabilities":
                    dict(
                        self.DRIVER_METADATA[
                            "capabilities"
                        ]
                    ),
            }


    # ========================================================
    # Frame
    # ========================================================

    def _require_running(
        self,
    ):
        if (
            not self._running
            or self._pipeline is None
        ):
            raise RuntimeError(
                "D405 尚未啟動，"
                "請先呼叫 start_camera()"
            )


    def get_frame(
        self,
    ):
        """
        取得一組同步 Camera frame。

        通用 camera_service contract：
            {
                "timestamp": float | None,
                "color_image": numpy.ndarray | None,
                "depth_image": numpy.ndarray | None,
            }

        額外保留：
            "depth_frame"

        depth_frame 是 D405 Driver 內部使用的 opaque object，
        供 get_distance()、deproject_pixel_to_point()、
        get_intrinsics()、get_point_cloud() 使用同一組 frame。

        上層 service 不應直接依賴 pyrealsense2 型別。
        """

        with self._frame_lock:
            self._require_running()

            frames = (
                self._pipeline
                .wait_for_frames(
                    self._frame_timeout_ms
                )
            )

            if self._align is not None:
                frames = (
                    self._align
                    .process(
                        frames
                    )
                )

            color_frame = (
                frames.get_color_frame()
                if self._enable_color
                else None
            )

            depth_frame = (
                frames.get_depth_frame()
                if self._enable_depth
                else None
            )

            if (
                self._enable_color
                and not color_frame
            ):
                raise RuntimeError(
                    "D405 color frame unavailable"
                )

            if (
                self._enable_depth
                and not depth_frame
            ):
                raise RuntimeError(
                    "D405 depth frame unavailable"
                )

            color_image = None
            depth_image = None

            if color_frame is not None:
                color_image = (
                    np.asanyarray(
                        color_frame
                        .get_data()
                    )
                )

            if depth_frame is not None:
                depth_image = (
                    np.asanyarray(
                        depth_frame
                        .get_data()
                    )
                )

            timestamp = None

            try:
                timestamp = float(
                    frames.get_timestamp()
                )
            except Exception:
                timestamp = None

            return {
                # Generic camera-service contract
                "timestamp":
                    timestamp,

                "color_image":
                    color_image,

                "depth_image":
                    depth_image,

                # D405-specific opaque object
                "depth_frame":
                    depth_frame,
            }


    # ========================================================
    # Depth
    # ========================================================

    def get_distance(
        self,
        x,
        y,
        frame=None,
    ):
        """
        取得指定 pixel 的深度，單位 meter。

        frame:
            建議傳入先前 get_frame() 的同一組 frame，
            避免 RGB 與 Depth 來自不同時間點。
        """

        x, y = (
            self._normalize_pixel(
                x,
                y,
            )
        )

        depth_frame = (
            self._get_depth_frame_from_frame(
                frame
            )
        )

        if depth_frame is None:
            frame = self.get_frame()

            depth_frame = (
                frame.get(
                    "depth_frame"
                )
            )

        if depth_frame is None:
            raise RuntimeError(
                "depth frame unavailable"
            )

        width = (
            depth_frame
            .get_width()
        )

        height = (
            depth_frame
            .get_height()
        )

        if not (
            0 <= x < width
            and
            0 <= y < height
        ):
            raise ValueError(
                f"pixel ({x}, {y}) "
                f"超出 depth image 範圍："
                f"{width}x{height}"
            )

        distance = float(
            depth_frame
            .get_distance(
                x,
                y,
            )
        )

        if not math.isfinite(
            distance
        ):
            raise RuntimeError(
                "depth distance 非有限數值"
            )

        if distance <= 0:
            raise RuntimeError(
                "depth distance 無效"
            )

        return distance


    def deproject_pixel_to_point(
        self,
        x,
        y,
        depth=None,
        frame=None,
    ):
        """
        將 image pixel + depth 轉成 Camera Coordinate XYZ。

        return:
            [x, y, z]

        單位：
            meter
        """

        x, y = (
            self._normalize_pixel(
                x,
                y,
            )
        )

        depth_frame = (
            self._get_depth_frame_from_frame(
                frame
            )
        )

        if depth_frame is None:
            frame = self.get_frame()

            depth_frame = (
                frame.get(
                    "depth_frame"
                )
            )

        if depth_frame is None:
            raise RuntimeError(
                "depth frame unavailable"
            )

        width = (
            depth_frame
            .get_width()
        )

        height = (
            depth_frame
            .get_height()
        )

        if not (
            0 <= x < width
            and
            0 <= y < height
        ):
            raise ValueError(
                f"pixel ({x}, {y}) "
                f"超出 depth image 範圍："
                f"{width}x{height}"
            )

        if depth is None:
            depth = float(
                depth_frame
                .get_distance(
                    x,
                    y,
                )
            )

        else:
            try:
                depth = float(
                    depth
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    "depth 必須是數值"
                ) from exc

        if (
            not math.isfinite(depth)
            or depth <= 0
        ):
            raise ValueError(
                "depth 必須是大於 0 的有限數值"
            )

        profile = (
            depth_frame
            .profile
            .as_video_stream_profile()
        )

        intrinsics = (
            profile
            .get_intrinsics()
        )

        point = (
            rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [
                    float(x),
                    float(y),
                ],
                depth,
            )
        )

        return [
            float(point[0]),
            float(point[1]),
            float(point[2]),
        ]


    # ========================================================
    # Intrinsics
    # ========================================================

    def get_intrinsics(
        self,
        frame=None,
    ):
        """
        取得目前影像座標系所使用的 Camera Intrinsics。

        return:
            {
                "width": int,
                "height": int,
                "fx": float,
                "fy": float,
                "cx": float,
                "cy": float,
                "distortion_model": str,
                "coeffs": list[float],
            }

        若 depth 已 align 到 color，
        優先使用 aligned depth frame 的 profile，
        確保 intrinsics 與 YOLO 的 color pixel 對齊。
        """

        self._require_running()

        depth_frame = (
            self._get_depth_frame_from_frame(
                frame
            )
        )

        profile = None

        if (
            self._align_to == "color"
            and depth_frame is not None
        ):
            profile = (
                depth_frame
                .profile
                .as_video_stream_profile()
            )

        elif self._enable_color:
            if self._profile is None:
                raise RuntimeError(
                    "camera profile unavailable"
                )

            profile = (
                self._profile
                .get_stream(
                    rs.stream.color
                )
                .as_video_stream_profile()
            )

        elif depth_frame is not None:
            profile = (
                depth_frame
                .profile
                .as_video_stream_profile()
            )

        elif self._enable_depth:
            if self._profile is None:
                raise RuntimeError(
                    "camera profile unavailable"
                )

            profile = (
                self._profile
                .get_stream(
                    rs.stream.depth
                )
                .as_video_stream_profile()
            )

        if profile is None:
            raise RuntimeError(
                "camera intrinsics unavailable"
            )

        intrinsics = (
            profile
            .get_intrinsics()
        )

        return {
            "width":
                int(
                    intrinsics.width
                ),

            "height":
                int(
                    intrinsics.height
                ),

            "fx":
                float(
                    intrinsics.fx
                ),

            "fy":
                float(
                    intrinsics.fy
                ),

            "cx":
                float(
                    intrinsics.ppx
                ),

            "cy":
                float(
                    intrinsics.ppy
                ),

            "distortion_model":
                str(
                    intrinsics.model
                ),

            "coeffs": [
                float(value)
                for value
                in intrinsics.coeffs
            ],
        }


    # ========================================================
    # Point Cloud
    # ========================================================

    def get_point_cloud(
        self,
        frame=None,
    ):
        """
        取得 Point Cloud。

        return:
            numpy.ndarray
            shape: [N, 3]
            dtype: float32
            unit: meter

        若傳入 frame，
        使用該次 get_frame() 的 depth_frame，
        讓 RGB / Depth / Point Cloud 使用同一組 frame。
        """

        self._require_running()

        depth_frame = (
            self._get_depth_frame_from_frame(
                frame
            )
        )

        if depth_frame is None:
            current_frame = (
                self.get_frame()
            )

            depth_frame = (
                current_frame.get(
                    "depth_frame"
                )
            )

        if depth_frame is None:
            raise RuntimeError(
                "depth frame unavailable"
            )

        pc = rs.pointcloud()

        points = (
            pc.calculate(
                depth_frame
            )
        )

        vertices = (
            np.asanyarray(
                points.get_vertices()
            )
        )

        xyz = (
            vertices
            .view(
                np.float32
            )
            .reshape(
                -1,
                3,
            )
            .copy()
        )

        return xyz