
import logging
import math
import threading
import time

import cv2
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
DEFAULT_FRAME_MAX_ATTEMPTS = 3

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
    Intel RealSense D405 camera driver.

    Driver responsibilities:
    - RealSense pipeline lifecycle
    - RGB / Depth stream
    - Frame acquisition and retry
    - Camera fault tracking
    - Pixel depth distance
    - Pixel to camera XYZ deprojection
    - Camera intrinsics
    - Point cloud

    Driver does not depend on config.py.
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

    def __init__(self, serial_number=None):

        if serial_number is not None:
            if not isinstance(serial_number, str) or not serial_number.strip():
                raise ValueError("serial_number 必須是非空字串或 None")
            serial_number = serial_number.strip()

        self.serial_number = serial_number

        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.RLock()

        self._pipeline = None
        self._profile = None
        self._align = None

        self._running = False
        self._fault = False
        self._last_frame_error = None

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
    def _normalize_int(value, name, minimum, maximum):

        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必須是整數") from exc

        if not minimum <= number <= maximum:
            raise ValueError(f"{name} 超出允許範圍：{number}，允許範圍 {minimum} ~ {maximum}")

        return number

    @staticmethod
    def _normalize_bool(value, name):

        if not isinstance(value, bool):
            raise ValueError(f"{name} 必須是 bool")

        return value

    @staticmethod
    def _normalize_align_to(align_to):

        if align_to is None:
            return None

        if not isinstance(align_to, str):
            raise ValueError("align_to 必須是字串或 None")

        align_to = align_to.strip().lower()

        if align_to not in ("color", "depth", "none"):
            raise ValueError("align_to 僅支援：color、depth、none")

        return None if align_to == "none" else align_to

    @staticmethod
    def _normalize_pixel(x, y):

        try:
            x = int(x)
            y = int(y)
        except (TypeError, ValueError) as exc:
            raise ValueError("pixel x、y 必須是整數") from exc

        if x < 0 or y < 0:
            raise ValueError("pixel x、y 不可小於 0")

        return x, y

    @staticmethod
    def _get_depth_frame_from_frame(frame):

        if frame is None:
            return None

        if not isinstance(frame, dict):
            raise ValueError("frame 必須是 dictionary")

        return frame.get("depth_frame")

    # ========================================================
    # Internal State
    # ========================================================

    def _require_running(self):

        if not self._running or self._pipeline is None:
            raise RuntimeError("D405 尚未啟動，請先呼叫 start_camera()")

    def _clear_stream_state(self):

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
    # START CAMERA
    # ========================================================

    def start_camera(self, width=None, height=None, fps=None, enable_color=None, enable_depth=None, align_to=None, frame_timeout_ms=None, image_size=None):
        """
        Start RealSense pipeline.

        Uses driver defaults when parameters are not provided.
        """

        if width is None:
            width = DEFAULT_WIDTH

        if height is None:
            height = DEFAULT_HEIGHT

        if fps is None:
            fps = DEFAULT_FPS

        if enable_color is None:
            enable_color = DEFAULT_ENABLE_COLOR

        if enable_depth is None:
            enable_depth = DEFAULT_ENABLE_DEPTH

        if align_to is None:
            align_to = DEFAULT_ALIGN_TO

        if frame_timeout_ms is None:
            frame_timeout_ms = DEFAULT_FRAME_TIMEOUT_MS

        width = self._normalize_int(width, "width", MIN_WIDTH, MAX_WIDTH)
        height = self._normalize_int(height, "height", MIN_HEIGHT, MAX_HEIGHT)
        fps = self._normalize_int(fps, "fps", MIN_FPS, MAX_FPS)

        enable_color = self._normalize_bool(enable_color, "enable_color")
        enable_depth = self._normalize_bool(enable_depth, "enable_depth")

        align_to = self._normalize_align_to(align_to)
        frame_timeout_ms = self._normalize_int(frame_timeout_ms, "frame_timeout_ms", MIN_FRAME_TIMEOUT_MS, MAX_FRAME_TIMEOUT_MS)

        if not enable_color and not enable_depth:
            raise ValueError("color 與 depth 至少必須啟用一個")

        if align_to == "color" and not enable_color:
            raise ValueError("align_to=color 時必須啟用 color stream")

        if align_to == "depth" and not enable_depth:
            raise ValueError("align_to=depth 時必須啟用 depth stream")

        with self._lifecycle_lock:

            if self._running:
                return True

            pipeline = rs.pipeline()
            rs_config = rs.config()

            if self.serial_number:
                rs_config.enable_device(self.serial_number)

            if enable_depth:
                rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

            if enable_color:
                rs_config.enable_stream(rs.stream.color, width, height, rs.format.yuyv, fps)

            try:
                profile = pipeline.start(rs_config)

            except Exception as exc:
                self._fault = True
                self._last_frame_error = str(exc)
                logger.exception("[D405] pipeline start failed")
                raise

            try:
                depth_scale = None

                if enable_depth:
                    device = profile.get_device()
                    depth_sensor = device.first_depth_sensor()
                    depth_scale = float(depth_sensor.get_depth_scale())

                align = None

                if align_to == "color":
                    align = rs.align(rs.stream.color)
                elif align_to == "depth":
                    align = rs.align(rs.stream.depth)

            except Exception as exc:
                try:
                    pipeline.stop()
                except Exception:
                    logger.exception("[D405] pipeline cleanup failed after start error")

                self._fault = True
                self._last_frame_error = str(exc)
                raise

            self._pipeline = pipeline
            self._profile = profile
            self._align = align

            self._depth_scale = depth_scale

            self._width = width
            self._height = height
            self._fps = fps

            self._enable_color = enable_color
            self._enable_depth = enable_depth
            self._align_to = align_to

            self._frame_timeout_ms = frame_timeout_ms

            self._running = True
            self._fault = False
            self._last_frame_error = None

            logger.info(
                "[D405] camera started serial=%s resolution=%sx%s fps=%s color=%s depth=%s align_to=%s",
                self.serial_number, width, height, fps, enable_color, enable_depth, align_to
            )

        return True

    # ========================================================
    # STOP CAMERA
    # ========================================================

    def stop_camera(self):
        """
        Stop RealSense pipeline.
        """

        with self._lifecycle_lock:

            if not self._running:
                return True

            pipeline = self._pipeline

            try:
                if pipeline is not None:
                    pipeline.stop()

            except Exception as exc:
                self._fault = True
                self._last_frame_error = str(exc)

                logger.exception("[D405] pipeline stop failed")
                raise

            self._clear_stream_state()

            self._fault = False
            self._last_frame_error = None

            logger.info("[D405] camera stopped serial=%s", self.serial_number)

        return True

    # ========================================================
    # GET CAMERA STATUS
    # ========================================================

    def get_camera_status(self):
        """
        Return camera status and diagnostics.
        """

        with self._lifecycle_lock:

            connected = bool(self._running and self._pipeline is not None)

            return {
                "connected": connected,
                "running": self._running,
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
                "color_enabled": self._enable_color,
                "depth_enabled": self._enable_depth,
                "fault": self._fault,

                "serial_number": self.serial_number,
                "align_to": self._align_to,
                "frame_timeout_ms": self._frame_timeout_ms,
                "depth_scale": self._depth_scale,
                "last_frame_error": self._last_frame_error,
                "capabilities": dict(self.DRIVER_METADATA["capabilities"]),
            }

    # ========================================================
    # GET FRAME
    # ========================================================

    def get_frame(self):
        """
        Get synchronized camera frame.

        Returns:
        {
            "frame_id": int | None,
            "host_received_monotonic": float,
            "timestamp": float | None,
            "color_image": numpy.ndarray | None,
            "depth_image": numpy.ndarray | None,
            "depth_frame": opaque object | None,
        }
        """

        with self._lifecycle_lock, self._frame_lock:

            self._require_running()

            try:

                # ====================================================
                # 1. Wait for Frames with Retry
                # ====================================================

                frames = None

                for attempt in range(DEFAULT_FRAME_MAX_ATTEMPTS):

                    try:
                        frames = self._pipeline.wait_for_frames(self._frame_timeout_ms)
                        break

                    except RuntimeError as exc:

                        if "Frame didn't arrive" not in str(exc):
                            raise

                        logger.warning(
                            "[D405] Frame timeout attempt=%s/%s serial=%s error=%s",
                            attempt + 1, DEFAULT_FRAME_MAX_ATTEMPTS, self.serial_number, exc
                        )

                        if attempt == DEFAULT_FRAME_MAX_ATTEMPTS - 1:
                            raise RuntimeError(
                                f"D405 frame acquisition failed after {DEFAULT_FRAME_MAX_ATTEMPTS} attempts: {exc}"
                            ) from exc

                if frames is None:
                    raise RuntimeError("D405 frame acquisition returned no frames")

                host_received_monotonic = time.monotonic()

                # ====================================================
                # 2. Align Frames
                # ====================================================

                if self._align is not None:
                    frames = self._align.process(frames)

                # ====================================================
                # 3. Get Color and Depth Frames
                # ====================================================

                color_frame = frames.get_color_frame() if self._enable_color else None
                depth_frame = frames.get_depth_frame() if self._enable_depth else None

                if self._enable_color and not color_frame:
                    raise RuntimeError("D405 color frame unavailable")

                if self._enable_depth and not depth_frame:
                    raise RuntimeError("D405 depth frame unavailable")

                color_image = None
                depth_image = None

                # ====================================================
                # 4. Decode Color Frame
                # ====================================================

                if color_frame is not None:

                    frame_width = color_frame.get_width()
                    frame_height = color_frame.get_height()
                    frame_format = color_frame.profile.format()

                    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

                    if frame_format == rs.format.yuyv:

                        expected_size = frame_width * frame_height * 2

                        if raw.size != expected_size:
                            raise RuntimeError(
                                f"D405 YUYV buffer size mismatch: expected={expected_size}, actual={raw.size}"
                            )

                        yuyv_image = raw.reshape(frame_height, frame_width, 2)
                        color_image = cv2.cvtColor(yuyv_image, cv2.COLOR_YUV2BGR_YUY2)

                    elif frame_format in (rs.format.bgr8, rs.format.rgb8):

                        expected_size = frame_width * frame_height * 3

                        if raw.size != expected_size:
                            raise RuntimeError(
                                f"D405 RGB/BGR buffer size mismatch: expected={expected_size}, actual={raw.size}"
                            )

                        three_channel_image = raw.reshape(frame_height, frame_width, 3)

                        if frame_format == rs.format.rgb8:
                            color_image = cv2.cvtColor(three_channel_image, cv2.COLOR_RGB2BGR)
                        else:
                            color_image = three_channel_image.copy()

                    else:
                        raise RuntimeError(f"Unsupported D405 color format: {frame_format}")

                    if color_image.shape != (frame_height, frame_width, 3):
                        raise RuntimeError(f"Invalid D405 BGR image shape: {color_image.shape}")

                # ====================================================
                # 5. Decode Depth Frame
                # ====================================================

                if depth_frame is not None:
                    depth_image = np.asanyarray(depth_frame.get_data()).copy()

                # ====================================================
                # 6. Timestamp and Frame ID
                # ====================================================

                timestamp = None
                frame_id = None

                try:
                    timestamp = float(frames.get_timestamp())
                except Exception:
                    pass

                if color_frame is not None:
                    try:
                        timestamp = float(color_frame.get_timestamp())
                        frame_id = int(color_frame.get_frame_number())
                    except Exception:
                        pass

                elif depth_frame is not None:
                    try:
                        timestamp = float(depth_frame.get_timestamp())
                        frame_id = int(depth_frame.get_frame_number())
                    except Exception:
                        pass

                # ====================================================
                # 7. Clear Fault on Success
                # ====================================================

                self._fault = False
                self._last_frame_error = None

                return {
                    "frame_id": frame_id,
                    "host_received_monotonic": host_received_monotonic,
                    "timestamp": timestamp,
                    "color_image": color_image,
                    "depth_image": depth_image,
                    "depth_frame": depth_frame,
                }

            except Exception as exc:

                self._fault = True
                self._last_frame_error = str(exc)

                logger.error("[D405] get_frame failed serial=%s error=%s", self.serial_number, exc)

                raise

    # ========================================================
    # GET DISTANCE
    # ========================================================

    def get_distance(self, x, y, frame=None):
        """
        Get depth distance in meters.
        """

        x, y = self._normalize_pixel(x, y)

        depth_frame = self._get_depth_frame_from_frame(frame)

        if depth_frame is None:
            current_frame = self.get_frame()
            depth_frame = current_frame.get("depth_frame")

        if depth_frame is None:
            raise RuntimeError("depth frame unavailable")

        width = depth_frame.get_width()
        height = depth_frame.get_height()

        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"pixel ({x}, {y}) 超出 depth image 範圍：{width}x{height}")

        distance = float(depth_frame.get_distance(x, y))

        if not math.isfinite(distance) or distance <= 0:
            raise RuntimeError("depth distance 無效")

        return distance

    # ========================================================
    # DEPROJECT PIXEL TO POINT
    # ========================================================

    def deproject_pixel_to_point(self, x, y, depth=None, frame=None):
        """
        Convert pixel and depth to camera XYZ.

        Returns:
            [x, y, z]

        Unit:
            meter
        """

        x, y = self._normalize_pixel(x, y)

        depth_frame = self._get_depth_frame_from_frame(frame)

        if depth_frame is None:
            current_frame = self.get_frame()
            depth_frame = current_frame.get("depth_frame")

        if depth_frame is None:
            raise RuntimeError("depth frame unavailable")

        width = depth_frame.get_width()
        height = depth_frame.get_height()

        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"pixel ({x}, {y}) 超出 depth image 範圍：{width}x{height}")

        if depth is None:
            depth = float(depth_frame.get_distance(x, y))
        else:
            try:
                depth = float(depth)
            except (TypeError, ValueError) as exc:
                raise ValueError("depth 必須是數值") from exc

        if not math.isfinite(depth) or depth <= 0:
            raise ValueError("depth 必須是大於 0 的有限數值")

        profile = depth_frame.profile.as_video_stream_profile()
        intrinsics = profile.get_intrinsics()

        point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], depth)

        return [
            float(point[0]),
            float(point[1]),
            float(point[2]),
        ]

    # ========================================================
    # GET INTRINSICS
    # ========================================================

    def get_intrinsics(self, frame=None):
        """
        Get camera intrinsics.
        """

        with self._lifecycle_lock:

            self._require_running()

            depth_frame = self._get_depth_frame_from_frame(frame)
            profile = None

            if self._align_to == "color" and depth_frame is not None:
                profile = depth_frame.profile.as_video_stream_profile()

            elif self._enable_color:
                profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()

            elif depth_frame is not None:
                profile = depth_frame.profile.as_video_stream_profile()

            elif self._enable_depth:
                profile = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()

            if profile is None:
                raise RuntimeError("camera intrinsics unavailable")

            intrinsics = profile.get_intrinsics()

            return {
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.ppx),
                "cy": float(intrinsics.ppy),
                "distortion_model": str(intrinsics.model),
                "coeffs": [float(value) for value in intrinsics.coeffs],
            }

    # ========================================================
    # GET POINT CLOUD
    # ========================================================

    def get_point_cloud(self, frame=None):
        """
        Get point cloud.

        Returns:
            numpy.ndarray [N, 3]
            dtype: float32
            unit: meter
        """

        depth_frame = self._get_depth_frame_from_frame(frame)

        if depth_frame is None:
            current_frame = self.get_frame()
            depth_frame = current_frame.get("depth_frame")

        if depth_frame is None:
            raise RuntimeError("depth frame unavailable")

        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)

        vertices = np.asanyarray(points.get_vertices())

        xyz = vertices.view(np.float32).reshape(-1, 3).copy()

        return xyz
