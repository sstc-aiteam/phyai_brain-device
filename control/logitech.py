import logging
import threading
import time

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ============================================================
# Driver Defaults
# ============================================================

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

DEFAULT_FRAME_TIMEOUT_MS = 3000

DEFAULT_FOURCC = "MJPG"


# ============================================================
# Hard Limits
# ============================================================

MIN_WIDTH = 1
MAX_WIDTH = 7680

MIN_HEIGHT = 1
MAX_HEIGHT = 4320

MIN_FPS = 1
MAX_FPS = 240

MIN_FRAME_TIMEOUT_MS = 100
MAX_FRAME_TIMEOUT_MS = 30000


# ============================================================
# USB Camera Driver
# ============================================================

class LogitechDriver:
    """
    Generic USB / UVC Camera Driver。

    Driver 責任：
    - OpenCV VideoCapture lifecycle
    - RGB frame acquisition
    - USB / UVC camera status

    不支援：
    - Depth
    - Pixel -> Camera XYZ deprojection
    - Point cloud

    Driver 不依賴 config.py。
    """

    DRIVER_METADATA = {
        "name": "usb_camera",
        "manufacturer": "Generic",
        "model": "UVC Camera",

        "capabilities": {
            "color": True,
            "depth": False,
            "deprojection": False,
            "intrinsics": False,
            "point_cloud": False,
        },
    }

    def __init__(
        self,
        device_index=0,
        device_path=None,
        fourcc=DEFAULT_FOURCC,
    ):
        """
        Args:
            device_index:
                OpenCV camera index。
                例如：
                    0 -> /dev/video0
                    6 -> /dev/video6

            device_path:
                可直接指定 Linux device path。
                例如：
                    /dev/video6

                如果有設定 device_path，
                優先使用 device_path。

            fourcc:
                OpenCV FOURCC。
                預設使用 MJPG，
                通常較適合 USB camera 的
                1280x720 / 1920x1080 @ 30 FPS。
        """

        if device_path is not None:
            if (
                not isinstance(device_path, str)
                or not device_path.strip()
            ):
                raise ValueError(
                    "device_path 必須是非空字串或 None"
                )

            device_path = (
                device_path
                .strip()
            )

        try:
            device_index = int(
                device_index
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "device_index 必須是整數"
            ) from exc

        if device_index < 0:
            raise ValueError(
                "device_index 不可小於 0"
            )

        if fourcc is not None:
            if (
                not isinstance(fourcc, str)
                or len(fourcc) != 4
            ):
                raise ValueError(
                    "fourcc 必須是 4 字元字串或 None"
                )

            fourcc = fourcc.upper()

        self.device_index = (
            device_index
        )

        self.device_path = (
            device_path
        )

        self.fourcc = (
            fourcc
        )

        self._lifecycle_lock = (
            threading.RLock()
        )

        self._frame_lock = (
            threading.RLock()
        )

        self._capture = None
        self._running = False

        self._width = None
        self._height = None
        self._fps = None

        self._frame_timeout_ms = None

        self._actual_width = None
        self._actual_height = None
        self._actual_fps = None

        self._last_frame_timestamp = None


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


    # ========================================================
    # Device
    # ========================================================

    def _get_device(
        self,
    ):
        """
        取得 OpenCV VideoCapture 使用的 device。

        device_path 優先於 device_index。
        """

        if self.device_path is not None:
            return self.device_path

        return self.device_index


    def _open_capture(
        self,
    ):
        """
        開啟 VideoCapture。

        Linux 優先使用 V4L2 backend。
        如果 V4L2 開啟失敗，
        再 fallback 到 OpenCV default backend。
        """

        device = (
            self._get_device()
        )

        capture = cv2.VideoCapture(
            device,
            cv2.CAP_V4L2,
        )

        if capture.isOpened():
            return capture

        capture.release()

        capture = cv2.VideoCapture(
            device
        )

        if capture.isOpened():
            return capture

        capture.release()

        raise RuntimeError(
            f"無法開啟 USB camera: "
            f"{device}"
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
        啟動 USB camera。

        為了與 D405Driver / camera_service
        使用相同 interface，
        保留：
            enable_color
            enable_depth
            align_to

        USB RGB camera 不支援 depth / align。
        """

        if width is None:
            width = DEFAULT_WIDTH

        if height is None:
            height = DEFAULT_HEIGHT

        if fps is None:
            fps = DEFAULT_FPS

        if frame_timeout_ms is None:
            frame_timeout_ms = (
                DEFAULT_FRAME_TIMEOUT_MS
            )

        if enable_color is None:
            enable_color = True

        if enable_depth is None:
            enable_depth = False

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

        frame_timeout_ms = (
            self._normalize_int(
                frame_timeout_ms,
                "frame_timeout_ms",
                MIN_FRAME_TIMEOUT_MS,
                MAX_FRAME_TIMEOUT_MS,
            )
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

        if not enable_color:
            raise ValueError(
                "USB camera 必須啟用 color stream"
            )

        if enable_depth:
            raise NotImplementedError(
                "USB camera 不支援 depth stream"
            )

        if align_to not in (
            None,
            "none",
        ):
            raise NotImplementedError(
                "USB camera 不支援 stream alignment"
            )

        with self._lifecycle_lock:
            if self._running:
                return True

            capture = None

            try:
                capture = (
                    self._open_capture()
                )

                # ============================================
                # Codec
                # ============================================

                if self.fourcc is not None:
                    fourcc_value = (
                        cv2.VideoWriter_fourcc(
                            *self.fourcc
                        )
                    )

                    capture.set(
                        cv2.CAP_PROP_FOURCC,
                        fourcc_value,
                    )

                # ============================================
                # Requested Stream Settings
                # ============================================

                capture.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    float(width),
                )

                capture.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    float(height),
                )

                capture.set(
                    cv2.CAP_PROP_FPS,
                    float(fps),
                )

                # Buffer 越小越適合 realtime vision。
                capture.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1,
                )

                # ============================================
                # Read Actual Settings
                # ============================================

                actual_width = int(
                    capture.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                actual_height = int(
                    capture.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                actual_fps = float(
                    capture.get(
                        cv2.CAP_PROP_FPS
                    )
                )

                # ============================================
                # Test Frame
                # ============================================

                ret, frame = (
                    capture.read()
                )

                if (
                    not ret
                    or frame is None
                ):
                    raise RuntimeError(
                        "USB camera 已開啟，"
                        "但無法取得影像"
                    )

                if not isinstance(
                    frame,
                    np.ndarray,
                ):
                    raise RuntimeError(
                        "USB camera frame "
                        "不是 numpy.ndarray"
                    )

                if (
                    frame.ndim != 3
                    or frame.shape[2] != 3
                ):
                    raise RuntimeError(
                        "USB camera frame "
                        "必須為 BGR [H, W, 3]"
                    )

                # ============================================
                # Commit State
                # ============================================

                self._capture = capture

                self._width = width
                self._height = height
                self._fps = fps

                self._actual_width = (
                    actual_width
                )

                self._actual_height = (
                    actual_height
                )

                self._actual_fps = (
                    actual_fps
                )

                self._frame_timeout_ms = (
                    frame_timeout_ms
                )

                self._last_frame_timestamp = (
                    time.time()
                )

                self._running = True

                logger.info(
                    "[USB_CAMERA] started "
                    "device=%s "
                    "requested=%sx%s@%s "
                    "actual=%sx%s@%.2f "
                    "fourcc=%s",
                    self._get_device(),
                    width,
                    height,
                    fps,
                    actual_width,
                    actual_height,
                    actual_fps,
                    self.fourcc,
                )

            except Exception:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass

                logger.exception(
                    "[USB_CAMERA] camera start failed "
                    "device=%s",
                    self._get_device(),
                )

                raise

        return True


    def stop_camera(
        self,
    ):
        """
        停止 USB camera。
        """

        with self._lifecycle_lock:
            capture = (
                self._capture
            )

            if not self._running:
                return True

            self._running = False
            self._capture = None

            if capture is not None:
                try:
                    capture.release()

                except Exception:
                    logger.exception(
                        "[USB_CAMERA] "
                        "camera release failed"
                    )
                    raise

            logger.info(
                "[USB_CAMERA] camera stopped "
                "device=%s",
                self._get_device(),
            )

        return True


    # ========================================================
    # Status
    # ========================================================

    def get_camera_status(
        self,
    ):
        """
        取得 Camera Driver 狀態。
        """

        with self._lifecycle_lock:
            capture = (
                self._capture
            )

            connected = bool(
                self._running
                and capture is not None
                and capture.isOpened()
            )

            return {
                "connected":
                    connected,

                "running":
                    self._running,

                "device_index":
                    self.device_index,

                "device_path":
                    self.device_path,

                "device":
                    self._get_device(),

                "width":
                    self._width,

                "height":
                    self._height,

                "fps":
                    self._fps,

                "actual_width":
                    self._actual_width,

                "actual_height":
                    self._actual_height,

                "actual_fps":
                    self._actual_fps,

                "fourcc":
                    self.fourcc,

                "frame_timeout_ms":
                    self._frame_timeout_ms,

                "last_frame_timestamp":
                    self._last_frame_timestamp,

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
            or self._capture is None
        ):
            raise RuntimeError(
                "USB camera 尚未啟動"
            )

        if not self._capture.isOpened():
            raise RuntimeError(
                "USB camera connection unavailable"
            )


    def get_frame(
        self,
    ):
        """
        取得最新 RGB frame。

        回傳格式與 D405Driver 保持一致概念：

        {
            "timestamp": float,
            "color_image": np.ndarray,
            "depth_image": None,
            "color_frame": None,
            "depth_frame": None,
        }

        vision_service 主要使用：
            frame["color_image"]
        """

        with self._frame_lock:
            self._require_running()

            capture = (
                self._capture
            )

            start_time = (
                time.monotonic()
            )

            timeout_s = (
                self._frame_timeout_ms
                / 1000.0
            )

            while True:
                ret, color_image = (
                    capture.read()
                )

                if (
                    ret
                    and color_image is not None
                ):
                    break

                elapsed = (
                    time.monotonic()
                    - start_time
                )

                if elapsed >= timeout_s:
                    raise TimeoutError(
                        "USB camera frame timeout: "
                        f"{self._frame_timeout_ms} ms"
                    )

                time.sleep(
                    0.01
                )

            if not isinstance(
                color_image,
                np.ndarray,
            ):
                raise RuntimeError(
                    "USB camera color_image "
                    "不是 numpy.ndarray"
                )

            if (
                color_image.ndim != 3
                or color_image.shape[2] != 3
            ):
                raise RuntimeError(
                    "USB camera color_image "
                    "必須為 [H, W, 3]"
                )

            timestamp = (
                time.time()
            )

            self._last_frame_timestamp = (
                timestamp
            )

            return {
                "timestamp":
                    timestamp,

                "color_image":
                    color_image,

                "depth_image":
                    None,

                "color_frame":
                    None,

                "depth_frame":
                    None,
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
        USB RGB camera 不支援 depth。
        """

        raise NotImplementedError(
            "USB camera does not support depth"
        )


    # ========================================================
    # Deprojection
    # ========================================================

    def deproject_pixel_to_point(
        self,
        x,
        y,
        depth=None,
        frame=None,
    ):
        """
        USB RGB camera 沒有 depth，
        無法直接 pixel -> camera XYZ。
        """

        raise NotImplementedError(
            "USB camera does not support deprojection"
        )


    # ========================================================
    # Intrinsics
    # ========================================================

    def get_intrinsics(
        self,
        frame=None,
    ):
        """
        一般 UVC camera 無法直接從 OpenCV VideoCapture
        取得可靠的 calibrated intrinsics。

        若未來完成 AVer calibration，
        可以另外加入 fx / fy / cx / cy。
        """

        raise NotImplementedError(
            "USB camera intrinsics are not configured"
        )


    # ========================================================
    # Point Cloud
    # ========================================================

    def get_point_cloud(
        self,
        frame=None,
    ):
        """
        USB RGB camera 不支援 point cloud。
        """

        raise NotImplementedError(
            "USB camera does not support point cloud"
        )


    # ========================================================
    # Destructor
    # ========================================================

    def __del__(
        self,
    ):
        try:
            self.stop_camera()

        except Exception:
            pass