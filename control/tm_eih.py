#安裝python3 -m pip install grpcio grpcio-tools protobuf
from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import grpc
import numpy as np
from google.protobuf import message_factory


logger = logging.getLogger(__name__)


DEFAULT_CONNECT_TIMEOUT = 3.0
DEFAULT_FRAME_TIMEOUT_MS = 3000



class TMEIHDriver:
    """
    Techman Robot Eye-in-Hand (EIH) camera driver.

    Intended for TMflow >= 2.20 EIH Camera API.

    Robot Brain camera contract:
        start_camera(...)
        stop_camera()
        get_camera_status()
        get_frame()

    Frame schema:
        {
            "timestamp": float,
            "color_image": numpy.ndarray,   # BGR [H, W, 3]
            "depth_image": None,
        }

    Important:
        Techman EIH Camera API is gRPC based and requires the official:
            EIHCameraAPI.proto
            EIHCamera.proto

        This driver does not guess a vendor RPC name. It loads the official
        protobuf definitions, discovers the available unary image RPC, and
        decodes the returned image.

    Proto search order:
        1. proto_dir passed to __init__
        2. environment variable TM_EIH_PROTO_DIR
        3. control/tm_eih_proto/
        4. control/proto/

    If only .proto files are present, grpc_tools.protoc is used to generate
    Python modules automatically into:
        <proto_dir>/_generated/
    """

    DRIVER_METADATA = {
        "name": "tm_eih",
        "manufacturer": "Techman Robot",
        "model": "Eye-in-Hand Camera",
        "capabilities": {
            "color": True,
            "depth": False,
            "deprojection": False,
            # Keep False until Robot Brain explicitly maps the vendor
            # Camera Matrix API into camera_service.get_intrinsics().
            "intrinsics": False,
            "point_cloud": False,
        },
    }

    _PROTO_BASENAMES = (
        "EIHCameraAPI",
        "EIHCamera",
    )

    _PREFERRED_IMAGE_METHOD_WORDS = (
        "image",
        "raw",
        "frame",
        "capture",
        "snapshot",
    )

    def __init__(
        self,
        ip,
        port,
        connect_timeout=DEFAULT_CONNECT_TIMEOUT,
        proto_dir=None,
    ):
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError(
                "ip must be a non-empty string"
            )

        try:
            port = int(port)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "port must be an integer"
            ) from exc

        if not (1 <= port <= 65535):
            raise ValueError(
                "port must be between 1 and 65535"
            )

        try:
            connect_timeout = float(
                connect_timeout
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "connect_timeout must be numeric"
            ) from exc

        if connect_timeout <= 0:
            raise ValueError(
                "connect_timeout must be > 0"
            )

        self.ip = ip.strip()
        self.port = port
        self.connect_timeout = connect_timeout

        self.proto_dir = (
            Path(proto_dir).expanduser().resolve()
            if proto_dir is not None
            else None
        )

        self._lifecycle_lock = (
            threading.RLock()
        )

        self._frame_lock = (
            threading.RLock()
        )

        self._channel = None
        self._running = False

        self._width = None
        self._height = None
        self._fps = None

        self._frame_timeout_ms = (
            DEFAULT_FRAME_TIMEOUT_MS
        )

        self._last_frame_timestamp = None
        self._last_error = None

        self._pb2_modules = []
        self._pb2_grpc_modules = []

        self._image_rpc = None
        self._image_request_class = None
        self._image_rpc_name = None

    # ========================================================
    # Proto / gRPC
    # ========================================================

    @property
    def _target(self):
        return f"{self.ip}:{self.port}"

    def _candidate_proto_dirs(self):
        module_dir = Path(__file__).resolve().parent

        candidates = []

        if self.proto_dir is not None:
            candidates.append(
                self.proto_dir
            )

        env_dir = os.environ.get(
            "TM_EIH_PROTO_DIR"
        )

        if env_dir:
            candidates.append(
                Path(env_dir)
                .expanduser()
                .resolve()
            )

        candidates.extend([
            module_dir / "tm_eih_proto",
            module_dir / "proto",
        ])

        result = []
        seen = set()

        for path in candidates:
            path = path.resolve()

            key = str(path)

            if key in seen:
                continue

            seen.add(key)
            result.append(path)

        return result

    @staticmethod
    def _has_generated_modules(path):
        return all(
            (path / f"{name}_pb2.py").is_file()
            for name in TMEIHDriver._PROTO_BASENAMES
        ) and all(
            (path / f"{name}_pb2_grpc.py").is_file()
            for name in TMEIHDriver._PROTO_BASENAMES
        )

    @staticmethod
    def _has_proto_files(path):
        return all(
            (path / f"{name}.proto").is_file()
            for name in TMEIHDriver._PROTO_BASENAMES
        )

    def _compile_proto_files(
        self,
        proto_dir,
    ):
        try:
            from grpc_tools import protoc
            import grpc_tools
        except ImportError as exc:
            raise RuntimeError(
                "Techman EIH proto files were found, but "
                "grpcio-tools is not installed. Install with: "
                "python3 -m pip install grpcio grpcio-tools protobuf"
            ) from exc

        generated_dir = (
            proto_dir / "_generated"
        )

        generated_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        grpc_include = (
            Path(grpc_tools.__file__)
            .resolve()
            .parent
            / "_proto"
        )

        args = [
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"-I{grpc_include}",
            f"--python_out={generated_dir}",
            f"--grpc_python_out={generated_dir}",
        ]

        args.extend(
            str(
                proto_dir
                / f"{name}.proto"
            )
            for name
            in self._PROTO_BASENAMES
        )

        result = protoc.main(args)

        if result != 0:
            raise RuntimeError(
                "Failed to compile Techman EIH proto files "
                f"(protoc exit code={result})"
            )

        if not self._has_generated_modules(
            generated_dir
        ):
            raise RuntimeError(
                "Techman EIH proto compilation completed, "
                "but generated Python modules are missing"
            )

        return generated_dir

    def _resolve_proto_module_dir(self):
        checked = []

        for path in (
            self._candidate_proto_dirs()
        ):
            checked.append(str(path))

            if self._has_generated_modules(
                path
            ):
                return path

            generated_dir = (
                path / "_generated"
            )

            if self._has_generated_modules(
                generated_dir
            ):
                return generated_dir

            if self._has_proto_files(
                path
            ):
                return self._compile_proto_files(
                    path
                )

        message = (
            "Techman EIH Camera API protobuf files were not found.\n"
            "Place the official files:\n"
            "  EIHCameraAPI.proto\n"
            "  EIHCamera.proto\n"
            "in one of:\n  - "
            + "\n  - ".join(checked)
            + "\n"
            "The official Techman tm_eih_cam_client repository "
            "contains these files under its proto/ directory."
        )

        raise RuntimeError(message)

    def _load_proto_modules(self):
        if (
            self._pb2_modules
            and self._pb2_grpc_modules
        ):
            return

        module_dir = (
            self._resolve_proto_module_dir()
        )

        module_dir_text = str(
            module_dir
        )

        if module_dir_text not in sys.path:
            sys.path.insert(
                0,
                module_dir_text,
            )

        pb2_modules = []
        grpc_modules = []

        try:
            for name in (
                self._PROTO_BASENAMES
            ):
                pb2_modules.append(
                    importlib.import_module(
                        f"{name}_pb2"
                    )
                )

                grpc_modules.append(
                    importlib.import_module(
                        f"{name}_pb2_grpc"
                    )
                )

        except Exception as exc:
            raise RuntimeError(
                "Failed to import generated Techman EIH "
                f"protobuf modules from {module_dir}: {exc}"
            ) from exc

        self._pb2_modules = pb2_modules
        self._pb2_grpc_modules = (
            grpc_modules
        )

    def _connect(self):
        if self._channel is not None:
            return

        self._load_proto_modules()

        channel = grpc.insecure_channel(
            self._target,
            options=[
                (
                    "grpc.max_receive_message_length",
                    64 * 1024 * 1024,
                ),
                (
                    "grpc.max_send_message_length",
                    16 * 1024 * 1024,
                ),
            ],
        )

        try:
            grpc.channel_ready_future(
                channel
            ).result(
                timeout=self.connect_timeout
            )

        except Exception as exc:
            try:
                channel.close()
            except Exception:
                pass

            raise RuntimeError(
                "Unable to connect to Techman EIH Camera API "
                f"at {self._target}: {exc}. "
                "Verify TMflow EIH Camera API Server is enabled "
                "and the configured port is correct."
            ) from exc

        self._channel = channel

        logger.info(
            "[TM_EIH] gRPC connected: %s",
            self._target,
        )

    def _disconnect(self):
        channel = self._channel
        self._channel = None

        self._image_rpc = None
        self._image_request_class = None
        self._image_rpc_name = None

        if channel is not None:
            try:
                channel.close()
            except Exception:
                logger.debug(
                    "[TM_EIH] channel close failed",
                    exc_info=True,
                )

    # ========================================================
    # RPC discovery
    # ========================================================

    @staticmethod
    def _message_class(
        descriptor,
    ):
        try:
            return message_factory.GetMessageClass(
                descriptor
            )
        except AttributeError:
            factory = (
                message_factory.MessageFactory()
            )

            return factory.GetPrototype(
                descriptor
            )

    @staticmethod
    def _iter_stub_classes(
        grpc_module,
    ):
        for name in dir(grpc_module):
            if not name.endswith("Stub"):
                continue

            value = getattr(
                grpc_module,
                name,
                None,
            )

            if isinstance(value, type):
                yield name, value

    def _method_score(
        self,
        service_name,
        method_name,
        output_descriptor,
    ):
        text = (
            f"{service_name} "
            f"{method_name} "
            f"{output_descriptor.full_name}"
        ).lower()

        score = 0

        if "image" in text:
            score += 100

        if "raw" in text:
            score += 30

        if "frame" in text:
            score += 20

        if "capture" in text:
            score += 15

        if "snapshot" in text:
            score += 15

        if method_name.lower().startswith(
            ("get", "read")
        ):
            score += 10

        return score

    def _discover_image_rpc(self):
        if self._image_rpc is not None:
            return

        if self._channel is None:
            raise RuntimeError(
                "EIH gRPC channel is not connected"
            )

        # Generated grpc module may contain one or more service stubs.
        stub_classes = {}

        for grpc_module in (
            self._pb2_grpc_modules
        ):
            for stub_name, stub_class in (
                self._iter_stub_classes(
                    grpc_module
                )
            ):
                stub_classes[
                    stub_name
                ] = stub_class

        candidates = []

        for pb2_module in (
            self._pb2_modules
        ):
            descriptor = getattr(
                pb2_module,
                "DESCRIPTOR",
                None,
            )

            if descriptor is None:
                continue

            for service in (
                descriptor
                .services_by_name
                .values()
            ):
                stub_name = (
                    f"{service.name}Stub"
                )

                stub_class = (
                    stub_classes.get(
                        stub_name
                    )
                )

                if stub_class is None:
                    continue

                for method in service.methods:
                    if (
                        method.client_streaming
                        or method.server_streaming
                    ):
                        continue

                    score = self._method_score(
                        service.full_name,
                        method.name,
                        method.output_type,
                    )

                    if score <= 0:
                        continue

                    candidates.append(
                        (
                            score,
                            service,
                            method,
                            stub_class,
                        )
                    )

        if not candidates:
            raise RuntimeError(
                "No unary image-related RPC was found in "
                "the Techman EIH protobuf definitions"
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        errors = []

        # Probe candidate methods. Some image-related RPCs may only
        # return configuration; we accept a method only if its response
        # can actually be decoded into an image.
        for (
            _,
            service,
            method,
            stub_class,
        ) in candidates:

            stub = stub_class(
                self._channel
            )

            rpc = getattr(
                stub,
                method.name,
                None,
            )

            if not callable(rpc):
                continue

            request_class = (
                self._message_class(
                    method.input_type
                )
            )

            try:
                request = (
                    request_class()
                )

                response = rpc(
                    request,
                    timeout=self._rpc_timeout_seconds(),
                )

                image = (
                    self._decode_image_response(
                        response
                    )
                )

                if image is None:
                    raise RuntimeError(
                        "response contains no decodable image"
                    )

            except Exception as exc:
                errors.append(
                    f"{service.full_name}."
                    f"{method.name}: {exc}"
                )
                continue

            self._image_rpc = rpc
            self._image_request_class = (
                request_class
            )
            self._image_rpc_name = (
                f"{service.full_name}."
                f"{method.name}"
            )

            logger.info(
                "[TM_EIH] image RPC selected: %s",
                self._image_rpc_name,
            )

            return

        raise RuntimeError(
            "Techman EIH protobuf definitions were loaded, "
            "but no image RPC returned a decodable image. "
            "Candidate errors: "
            + " | ".join(errors[:8])
        )

    # ========================================================
    # Protobuf response helpers
    # ========================================================

    @staticmethod
    def _field_items(
        message,
    ):
        if message is None:
            return []

        descriptor = getattr(
            message,
            "DESCRIPTOR",
            None,
        )

        if descriptor is None:
            return []

        items = []

        for field in descriptor.fields:
            try:
                value = getattr(
                    message,
                    field.name,
                )
            except Exception:
                continue

            items.append(
                (field, value)
            )

        return items

    @classmethod
    def _walk_message(
        cls,
        message,
        prefix="",
        depth=0,
    ):
        if depth > 5:
            return

        for field, value in (
            cls._field_items(message)
        ):
            path = (
                f"{prefix}.{field.name}"
                if prefix
                else field.name
            )

            yield path, field, value

            # TYPE_MESSAGE == 11
            if field.type == field.TYPE_MESSAGE:
                if field.is_repeated:
                    for index, child in (
                        enumerate(value)
                    ):
                        yield from cls._walk_message(
                            child,
                            prefix=(
                                f"{path}[{index}]"
                            ),
                            depth=depth + 1,
                        )
                else:
                    # In proto3 HasField may not be legal for
                    # non-message scalar fields, but is fine here.
                    try:
                        present = message.HasField(
                            field.name
                        )
                    except Exception:
                        present = True

                    if present:
                        yield from cls._walk_message(
                            value,
                            prefix=path,
                            depth=depth + 1,
                        )

    @classmethod
    def _find_scalar(
        cls,
        message,
        names,
    ):
        names = {
            name.lower()
            for name in names
        }

        for path, field, value in (
            cls._walk_message(message)
        ):
            leaf = (
                path.split(".")[-1]
                .split("[")[0]
                .lower()
            )

            if leaf not in names:
                continue

            if field.type in (
                field.TYPE_INT32,
                field.TYPE_INT64,
                field.TYPE_UINT32,
                field.TYPE_UINT64,
                field.TYPE_SINT32,
                field.TYPE_SINT64,
                field.TYPE_FIXED32,
                field.TYPE_FIXED64,
                field.TYPE_SFIXED32,
                field.TYPE_SFIXED64,
                field.TYPE_FLOAT,
                field.TYPE_DOUBLE,
                field.TYPE_STRING,
                field.TYPE_ENUM,
            ):
                return value

        return None

    @classmethod
    def _find_bytes_candidates(
        cls,
        message,
    ):
        preferred_names = {
            "image",
            "data",
            "image_data",
            "raw_image",
            "rawdata",
            "raw_data",
            "buffer",
            "bytes",
            "payload",
        }

        candidates = []

        for path, field, value in (
            cls._walk_message(message)
        ):
            if field.type != field.TYPE_BYTES:
                continue

            if not value:
                continue

            leaf = (
                path.split(".")[-1]
                .split("[")[0]
                .lower()
            )

            score = len(value)

            if leaf in preferred_names:
                score += 100_000_000

            if "image" in leaf:
                score += 50_000_000

            candidates.append(
                (
                    score,
                    path,
                    bytes(value),
                )
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates

    @staticmethod
    def _decode_compressed_image(
        data,
    ):
        if len(data) < 4:
            return None

        is_jpeg = (
            data[0:2] == b"\xff\xd8"
        )

        is_png = (
            data[0:8]
            == b"\x89PNG\r\n\x1a\n"
        )

        is_bmp = (
            data[0:2] == b"BM"
        )

        if not (
            is_jpeg
            or is_png
            or is_bmp
        ):
            return None

        array = np.frombuffer(
            data,
            dtype=np.uint8,
        )

        return cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

    @staticmethod
    def _normalize_pixel_format(
        value,
    ):
        if value is None:
            return ""

        if isinstance(value, int):
            return str(value)

        return (
            str(value)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )

    @classmethod
    def _decode_raw_image(
        cls,
        data,
        width,
        height,
        pixel_format,
    ):
        try:
            width = int(width)
            height = int(height)
        except (TypeError, ValueError):
            return None

        if width <= 0 or height <= 0:
            return None

        size = len(data)
        pixels = width * height

        if pixels <= 0:
            return None

        fmt = cls._normalize_pixel_format(
            pixel_format
        )

        array = np.frombuffer(
            data,
            dtype=np.uint8,
        )

        # 3-channel RGB/BGR
        if size == pixels * 3:
            image = array.reshape(
                height,
                width,
                3,
            )

            if (
                "rgb" in fmt
                and "bgr" not in fmt
            ):
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                )

            return image.copy()

        # 4-channel RGBA/BGRA
        if size == pixels * 4:
            image = array.reshape(
                height,
                width,
                4,
            )

            if "rgba" in fmt:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2BGR,
                )
            else:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2BGR,
                )

            return image

        # Mono / Bayer 8-bit
        if size == pixels:
            image = array.reshape(
                height,
                width,
            )

            bayer_codes = {
                "bayerrg8":
                    cv2.COLOR_BAYER_RG2BGR,
                "bayerbg8":
                    cv2.COLOR_BAYER_BG2BGR,
                "bayergr8":
                    cv2.COLOR_BAYER_GR2BGR,
                "bayergb8":
                    cv2.COLOR_BAYER_GB2BGR,
                "rg8":
                    cv2.COLOR_BAYER_RG2BGR,
                "bg8":
                    cv2.COLOR_BAYER_BG2BGR,
                "gr8":
                    cv2.COLOR_BAYER_GR2BGR,
                "gb8":
                    cv2.COLOR_BAYER_GB2BGR,
            }

            for key, code in (
                bayer_codes.items()
            ):
                if key in fmt:
                    return cv2.cvtColor(
                        image,
                        code,
                    )

            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        # Packed YUYV / YUY2
        if size == pixels * 2:
            image = array.reshape(
                height,
                width,
                2,
            )

            if (
                "yuy" in fmt
                or "yuv422" in fmt
            ):
                return cv2.cvtColor(
                    image,
                    cv2.COLOR_YUV2BGR_YUY2,
                )

        return None

    @classmethod
    def _decode_image_response(
        cls,
        response,
    ):
        if response is None:
            return None

        width = cls._find_scalar(
            response,
            {
                "width",
                "image_width",
                "imagewidth",
                "cols",
            },
        )

        height = cls._find_scalar(
            response,
            {
                "height",
                "image_height",
                "imageheight",
                "rows",
            },
        )

        pixel_format = cls._find_scalar(
            response,
            {
                "pixel_format",
                "pixelformat",
                "format",
                "encoding",
                "image_type",
                "imagetype",
            },
        )

        byte_candidates = (
            cls._find_bytes_candidates(
                response
            )
        )

        for _, _, data in byte_candidates:
            compressed = (
                cls._decode_compressed_image(
                    data
                )
            )

            if compressed is not None:
                return compressed

            raw = cls._decode_raw_image(
                data,
                width=width,
                height=height,
                pixel_format=pixel_format,
            )

            if raw is not None:
                return raw

        return None

    # ========================================================
    # Camera lifecycle
    # ========================================================

    def _rpc_timeout_seconds(self):
        timeout_ms = (
            self._frame_timeout_ms
            if self._frame_timeout_ms is not None
            else DEFAULT_FRAME_TIMEOUT_MS
        )

        return max(
            0.1,
            float(timeout_ms) / 1000.0,
        )

    @staticmethod
    def _normalize_optional_positive_int(
        value,
        name,
    ):
        if value is None:
            return None

        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be an integer or None"
            ) from exc

        if value <= 0:
            raise ValueError(
                f"{name} must be > 0"
            )

        return value

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
        Connect to the Techman EIH Camera API and verify one image.

        width / height / fps:
            Kept for the common Robot Brain camera interface.
            EIH image size should be configured through the Techman
            camera API / TMflow, so these values are not forced here.

        enable_depth:
            Must be False. The TM EIH camera is treated as RGB-only
            in this Robot Brain driver.

        align_to:
            Must be None/"none".
        """

        if (
            enable_color is not None
            and not bool(enable_color)
        ):
            raise ValueError(
                "TM EIH camera requires color stream"
            )

        if bool(enable_depth):
            raise NotImplementedError(
                "TM EIH camera driver does not provide depth"
            )

        if (
            align_to is not None
            and str(align_to).strip().lower()
            not in {"", "none"}
        ):
            raise NotImplementedError(
                "TM EIH camera does not support "
                "Robot Brain depth/color alignment"
            )

        if frame_timeout_ms is None:
            frame_timeout_ms = (
                DEFAULT_FRAME_TIMEOUT_MS
            )

        frame_timeout_ms = (
            self._normalize_optional_positive_int(
                frame_timeout_ms,
                "frame_timeout_ms",
            )
        )

        with self._lifecycle_lock:
            if self._running:
                return True

            self._frame_timeout_ms = (
                frame_timeout_ms
            )

            self._last_error = None

            try:
                self._connect()
                self._discover_image_rpc()

                image = self._read_image()

                if (
                    not isinstance(
                        image,
                        np.ndarray,
                    )
                    or image.ndim != 3
                    or image.shape[2] != 3
                ):
                    raise RuntimeError(
                        "TM EIH image must be BGR [H, W, 3]"
                    )

                self._height = int(
                    image.shape[0]
                )

                self._width = int(
                    image.shape[1]
                )

                # The RPC is snapshot-oriented; do not invent an FPS.
                self._fps = (
                    float(fps)
                    if fps is not None
                    else None
                )

                self._last_frame_timestamp = (
                    time.time()
                )

                self._running = True

                logger.info(
                    "[TM_EIH] camera started "
                    "target=%s resolution=%sx%s rpc=%s",
                    self._target,
                    self._width,
                    self._height,
                    self._image_rpc_name,
                )

                return True

            except Exception as exc:
                self._last_error = str(exc)
                self._running = False

                self._disconnect()

                logger.exception(
                    "[TM_EIH] camera start failed"
                )

                raise

    def stop_camera(self):
        with self._lifecycle_lock:
            self._running = False

            self._disconnect()

            logger.info(
                "[TM_EIH] camera stopped: %s",
                self._target,
            )

            return True

    # ========================================================
    # Status
    # ========================================================

    def get_camera_status(self):
        with self._lifecycle_lock:
            connected = bool(
                self._running
                and self._channel is not None
                and self._image_rpc is not None
            )

            return {
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
                    True,

                "depth_enabled":
                    False,

                "fault":
                    self._last_error is not None,

                # TM-specific diagnostics
                "ip":
                    self.ip,

                "port":
                    self.port,

                "grpc_target":
                    self._target,

                "image_rpc":
                    self._image_rpc_name,

                "last_frame_timestamp":
                    self._last_frame_timestamp,

                "error":
                    self._last_error,
            }

    # ========================================================
    # Frame
    # ========================================================

    def _read_image(self):
        if self._image_rpc is None:
            raise RuntimeError(
                "TM EIH image RPC has not been initialized"
            )

        if self._image_request_class is None:
            raise RuntimeError(
                "TM EIH image request type has not been initialized"
            )

        request = (
            self._image_request_class()
        )

        response = self._image_rpc(
            request,
            timeout=self._rpc_timeout_seconds(),
        )

        image = (
            self._decode_image_response(
                response
            )
        )

        if image is None:
            raise RuntimeError(
                "TM EIH image RPC returned a response "
                "that could not be decoded"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
        ):
            raise RuntimeError(
                "Decoded TM EIH image is not BGR [H, W, 3]"
            )

        return np.ascontiguousarray(
            image,
            dtype=np.uint8,
        )

    def get_frame(self):
        with self._lifecycle_lock:
            if not self._running:
                raise RuntimeError(
                    "TM EIH camera is not running. "
                    "Call start_camera() first."
                )

        with self._frame_lock:
            try:
                image = self._read_image()

                timestamp = time.time()

                self._height = int(
                    image.shape[0]
                )

                self._width = int(
                    image.shape[1]
                )

                self._last_frame_timestamp = (
                    timestamp
                )

                self._last_error = None

                return {
                    "timestamp":
                        timestamp,

                    "color_image":
                        image,

                    "depth_image":
                        None,
                }

            except Exception as exc:
                self._last_error = str(exc)

                logger.exception(
                    "[TM_EIH] get_frame failed"
                )

                raise

    # ========================================================
    # Optional lifecycle alias
    # ========================================================

    def shutdown(self):
        return self.stop_camera()
