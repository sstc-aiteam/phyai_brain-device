import numpy as np
import config
from services import (arm_service, camera_service)
from utils.response import (success, error)

MODULE = "coordinate"

# ============================================================
# Validation Helpers
# ============================================================

def _as_float(
    value,
    name,
):
    if value is None:
        raise ValueError(
            f"{name} is None"
        )

    return float(
        value
    )


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


def _get_matrix_4x4(
    matrix_data,
    name,
):
    matrix = np.array(
        matrix_data,
        dtype=float,
    )

    if matrix.shape != (
        4,
        4,
    ):
        raise ValueError(
            f"{name} must be 4x4, "
            f"got {matrix.shape}"
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise ValueError(
            f"{name} contains "
            "non-finite values"
        )

    return matrix


def _get_camera_mount(
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

    mount = (
        camera_config.get(
            "mount"
        )
    )

    if not isinstance(
        mount,
        dict,
    ):
        raise ValueError(
            f"Camera '{camera_name}' "
            "mount config is missing "
            "or invalid"
        )

    mode = (
        mount.get(
            "mode"
        )
    )

    if not isinstance(
        mode,
        str,
    ):
        raise ValueError(
            f"Camera '{camera_name}' "
            "mount.mode must be a string"
        )

    mode = (
        mode
        .strip()
        .lower()
    )

    if mode not in {
        "fixed",
        "wrist",
    }:
        raise ValueError(
            f"Camera '{camera_name}' "
            "mount.mode must be "
            "'fixed' or 'wrist'"
        )

    if "T_matrix" not in mount:
        raise ValueError(
            f"Camera '{camera_name}' "
            "mount.T_matrix is missing"
        )

    t_matrix = (
        _get_matrix_4x4(
            mount[
                "T_matrix"
            ],
            (
                f"CAMERAS['{camera_name}']"
                "['mount']['T_matrix']"
            ),
        )
    )

    arm_name = (
        mount.get(
            "arm_name"
        )
    )

    if mode == "wrist":
        if not isinstance(
            arm_name,
            str,
        ) or not arm_name.strip():
            raise ValueError(
                f"Camera '{camera_name}' "
                "wrist mount requires "
                "arm_name"
            )

        arm_name = (
            arm_name
            .strip()
            .lower()
        )

        if arm_name not in config.ARMS:
            raise ValueError(
                f"Camera '{camera_name}' "
                f"references unsupported arm: "
                f"{arm_name}"
            )

    else:
        arm_name = None

    return {
        "camera_name":
            camera_name,

        "mode":
            mode,

        "T_matrix":
            t_matrix,

        "arm_name":
            arm_name,
    }


def _tcp_pose_to_t_base_tcp(
    tcp_pose,
):
    """
    將 UR TCP pose：
        [x, y, z, rx, ry, rz]

    轉為：
        T_base_tcp
    """

    if tcp_pose is None:
        raise ValueError(
            "tcp_pose is required"
        )

    if not isinstance(
        tcp_pose,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):
        raise ValueError(
            "tcp_pose must be a sequence"
        )

    if len(
        tcp_pose
    ) != 6:
        raise ValueError(
            f"tcp_pose must have "
            f"6 values, got "
            f"{len(tcp_pose)}"
        )

    (
        x,
        y,
        z,
        rx,
        ry,
        rz,
    ) = [
        float(v)
        for v in tcp_pose
    ]

    rotation_vector = (
        np.array(
            [
                rx,
                ry,
                rz,
            ],
            dtype=float,
        )
    )

    theta = float(
        np.linalg.norm(
            rotation_vector
        )
    )

    if theta < 1e-12:
        rotation_matrix = (
            np.eye(
                3,
                dtype=float,
            )
        )

    else:
        k = (
            rotation_vector
            / theta
        )

        kx = np.array(
            [
                [
                    0.0,
                    -k[2],
                    k[1],
                ],
                [
                    k[2],
                    0.0,
                    -k[0],
                ],
                [
                    -k[1],
                    k[0],
                    0.0,
                ],
            ],
            dtype=float,
        )

        rotation_matrix = (
            np.eye(
                3,
                dtype=float,
            )
            + np.sin(
                theta
            ) * kx
            + (
                1.0
                - np.cos(
                    theta
                )
            )
            * (
                kx
                @ kx
            )
        )

    t_base_tcp = (
        np.eye(
            4,
            dtype=float,
        )
    )

    t_base_tcp[
        :3,
        :3,
    ] = (
        rotation_matrix
    )

    t_base_tcp[
        :3,
        3,
    ] = [
        x,
        y,
        z,
    ]

    return t_base_tcp


def _extract_tcp_pose(
    arm_pose_result,
):
    """
    支援 arm_service.get_arm_pose() 常見 response 格式。

    優先尋找：
        data.pose
        data.arm_pose
        pose
        arm_pose
    """

    if not isinstance(
        arm_pose_result,
        dict,
    ):
        raise RuntimeError(
            "invalid arm pose response"
        )

    if (
        arm_pose_result.get(
            "status"
        )
        == "error"
    ):
        raise RuntimeError(
            arm_pose_result.get(
                "message",
                "failed to get arm pose",
            )
        )

    candidates = []

    data = (
        arm_pose_result.get(
            "data"
        )
    )

    if isinstance(
        data,
        dict,
    ):
        candidates.extend([
            data.get(
                "pose"
            ),
            data.get(
                "arm_pose"
            ),
            data.get(
                "tcp_pose"
            ),
        ])

    candidates.extend([
        arm_pose_result.get(
            "pose"
        ),
        arm_pose_result.get(
            "arm_pose"
        ),
        arm_pose_result.get(
            "tcp_pose"
        ),
    ])

    for candidate in candidates:
        if (
            isinstance(
                candidate,
                (
                    list,
                    tuple,
                    np.ndarray,
                ),
            )
            and len(
                candidate
            ) == 6
        ):
            return [
                float(v)
                for v in candidate
            ]

    raise RuntimeError(
        "arm pose response does not "
        "contain a valid TCP pose"
    )


def _get_current_tcp_pose(
    arm_name,
):
    result = (
        arm_service
        .get_arm_pose(
            arm_name=
                arm_name
        )
    )

    return (
        _extract_tcp_pose(
            result
        )
    )


# ============================================================
# Camera Transform
# ============================================================

def get_t_base_camera(
    camera_name,
):
    """
    依 Camera mount config 計算目前：
        T_base_camera

    fixed:
        T_matrix = T_base_camera

    wrist:
        T_matrix = T_tcp_camera
        T_base_camera =
            T_base_tcp @ T_tcp_camera
    """

    mount = (
        _get_camera_mount(
            camera_name
        )
    )

    mode = (
        mount[
            "mode"
        ]
    )

    t_matrix = (
        mount[
            "T_matrix"
        ]
    )

    if mode == "fixed":
        return (
            t_matrix
        )

    if mode == "wrist":
        arm_name = (
            mount[
                "arm_name"
            ]
        )

        tcp_pose = (
            _get_current_tcp_pose(
                arm_name
            )
        )

        t_base_tcp = (
            _tcp_pose_to_t_base_tcp(
                tcp_pose
            )
        )

        return (
            t_base_tcp
            @ t_matrix
        )

    raise RuntimeError(
        f"Unsupported mount mode: "
        f"{mode}"
    )


# ============================================================
# Pixel + Depth -> Camera XYZ
# ============================================================

def pixel_depth_to_camera_xyz(
    camera_name,
    pixel_x,
    pixel_y,
    depth_m,
):
    """
    pixel + depth
        ↓
    camera XYZ

    Camera-specific deprojection
    由 camera_service / driver 負責。
    """

    action = (
        "pixel_depth_to_camera_xyz"
    )

    try:
        camera_name = (
            _normalize_camera_name(
                camera_name
            )
        )

        u = _as_float(
            pixel_x,
            "pixel_x",
        )

        v = _as_float(
            pixel_y,
            "pixel_y",
        )

        depth = _as_float(
            depth_m,
            "depth_m",
        )

        if depth <= 0:
            raise ValueError(
                "depth_m must be "
                "greater than 0"
            )

        point = (
            camera_service
            .deproject_pixel_to_point_value(
                camera_name=
                    camera_name,

                x=u,
                y=v,

                depth=depth,

                frame=None,
            )
        )

        return success(
            MODULE,
            action,
            result=True,
            data={
                "camera_name":
                    camera_name,

                "camera_xyz": [
                    float(
                        point[0]
                    ),
                    float(
                        point[1]
                    ),
                    float(
                        point[2]
                    ),
                ],
            },
        )

    except NotImplementedError as exc:
        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                "DeprojectionNotSupported",
        )

    except Exception as exc:
        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# Camera XYZ -> Robot Base XYZ
# ============================================================

def camera_xyz_to_robot_xyz(
    camera_name,
    camera_xyz,
):
    """
    camera XYZ
        ↓
    Robot Base XYZ

    mount.mode 與 T_matrix
    直接從 config.CAMERAS[camera_name]
    取得。
    """

    action = (
        "camera_xyz_to_robot_xyz"
    )

    try:
        camera_name = (
            _normalize_camera_name(
                camera_name
            )
        )

        if (
            not isinstance(
                camera_xyz,
                (
                    list,
                    tuple,
                    np.ndarray,
                ),
            )
            or len(
                camera_xyz
            ) != 3
        ):
            raise ValueError(
                "camera_xyz must be "
                "[x, y, z]"
            )

        x, y, z = [
            float(v)
            for v in camera_xyz
        ]

        point_camera = np.array(
            [
                x,
                y,
                z,
                1.0,
            ],
            dtype=float,
        )

        t_base_camera = (
            get_t_base_camera(
                camera_name
            )
        )

        point_base = (
            t_base_camera
            @ point_camera
        )

        mount = (
            _get_camera_mount(
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

                "mount_mode":
                    mount[
                        "mode"
                    ],

                "arm_name":
                    mount[
                        "arm_name"
                    ],

                "camera_xyz": [
                    x,
                    y,
                    z,
                ],

                "robot_xyz": [
                    float(
                        point_base[0]
                    ),
                    float(
                        point_base[1]
                    ),
                    float(
                        point_base[2]
                    ),
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


# ============================================================
# Pixel + Depth -> Robot Base XYZ
# ============================================================

def pixel_depth_to_robot_xyz(
    camera_name,
    pixel_x,
    pixel_y,
    depth_m,
):
    """
    pixel + depth
        ↓
    camera XYZ
        ↓
    robot base XYZ
    """

    action = (
        "pixel_depth_to_robot_xyz"
    )

    try:
        camera_name = (
            _normalize_camera_name(
                camera_name
            )
        )

        u = _as_float(
            pixel_x,
            "pixel_x",
        )

        v = _as_float(
            pixel_y,
            "pixel_y",
        )

        depth = _as_float(
            depth_m,
            "depth_m",
        )

        if depth <= 0:
            raise ValueError(
                "depth_m must be "
                "greater than 0"
            )

        camera_xyz = (
            camera_service
            .deproject_pixel_to_point_value(
                camera_name=
                    camera_name,

                x=u,
                y=v,

                depth=depth,

                frame=None,
            )
        )

        t_base_camera = (
            get_t_base_camera(
                camera_name
            )
        )

        point_camera = np.array(
            [
                float(
                    camera_xyz[0]
                ),
                float(
                    camera_xyz[1]
                ),
                float(
                    camera_xyz[2]
                ),
                1.0,
            ],
            dtype=float,
        )

        point_base = (
            t_base_camera
            @ point_camera
        )

        mount = (
            _get_camera_mount(
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

                "mount_mode":
                    mount[
                        "mode"
                    ],

                "arm_name":
                    mount[
                        "arm_name"
                    ],

                "camera_xyz": [
                    float(
                        camera_xyz[0]
                    ),
                    float(
                        camera_xyz[1]
                    ),
                    float(
                        camera_xyz[2]
                    ),
                ],

                "robot_xyz": [
                    float(
                        point_base[0]
                    ),
                    float(
                        point_base[1]
                    ),
                    float(
                        point_base[2]
                    ),
                ],
            },
        )

    except NotImplementedError as exc:
        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                "DeprojectionNotSupported",
        )

    except Exception as exc:
        return error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# Camera XYZ -> Robot Base XYZ Value API
# ============================================================

def camera_xyz_to_robot_xyz_value(
    camera_name,
    camera_xyz,
):
    """
    給 vision_service 等內部 Python service 使用。

    成功：
        return [x, y, z]

    失敗：
        raise Exception
    """

    camera_name = (
        _normalize_camera_name(
            camera_name
        )
    )

    if (
        not isinstance(
            camera_xyz,
            (
                list,
                tuple,
                np.ndarray,
            ),
        )
        or len(
            camera_xyz
        ) != 3
    ):
        raise ValueError(
            "camera_xyz must be "
            "[x, y, z]"
        )

    point_camera = np.array(
        [
            float(
                camera_xyz[0]
            ),
            float(
                camera_xyz[1]
            ),
            float(
                camera_xyz[2]
            ),
            1.0,
        ],
        dtype=float,
    )

    t_base_camera = (
        get_t_base_camera(
            camera_name
        )
    )

    point_base = (
        t_base_camera
        @ point_camera
    )

    return [
        float(
            point_base[0]
        ),
        float(
            point_base[1]
        ),
        float(
            point_base[2]
        ),
    ]