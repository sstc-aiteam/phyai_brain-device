import cv2
import numpy as np
import config

from services import worldmodel_service
from utils.response import success, error

MODULE = "coordinate"


# ============================================================
# Validation Helpers
# ============================================================

def _as_float(value, name):
    if value is None:
        raise ValueError(f"{name} is None")

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def _normalize_camera_name(camera_name):
    if not isinstance(camera_name, str):
        raise ValueError("camera_name must be a string")

    camera_name = camera_name.strip().lower()

    if not camera_name:
        raise ValueError("camera_name must not be empty")

    if camera_name not in config.CAMERAS:
        raise ValueError(f"Unsupported camera: {camera_name}. Supported cameras: {', '.join(sorted(config.CAMERAS))}")

    return camera_name


def _normalize_arm_name(arm_name):
    if not isinstance(arm_name, str):
        raise ValueError("arm_name must be a string")

    arm_name = arm_name.strip().lower()

    if not arm_name:
        raise ValueError("arm_name must not be empty")

    if arm_name not in config.ARMS:
        raise ValueError(f"Unsupported arm: {arm_name}. Supported arms: {', '.join(sorted(config.ARMS))}")

    return arm_name


def _normalize_xyz(xyz, name):
    if not isinstance(xyz, (list, tuple, np.ndarray)) or np.asarray(xyz).shape != (3,):
        raise ValueError(f"{name} must be [x, y, z]")

    return [
        _as_float(xyz[0], f"{name}[0]"),
        _as_float(xyz[1], f"{name}[1]"),
        _as_float(xyz[2], f"{name}[2]")
    ]


def _get_matrix_4x4(matrix_data, name):
    try:
        matrix = np.array(matrix_data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")

    return matrix


# ============================================================
# Camera Mount
# ============================================================

def _get_camera_mount(camera_name):
    """
    fixed:
        T_matrix = T_base_camera
        arm_name = None

    wrist:
        T_matrix = T_tcp_camera
        arm_name = associated arm
    """

    camera_name = _normalize_camera_name(camera_name)
    camera_config = config.CAMERAS[camera_name]
    mount = camera_config.get("mount")

    if not isinstance(mount, dict):
        raise ValueError(f"Camera '{camera_name}' mount config is missing or invalid")

    mode = mount.get("mode")

    if not isinstance(mode, str):
        raise ValueError(f"Camera '{camera_name}' mount.mode must be a string")

    mode = mode.strip().lower()

    if mode not in {"fixed", "wrist"}:
        raise ValueError(f"Camera '{camera_name}' mount.mode must be 'fixed' or 'wrist'")

    if "T_matrix" not in mount:
        raise ValueError(f"Camera '{camera_name}' mount.T_matrix is missing")

    t_matrix = _get_matrix_4x4(mount["T_matrix"], f"CAMERAS['{camera_name}']['mount']['T_matrix']")

    arm_name = None if mode == "fixed" else _normalize_arm_name(mount.get("arm_name"))

    return {
        "camera_name": camera_name,
        "mode": mode,
        "T_matrix": t_matrix,
        "arm_name": arm_name
    }


# ============================================================
# UR TCP Pose -> Homogeneous Matrix
# ============================================================

def _tcp_pose_to_t_base_tcp(tcp_pose):
    """
    UR TCP pose:
        [x, y, z, rx, ry, rz]

    Rotation vector:
        [rx, ry, rz] in radians

    Return:
        T_base_tcp
    """

    if not isinstance(tcp_pose, (list, tuple, np.ndarray)) or np.asarray(tcp_pose).shape != (6,):
        raise ValueError("tcp_pose must contain 6 values: [x, y, z, rx, ry, rz]")

    values = [_as_float(value, f"tcp_pose[{index}]") for index, value in enumerate(tcp_pose)]

    x, y, z, rx, ry, rz = values

    rotation_vector = np.array([rx, ry, rz], dtype=float)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    t_base_tcp = np.eye(4, dtype=float)
    t_base_tcp[:3, :3] = rotation_matrix
    t_base_tcp[:3, 3] = [x, y, z]

    return t_base_tcp


# ============================================================
# World Model -> Current TCP Pose
# ============================================================

def _extract_tcp_pose(arm_pose_result, arm_name):
    """
    Parse worldmodel_service.get_arm_pose() response.

    Expected response:
        {
            "status": "success",
            "result": true,
            "data": {
                "arms": [
                    {
                        "arm_name": "left",
                        "pose": [x, y, z, rx, ry, rz]
                    }
                ]
            }
        }
    """

    arm_name = _normalize_arm_name(arm_name)

    if not isinstance(arm_pose_result, dict):
        raise RuntimeError("invalid arm pose response")

    if arm_pose_result.get("status") == "error" or arm_pose_result.get("result") is False:
        raise RuntimeError(arm_pose_result.get("message") or f"failed to get pose for arm '{arm_name}'")

    data = arm_pose_result.get("data")

    if not isinstance(data, dict):
        raise RuntimeError("arm pose response does not contain valid data")

    arms = data.get("arms")

    if not isinstance(arms, list):
        raise RuntimeError("arm pose response does not contain arms list")

    target_arm = None

    for arm_data in arms:
        if not isinstance(arm_data, dict):
            continue

        current_name = arm_data.get("arm_name")

        if isinstance(current_name, str) and current_name.strip().lower() == arm_name:
            target_arm = arm_data
            break

    if target_arm is None:
        raise RuntimeError(f"arm '{arm_name}' pose not found in response")

    pose = target_arm.get("pose")

    if not isinstance(pose, (list, tuple, np.ndarray)) or np.asarray(pose).shape != (6,):
        raise RuntimeError(f"arm '{arm_name}' does not contain a valid TCP pose")

    return [_as_float(value, f"{arm_name}.pose[{index}]") for index, value in enumerate(pose)]


def _get_current_tcp_pose(arm_name):
    arm_name = _normalize_arm_name(arm_name)

    result = worldmodel_service.get_arm_pose(arm_name=arm_name)

    return _extract_tcp_pose(result, arm_name)


# ============================================================
# Camera Transform
# ============================================================

def get_t_base_camera(camera_name):
    """
    Calculate current T_base_camera.

    fixed:
        T_base_camera = T_matrix

    wrist:
        T_base_camera = T_base_tcp @ T_tcp_camera
    """

    mount = _get_camera_mount(camera_name)

    mode = mount["mode"]
    t_matrix = mount["T_matrix"]

    if mode == "fixed":
        return t_matrix.copy()

    if mode == "wrist":
        arm_name = mount["arm_name"]
        tcp_pose = _get_current_tcp_pose(arm_name)
        t_base_tcp = _tcp_pose_to_t_base_tcp(tcp_pose)

        return t_base_tcp @ t_matrix

    raise RuntimeError(f"Unsupported mount mode: {mode}")


# ============================================================
# Camera XYZ -> Robot Base XYZ (Value)
# ============================================================

def _camera_xyz_to_robot_xyz(camera_name, camera_xyz):
    camera_name = _normalize_camera_name(camera_name)
    camera_xyz = _normalize_xyz(camera_xyz, "camera_xyz")

    point_camera = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2], 1.0], dtype=float)

    t_base_camera = get_t_base_camera(camera_name)
    point_base = t_base_camera @ point_camera

    if not np.all(np.isfinite(point_base)):
        raise RuntimeError("coordinate transform produced non-finite values")

    return [
        float(point_base[0]),
        float(point_base[1]),
        float(point_base[2])
    ]


# ============================================================
# Camera XYZ -> Robot Base XYZ
# ============================================================

def camera_xyz_to_robot_xyz(camera_name, camera_xyz):
    action = "camera_xyz_to_robot_xyz"

    try:
        camera_name = _normalize_camera_name(camera_name)
        camera_xyz = _normalize_xyz(camera_xyz, "camera_xyz")

        robot_xyz = _camera_xyz_to_robot_xyz(
            camera_name=camera_name,
            camera_xyz=camera_xyz
        )

        mount = _get_camera_mount(camera_name)

        return success(
            MODULE,
            action,
            result=True,
            data={
                "camera_name": camera_name,
                "mount_mode": mount["mode"],
                "arm_name": mount["arm_name"],
                "camera_xyz": camera_xyz,
                "robot_xyz": robot_xyz
            }
        )

    except Exception as exc:
        return error(MODULE, action, error=exc, error_type=type(exc).__name__)


# ============================================================
# Camera XYZ -> Robot Base XYZ Value API
# ============================================================

def camera_xyz_to_robot_xyz_value(camera_name, camera_xyz):
    return _camera_xyz_to_robot_xyz(
        camera_name=camera_name,
        camera_xyz=camera_xyz
    )


# ============================================================
# TCP Delta Action -> Target TCP Pose
# ============================================================

def delta_tcp_to_base_pose(tcp_pose, delta_pose):
    """
    Convert TCP-relative delta pose to a target pose in Base frame.

    tcp_pose:
        [x, y, z, rx, ry, rz]

    delta_pose:
        [dx, dy, dz, drx, dry, drz]

    Translation:
        meters

    Rotation:
        radians (rotation vector)
    """

    t_base_tcp = _tcp_pose_to_t_base_tcp(tcp_pose)

    delta_pose = np.asarray(delta_pose, dtype=float)

    if delta_pose.shape != (6,) or not np.isfinite(delta_pose).all():
        raise ValueError("delta_pose must contain 6 finite values")

    t_tcp_target = np.eye(4, dtype=float)
    t_tcp_target[:3, :3], _ = cv2.Rodrigues(delta_pose[3:6])
    t_tcp_target[:3, 3] = delta_pose[:3]

    t_base_target = t_base_tcp @ t_tcp_target
    target_rotvec, _ = cv2.Rodrigues(t_base_target[:3, :3])

    return [
        *t_base_target[:3, 3].tolist(),
        *target_rotvec.reshape(3).tolist()
    ]
