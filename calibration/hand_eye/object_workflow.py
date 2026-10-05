import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

from calibration.hand_eye.checktarget import find_calibration_targets
from calibration.hand_eye.objecttarget import estimate_object_target_pose_from_best_target
from calibration.hand_eye.solver import solve_hand_eye
from calibration.hand_eye.validation import validate_hand_eye_inputs

from services import (
    arm_service,
    camera_service,
)


# ============================================================
# Basic Helpers
# ============================================================

def _normalize_name(
    value,
    name,
):
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{name} must be a non-empty string"
        )

    return value.strip().lower()


# ============================================================
# Calibration Context
# ============================================================

def _get_calibration_context(
    arm_name,
    camera_name,
):
    """
    Eye-in-Hand 任意物件標定版本。

    驗證：
    1. arm 存在於 config.ARMS
    2. camera 存在於 config.CAMERAS
    3. camera mount.mode == wrist
    4. camera mount.arm_name == arm_name
    """

    arm_name = _normalize_name(
        arm_name,
        "arm_name",
    )

    camera_name = _normalize_name(
        camera_name,
        "camera_name",
    )

    if arm_name not in config.ARMS:
        raise ValueError(
            f"Arm '{arm_name}' not found in config.ARMS"
        )

    if camera_name not in config.CAMERAS:
        raise ValueError(
            f"Camera '{camera_name}' not found in config.CAMERAS"
        )

    arm_config = config.ARMS[
        arm_name
    ]

    camera_config = config.CAMERAS[
        camera_name
    ]

    mount = camera_config.get(
        "mount"
    )

    if not isinstance(
        mount,
        dict,
    ):
        raise ValueError(
            f"Camera '{camera_name}' mount config missing"
        )

    mode = (
        str(
            mount.get(
                "mode",
                "",
            )
        )
        .strip()
        .lower()
    )

    if mode != "wrist":
        raise ValueError(
            f"Camera '{camera_name}' is not wrist-mounted "
            f"(mode={mode!r})"
        )

    mounted_arm_name = (
        str(
            mount.get(
                "arm_name",
                "",
            )
        )
        .strip()
        .lower()
    )

    if mounted_arm_name != arm_name:
        raise ValueError(
            f"Camera '{camera_name}' is mounted on arm "
            f"'{mounted_arm_name}', not '{arm_name}'"
        )

    return (
        arm_name,
        camera_name,
        arm_config,
        camera_config,
    )


# ============================================================
# Arm Adapter
# ============================================================

def _get_arm_pose_value(
    arm_name,
):
    """
    透過 arm_service 取得目前 TCP pose：

        [x, y, z, rx, ry, rz]
    """

    result = (
        arm_service
        .get_arm_pose(
            arm_name=arm_name,
        )
    )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "invalid arm_service response"
        )

    if result.get("status") == "error":
        raise RuntimeError(
            result.get(
                "message",
                "get_arm_pose failed",
            )
        )

    data = result.get(
        "data",
        {},
    )

    arms = (
        data.get(
            "arms",
            [],
        )
        if isinstance(
            data,
            dict,
        )
        else []
    )

    for arm in arms:
        if (
            isinstance(
                arm,
                dict,
            )
            and arm.get(
                "arm_name"
            )
            == arm_name
        ):
            pose = arm.get(
                "pose"
            )

            if (
                isinstance(
                    pose,
                    (
                        list,
                        tuple,
                    ),
                )
                and len(pose) == 6
            ):
                return [
                    float(value)
                    for value in pose
                ]

    raise RuntimeError(
        f"Pose for arm '{arm_name}' unavailable"
    )


def _move_arm_pose(
    arm_name,
    pose,
    speed,
    acceleration,
):
    """
    所有 motion / safety 都交給 arm_service。
    """

    result = (
        arm_service
        .move_arm_pose(
            arm_name=arm_name,
            pose=pose,
            speed=speed,
            acceleration=acceleration,
            wait=True,
        )
    )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "invalid move_arm_pose response"
        )

    if result.get("status") == "error":
        raise RuntimeError(
            result.get(
                "message",
                "move_arm_pose failed",
            )
        )

    if result.get("result") is False:
        raise RuntimeError(
            result.get(
                "message",
                "arm did not reach target",
            )
        )


# ============================================================
# Robot Pose Conversion
# ============================================================

def _tcp_pose_to_rt(
    tcp_pose,
):
    """
    [x, y, z, rx, ry, rz]

    轉成：

        R_gripper2base
        t_gripper2base
    """

    pose = np.asarray(
        tcp_pose,
        dtype=float,
    )

    if pose.shape != (6,):
        raise ValueError(
            "tcp_pose must have 6 values"
        )

    R_gripper2base, _ = (
        cv2.Rodrigues(
            pose[
                3:
            ].reshape(
                3,
                1,
            )
        )
    )

    t_gripper2base = (
        pose[
            :3
        ].reshape(
            3,
            1,
        )
    )

    return (
        R_gripper2base,
        t_gripper2base,
    )


# ============================================================
# Perturbation
# ============================================================

def _valid_xyz(
    value,
):
    if (
        not isinstance(
            value,
            (
                list,
                tuple,
            ),
        )
        or len(value) != 3
    ):
        return False

    for item in value:
        try:
            number = float(
                item
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

        if not np.isfinite(
            number
        ):
            return False

    return True


def _rotation_matrix_from_axis_angle(
    axis,
    angle_rad,
):
    axis = np.asarray(
        axis,
        dtype=float,
    ).reshape(
        3,
    )

    norm = float(
        np.linalg.norm(
            axis
        )
    )

    if norm < 1e-9:
        return np.eye(
            3,
            dtype=float,
        )

    axis = axis / norm

    rvec = (
        axis
        * float(
            angle_rad
        )
    ).reshape(
        3,
        1,
    )

    rotation, _ = cv2.Rodrigues(
        rvec
    )

    return rotation


def _make_planned_pose(
    center_pose,
    index,
    target_robot_xyz,
    xyz_range_m,
    rotation_range_rad,
):
    """
    以物件中心點為中心，產生繞物件觀測的 TCP pose。

    重點：
    1. TCP 位置繞著 target_robot_xyz 做 orbit。
    2. TCP 姿態使用 R_orbit @ R_start_tcp，而不是直接對 rx/ry/rz 做加法。
    3. 這會讓相機比較像繞著物件中心轉，而不是 random rotate。
    """

    if (
        not isinstance(
            center_pose,
            (
                list,
                tuple,
            ),
        )
        or len(center_pose) != 6
    ):
        raise ValueError(
            "center_pose must be [x,y,z,rx,ry,rz]"
        )

    if not _valid_xyz(
        center_pose[:3]
    ):
        raise ValueError(
            "center_pose xyz values are invalid"
        )

    if not _valid_xyz(
        target_robot_xyz
    ):
        raise ValueError(
            "target_robot_xyz unavailable; cannot create object-centered planned pose"
        )

    pose = list(
        center_pose
    )

    target = np.asarray(
        target_robot_xyz,
        dtype=float,
    ).reshape(
        3,
    )

    tcp_position = np.asarray(
        center_pose[:3],
        dtype=float,
    ).reshape(
        3,
    )

    start_rvec = np.asarray(
        center_pose[3:6],
        dtype=float,
    ).reshape(
        3,
        1,
    )

    R_start_tcp, _ = cv2.Rodrigues(
        start_rvec
    )

    radius_vector = (
        tcp_position
        - target
    )

    radius_norm = float(
        np.linalg.norm(
            radius_vector
        )
    )

    if radius_norm < 1e-6:
        raise RuntimeError(
            "TCP position is too close to target center; cannot create orbit path"
        )

    # ------------------------------------------------------------
    # 固定 orbit pattern。
    #
    # 使用不同 base 軸做 orbit，讓 hand-eye solver 看到更有資訊量的旋轉。
    # ------------------------------------------------------------

    pattern = [
        # orbit_axis, orbit_scale
        ([0.0, 0.0, 1.0],  0.00),
        ([0.0, 0.0, 1.0],  1.00),
        ([0.0, 0.0, 1.0], -1.00),

        ([0.0, 1.0, 0.0],  1.00),
        ([0.0, 1.0, 0.0], -1.00),

        ([1.0, 0.0, 0.0],  1.00),
        ([1.0, 0.0, 0.0], -1.00),

        ([1.0, 1.0, 0.0],  0.85),
        ([1.0, 1.0, 0.0], -0.85),

        ([1.0, 0.0, 1.0],  0.85),
        ([1.0, 0.0, 1.0], -0.85),

        ([0.0, 1.0, 1.0],  0.85),
        ([0.0, 1.0, 1.0], -0.85),
    ]

    orbit_axis, orbit_scale = pattern[
        int(index) % len(pattern)
    ]

    orbit_angle = (
        float(rotation_range_rad)
        * float(orbit_scale)
    )

    R_orbit = _rotation_matrix_from_axis_angle(
        orbit_axis,
        orbit_angle,
    )

    # ------------------------------------------------------------
    # 1. TCP 位置繞物件中心旋轉
    # ------------------------------------------------------------

    new_radius_vector = (
        R_orbit
        @ radius_vector.reshape(
            3,
            1,
        )
    ).reshape(
        3,
    )

    new_position = (
        target
        + new_radius_vector
    )

    # ------------------------------------------------------------
    # 2. 限制最大位移，避免 orbit 一次跑太遠
    # ------------------------------------------------------------

    position_delta = (
        new_position
        - tcp_position
    )

    delta_norm = float(
        np.linalg.norm(
            position_delta
        )
    )

    max_position_delta = max(
        float(xyz_range_m) * 10.0,
        float(xyz_range_m),
    )

    if delta_norm > max_position_delta:
        position_delta = (
            position_delta
            / delta_norm
            * max_position_delta
        )

        new_position = (
            tcp_position
            + position_delta
        )

    pose[0] = float(
        new_position[0]
    )

    pose[1] = float(
        new_position[1]
    )

    pose[2] = float(
        new_position[2]
    )

    # ------------------------------------------------------------
    # 3. TCP 姿態同步做 orbit rotation
    #
    # 這比直接 rx += ... / ry += ... / rz += ... 更正確。
    # ------------------------------------------------------------

    R_new_tcp = (
        R_orbit
        @ R_start_tcp
    )

    new_rvec, _ = cv2.Rodrigues(
        R_new_tcp
    )

    pose[3] = float(
        new_rvec[0, 0]
    )

    pose[4] = float(
        new_rvec[1, 0]
    )

    pose[5] = float(
        new_rvec[2, 0]
    )

    return pose

# ============================================================
# Object Target Capture
# ============================================================

def _capture_object_target_pose(
    camera_name,
    *,
    locked_class_name=None,
    min_confidence=0.50,
    min_point_count=50,
    min_box_area_ratio=0.003,
    min_distance_m=0.10,
    max_distance_m=2.50,
    max_candidates=10,
    include_robot_xyz=True,
):
    """
    使用 checktarget.py + objecttarget.py 取得任意物件 target pose。

    回傳格式與 checkerboard.py 類似：

        {
            "R_target2cam": np.ndarray(3, 3),
            "t_target2cam": np.ndarray(3, 1),
            ...
        }

    locked_class_name:
        第一筆 sample 選定物件類別後，
        後續 sample 只接受同一 class，避免中途換目標。
    """

    allowed_classes = None

    if locked_class_name:
        allowed_classes = [
            locked_class_name
        ]

    check_result = find_calibration_targets(
        camera_name=camera_name,
        min_confidence=min_confidence,
        min_point_count=min_point_count,
        min_box_area_ratio=min_box_area_ratio,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        max_candidates=max_candidates,
        include_robot_xyz=include_robot_xyz,
        allowed_classes=allowed_classes,
    )

    best_target = check_result.get(
        "best_target"
    )

    if not isinstance(
        best_target,
        dict,
    ):
        raise RuntimeError(
            "no suitable arbitrary object target found"
        )

    object_pose = estimate_object_target_pose_from_best_target(
        check_result,
        min_points=min_point_count,
    )

    return {
        "check_result":
            check_result,

        "best_target":
            best_target,

        "object_pose":
            object_pose,

        "class_name":
            object_pose.get(
                "class_name"
            ),

        "R_target2cam":
            object_pose[
                "R_target2cam"
            ],

        "t_target2cam":
            object_pose[
                "t_target2cam"
            ],
    }


# ============================================================
# Serialization
# ============================================================

def _matrix_to_list(
    value,
):
    if value is None:
        return None

    return (
        np.asarray(
            value,
            dtype=float,
        )
        .tolist()
    )


def _sample_to_serializable(
    sample,
):
    return {
        "index":
            sample.get(
                "index"
            ),

        "class_name":
            sample.get(
                "class_name"
            ),

        "tcp_pose":
            sample.get(
                "tcp_pose"
            ),

        "R_gripper2base":
            _matrix_to_list(
                sample.get(
                    "R_gripper2base"
                )
            ),

        "t_gripper2base":
            _matrix_to_list(
                sample.get(
                    "t_gripper2base"
                )
            ),

        "R_target2cam":
            _matrix_to_list(
                sample.get(
                    "R_target2cam"
                )
            ),

        "t_target2cam":
            _matrix_to_list(
                sample.get(
                    "t_target2cam"
                )
            ),

        "target_quality":
            sample.get(
                "target_quality"
            ),

        "target_warnings":
            sample.get(
                "target_warnings"
            ),
    }


# ============================================================
# Public Workflow
# ============================================================

def calibrate_eye_in_hand(
    arm_name,
    camera_name,
    *,
    sample_count=15,
    max_attempts=40,
    xyz_range_m=0.025,
    rotation_range_rad=0.4,
    speed=None,
    acceleration=None,
    settle_time_sec=0.5,
    method="tsai",
    return_to_start=True,
    min_confidence=0.50,
    min_point_count=50,
    min_box_area_ratio=0.003,
    min_distance_m=0.10,
    max_distance_m=2.50,
    max_candidates=10,
    lock_class_after_first_sample=True,
):
    """
    執行 Eye-in-Hand 任意物件手眼標定。

    注意：
        1. 不寫入 config。
        2. 不覆蓋 camera mount matrix。
        3. 只輸出 samples 與 solver 結果。
        4. 此 function 發生錯誤時直接 raise。
    """

    start_pose = None
    locked_class_name = None
    locked_target_robot_xyz = None

    (
        arm_name,
        camera_name,
        arm_config,
        _camera_config,
    ) = (
        _get_calibration_context(
            arm_name,
            camera_name,
        )
    )

    sample_count = int(
        sample_count
    )

    max_attempts = int(
        max_attempts
    )

    xyz_range_m = float(
        xyz_range_m
    )

    rotation_range_rad = float(
        rotation_range_rad
    )

    settle_time_sec = float(
        settle_time_sec
    )

    if sample_count < 5:
        raise ValueError(
            "sample_count must be >= 5"
        )

    if max_attempts < sample_count:
        raise ValueError(
            "max_attempts must be >= sample_count"
        )

    motion = arm_config.get(
        "motion",
        {},
    )

    if speed is None:
        speed = motion.get(
            "speed"
        )

    if acceleration is None:
        acceleration = (
            motion.get(
                "acceleration"
            )
        )

    if speed is None:
        raise ValueError(
            "arm motion.speed not configured"
        )

    if acceleration is None:
        raise ValueError(
            "arm motion.acceleration not configured"
        )

    speed = float(
        speed
    )

    acceleration = float(
        acceleration
    )

    try:
        # ====================================================
        # 1. Start camera
        # ====================================================

        camera_service.start_camera(
            camera_name
        )

        # ====================================================
        # 2. Start Pose
        # ====================================================

        start_pose = (
            _get_arm_pose_value(
                arm_name
            )
        )

        # ====================================================
        # 3. Object Target Pre-check
        # ====================================================

        precheck = _capture_object_target_pose(
            camera_name=camera_name,
            locked_class_name=None,
            min_confidence=min_confidence,
            min_point_count=min_point_count,
            min_box_area_ratio=min_box_area_ratio,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            max_candidates=max_candidates,
            include_robot_xyz=True,
        )

        locked_class_name = (
            precheck.get(
                "class_name"
            )
        )

        if not locked_class_name:
            raise RuntimeError(
                "failed to lock arbitrary object class"
            )

        best_target = (
            precheck.get(
                "best_target"
            )
            or {}
        )

        locked_target_robot_xyz = (
            best_target.get(
                "robot_xyz"
            )
        )

        if not _valid_xyz(
            locked_target_robot_xyz
        ):
            raise RuntimeError(
                "failed to lock arbitrary object robot_xyz center"
            )

        print(
            f"[object_workflow] locked target "
            f"class={locked_class_name} "
            f"robot_xyz={locked_target_robot_xyz}"
        )

        # ====================================================
        # 4. Calibration Samples
        # ====================================================

        R_gripper2base: List[np.ndarray] = []
        t_gripper2base: List[np.ndarray] = []

        R_target2cam: List[np.ndarray] = []
        t_target2cam: List[np.ndarray] = []

        samples: List[Dict[str, Any]] = []
        rejected_samples: List[Dict[str, Any]] = []

        attempts = 0

        while (
            len(
                R_gripper2base
            )
            < sample_count
            and attempts
            < max_attempts
        ):
            attempts += 1

            target_pose = (
                _make_planned_pose(
                    center_pose=start_pose,
                    index=attempts - 1,
                    target_robot_xyz=locked_target_robot_xyz,
                    xyz_range_m=xyz_range_m,
                    rotation_range_rad=rotation_range_rad,
                )
            )

            try:
                # -----------------------------------------------
                # Move Robot
                # -----------------------------------------------

                _move_arm_pose(
                    arm_name=arm_name,
                    pose=target_pose,
                    speed=speed,
                    acceleration=acceleration,
                )

                if settle_time_sec > 0:
                    time.sleep(
                        settle_time_sec
                    )

                # -----------------------------------------------
                # Get Actual Robot Pose
                # -----------------------------------------------

                actual_pose = (
                    _get_arm_pose_value(
                        arm_name
                    )
                )

                # -----------------------------------------------
                # Capture Object Target Pose
                # -----------------------------------------------

                capture = _capture_object_target_pose(
                    camera_name=camera_name,
                    locked_class_name=(
                        locked_class_name
                        if lock_class_after_first_sample
                        else None
                    ),
                    min_confidence=min_confidence,
                    min_point_count=min_point_count,
                    min_box_area_ratio=min_box_area_ratio,
                    min_distance_m=min_distance_m,
                    max_distance_m=max_distance_m,
                    max_candidates=max_candidates,
                    include_robot_xyz=True,
                )

                object_pose = capture[
                    "object_pose"
                ]

                # -----------------------------------------------
                # Convert Robot Pose
                # -----------------------------------------------

                (
                    R_robot,
                    t_robot,
                ) = (
                    _tcp_pose_to_rt(
                        actual_pose
                    )
                )

                R_obj = np.asarray(
                    capture[
                        "R_target2cam"
                    ],
                    dtype=float,
                ).reshape(
                    3,
                    3,
                )

                t_obj = np.asarray(
                    capture[
                        "t_target2cam"
                    ],
                    dtype=float,
                ).reshape(
                    3,
                    1,
                )

                # -----------------------------------------------
                # Save Sample
                # -----------------------------------------------

                R_gripper2base.append(
                    R_robot
                )

                t_gripper2base.append(
                    t_robot
                )

                R_target2cam.append(
                    R_obj
                )

                t_target2cam.append(
                    t_obj
                )

                samples.append({
                    "index":
                        len(samples) + 1,

                    "class_name":
                        capture.get(
                            "class_name"
                        ),

                    "tcp_pose":
                        actual_pose,

                    "R_gripper2base":
                        R_robot,

                    "t_gripper2base":
                        t_robot,

                    "R_target2cam":
                        R_obj,

                    "t_target2cam":
                        t_obj,

                    "target_quality":
                        object_pose.get(
                            "quality"
                        ),

                    "target_warnings":
                        object_pose.get(
                            "warnings"
                        ),
                })

                print(
                    f"[object_workflow] sample {len(samples)} OK "
                    f"class={capture.get('class_name')} "
                    f"tcp_pose={actual_pose} "
                    f"target_t={t_obj.reshape(3).tolist()}"
                )

            except Exception as exc:
                rejected_samples.append({
                    "attempt":
                        attempts,

                    "error_type":
                        type(exc).__name__,

                    "message":
                        str(exc),
                })
                continue

        if len(R_gripper2base) < sample_count:
            recent_rejections = rejected_samples[-8:]
            raise RuntimeError(
                f"not enough valid object calibration samples: "
                f"{len(R_gripper2base)} / {sample_count}, "
                f"attempts={attempts}, "
                f"recent_rejections={recent_rejections}"
            )

        # ====================================================
        # 5. Validation
        # ====================================================

        validation_result = (
            validate_hand_eye_inputs(
                R_gripper2base=
                    R_gripper2base,
                t_gripper2base=
                    t_gripper2base,
                R_target2cam=
                    R_target2cam,
                t_target2cam=
                    t_target2cam,
                minimum_samples=
                    sample_count,
            )
        )

        # ====================================================
        # 6. Hand-Eye Solve
        # ====================================================

        solve_result = (
            solve_hand_eye(
                R_gripper2base=
                    R_gripper2base,
                t_gripper2base=
                    t_gripper2base,
                R_target2cam=
                    R_target2cam,
                t_target2cam=
                    t_target2cam,
                method=method,
            )
        )

        T_tcp_camera = (
            solve_result[
                "T_tcp_camera"
            ]
        )

        # ====================================================
        # 7. Return Result
        # ====================================================

        return {
            "arm_name":
                arm_name,

            "camera_name":
                camera_name,

            "arm_driver":
                config.ARMS[
                    arm_name
                ].get(
                    "driver"
                ),

            "camera_driver":
                config.CAMERAS[
                    camera_name
                ].get(
                    "driver"
                ),

            "mount_mode":
                "wrist",

            "target_type":
                "arbitrary_object_point_cloud",

            "locked_class_name":
                locked_class_name,

            "sample_count":
                len(
                    R_gripper2base
                ),

            "attempt_count":
                attempts,

            "rejected_count":
                len(
                    rejected_samples
                ),

            "rejected_samples":
                rejected_samples,

            "method":
                solve_result[
                    "method"
                ],

            "T_tcp_camera":
                T_tcp_camera.tolist(),

            "R_cam2gripper":
                solve_result[
                    "R_cam2gripper"
                ].tolist(),

            "t_cam2gripper":
                solve_result[
                    "t_cam2gripper"
                ].tolist(),

            "sample_spread":
                validation_result.get(
                    "sample_spread"
                ),

            "samples":
                [
                    _sample_to_serializable(
                        sample
                    )
                    for sample in samples
                ],

            "write_config":
                False,
        }

    finally:
        # ====================================================
        # Return To Start Pose
        # ====================================================

        if (
            return_to_start
            and start_pose
            is not None
        ):
            try:
                motion = (
                    arm_config.get(
                        "motion",
                        {},
                    )
                )

                return_speed = float(
                    motion.get(
                        "speed",
                        0.05,
                    )
                )

                return_acceleration = float(
                    motion.get(
                        "acceleration",
                        0.05,
                    )
                )

                _move_arm_pose(
                    arm_name=arm_name,
                    pose=start_pose,
                    speed=return_speed,
                    acceleration=return_acceleration,
                )

            except Exception:
                # 避免 return-to-start 的錯誤覆蓋原本 calibration error。
                pass