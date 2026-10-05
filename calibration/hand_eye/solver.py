import cv2
import numpy as np


_HAND_EYE_METHODS = {
    "tsai":
        cv2.CALIB_HAND_EYE_TSAI,

    "park":
        cv2.CALIB_HAND_EYE_PARK,

    "horaud":
        cv2.CALIB_HAND_EYE_HORAUD,

    "andreff":
        cv2.CALIB_HAND_EYE_ANDREFF,

    "daniilidis":
        cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _normalize_method(
    method,
):
    if not isinstance(
        method,
        str,
    ) or not method.strip():
        raise ValueError(
            "method must be a "
            "non-empty string"
        )

    method = (
        method
        .strip()
        .lower()
    )

    if method not in _HAND_EYE_METHODS:
        raise ValueError(
            f"Unsupported hand-eye method: "
            f"{method}. Supported methods: "
            f"{', '.join(_HAND_EYE_METHODS)}"
        )

    return method


def build_transform(
    rotation,
    translation,
):
    rotation = np.asarray(
        rotation,
        dtype=float,
    )

    translation = np.asarray(
        translation,
        dtype=float,
    )

    if rotation.shape != (
        3,
        3,
    ):
        raise ValueError(
            "rotation must be 3x3"
        )

    if translation.size != 3:
        raise ValueError(
            "translation must contain "
            "3 values"
        )

    transform = np.eye(
        4,
        dtype=float,
    )

    transform[
        :3,
        :3,
    ] = rotation

    transform[
        :3,
        3,
    ] = translation.reshape(
        3,
    )

    return transform


def solve_hand_eye(
    R_gripper2base,
    t_gripper2base,
    R_target2cam,
    t_target2cam,
    method="tsai",
):
    """
    Eye-in-Hand 手眼標定。

    輸入：
        R_gripper2base
        t_gripper2base
        R_target2cam
        t_target2cam

    輸出：
        T_tcp_camera

    OpenCV calibrateHandEye 回傳的是：
        R_cam2gripper
        t_cam2gripper

    在目前 Robot Brain 的 frame 定義中：
        gripper == tcp

    因此：
        T_tcp_camera
    """

    method = (
        _normalize_method(
            method
        )
    )

    sample_count = len(
        R_gripper2base
    )

    lengths = [
        sample_count,
        len(
            t_gripper2base
        ),
        len(
            R_target2cam
        ),
        len(
            t_target2cam
        ),
    ]

    if len(
        set(
            lengths
        )
    ) != 1:
        raise ValueError(
            "hand-eye input sample "
            "counts do not match"
        )

    if sample_count < 3:
        raise ValueError(
            "at least 3 samples are required"
        )

    (
        R_cam2gripper,
        t_cam2gripper,
    ) = (
        cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=
                _HAND_EYE_METHODS[
                    method
                ],
        )
    )

    T_tcp_camera = (
        build_transform(
            R_cam2gripper,
            t_cam2gripper,
        )
    )

    if not np.all(
        np.isfinite(
            T_tcp_camera
        )
    ):
        raise RuntimeError(
            "hand-eye result contains "
            "non-finite values"
        )

    return {
        "method":
            method,

        "R_cam2gripper":
            R_cam2gripper,

        "t_cam2gripper":
            t_cam2gripper.reshape(
                3,
                1,
            ),

        "T_tcp_camera":
            T_tcp_camera,
    }