import numpy as np


def _rotation_matrix_to_angle_deg(
    rotation,
):
    rotation = np.asarray(
        rotation,
        dtype=float,
    )

    if rotation.shape != (
        3,
        3,
    ):
        raise ValueError(
            "rotation must be 3x3"
        )

    cos_angle = (
        float(
            np.trace(
                rotation
            )
        )
        - 1.0
    ) / 2.0

    cos_angle = float(
        np.clip(
            cos_angle,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(
                cos_angle
            )
        )
    )


def evaluate_sample_spread(
    rotations,
    translations,
):
    """
    評估 robot calibration samples
    的基本姿態分散程度。

    注意：
        這只是 sampling quality 指標，
        不是完整的 calibration accuracy。
    """

    if len(
        rotations
    ) != len(
        translations
    ):
        raise ValueError(
            "rotation and translation "
            "sample counts do not match"
        )

    if not rotations:
        return {
            "translation_span_m":
                0.0,

            "rotation_span_deg":
                0.0,
        }

    positions = np.vstack([
        np.asarray(
            t,
            dtype=float,
        ).reshape(
            1,
            3,
        )
        for t in translations
    ])

    translation_span_m = float(
        np.linalg.norm(
            np.max(
                positions,
                axis=0,
            )
            - np.min(
                positions,
                axis=0,
            )
        )
    )

    reference_rotation = np.asarray(
        rotations[0],
        dtype=float,
    )

    max_rotation_angle = 0.0

    for rotation in rotations[1:]:
        rotation = np.asarray(
            rotation,
            dtype=float,
        )

        relative_rotation = (
            reference_rotation.T
            @ rotation
        )

        angle = (
            _rotation_matrix_to_angle_deg(
                relative_rotation
            )
        )

        max_rotation_angle = max(
            max_rotation_angle,
            angle,
        )

    return {
        "translation_span_m":
            translation_span_m,

        "rotation_span_deg":
            float(
                max_rotation_angle
            ),
    }


def validate_hand_eye_inputs(
    R_gripper2base,
    t_gripper2base,
    R_target2cam,
    t_target2cam,
    minimum_samples=5,
):
    """
    驗證 hand-eye solver 輸入是否合法。

    這裡不強制設定「幾公分 / 幾度才合格」，
    避免把特定硬體條件寫死在純演算法層。
    """

    minimum_samples = int(
        minimum_samples
    )

    if minimum_samples < 3:
        raise ValueError(
            "minimum_samples must be "
            "at least 3"
        )

    lengths = [
        len(
            R_gripper2base
        ),
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
            "calibration sample counts "
            "do not match"
        )

    sample_count = lengths[0]

    if sample_count < minimum_samples:
        raise ValueError(
            f"Not enough calibration "
            f"samples: {sample_count}. "
            f"Need at least "
            f"{minimum_samples}"
        )

    spread = (
        evaluate_sample_spread(
            R_gripper2base,
            t_gripper2base,
        )
    )

    return {
        "sample_count":
            sample_count,

        "sample_spread":
            spread,
    }
