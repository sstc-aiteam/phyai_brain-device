from calibration.hand_eye import object_workflow
from utils import configmanager, response


MODULE = "calibration"


def _service_error(action, exc):
    return response.error(
        MODULE,
        action,
        error=exc,
        error_type=type(exc).__name__,
    )


# ============================================================
# Arbitrary Object Eye-in-Hand Calibration
# ============================================================

def calibrate_eye_in_hand(
    arm_name,
    camera_name,
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
    任意物件 Eye-in-Hand 手眼校正服務入口。

    流程：
        1. 偵測並篩選可作為標定目標的任意物件
        2. 由任意物件 segmentation point cloud 建立 target pose
        3. 以物件中心產生 object-centered planned poses
        4. 收集多組 robot pose 與 object target pose
        5. 呼叫 hand-eye solver 計算 T_tcp_camera
        6. 將 T_tcp_camera 寫入 config.CAMERAS[camera_name]["mount"]["T_matrix"]
        7. 回傳校正結果與 config 寫入結果
    """

    action = "calibrate_eye_in_hand"

    try:
        calibration_result = object_workflow.calibrate_eye_in_hand(
            arm_name=arm_name,
            camera_name=camera_name,
            sample_count=sample_count,
            max_attempts=max_attempts,
            xyz_range_m=xyz_range_m,
            rotation_range_rad=rotation_range_rad,
            speed=speed,
            acceleration=acceleration,
            settle_time_sec=settle_time_sec,
            method=method,
            return_to_start=return_to_start,
            min_confidence=min_confidence,
            min_point_count=min_point_count,
            min_box_area_ratio=min_box_area_ratio,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            max_candidates=max_candidates,
            lock_class_after_first_sample=lock_class_after_first_sample,
        )

        T_tcp_camera = calibration_result.get("T_tcp_camera")

        if T_tcp_camera is None:
            raise RuntimeError(
                "calibration result missing T_tcp_camera"
            )

        config_result = configmanager.save_camera_mount_matrix(
            camera_name=camera_name,
            matrix=T_tcp_camera,
        )

        calibration_result["write_config"] = True
        calibration_result["config_update"] = config_result

        return response.success(
            MODULE,
            action,
            result=True,
            data=calibration_result,
        )

    except Exception as exc:
        return _service_error(
            action,
            exc,
        )