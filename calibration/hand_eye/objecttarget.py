#segmentation point cloud → 移除 NaN / 離群點 → PCA 估計 3D 主軸 → 平面擬合估計 normal → 組成 X/Y/Z 三軸 → 輸出 R_target2cam / t_target2cam

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def _valid_xyz(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False

    for item in value:
        if _as_float(item) is None:
            return False

    return True


def _normalize_vector(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm < 1e-9:
        raise RuntimeError(f"{name} vector norm too small")

    return vector / norm


def _safe_cross(a: np.ndarray, b: np.ndarray, name: str) -> np.ndarray:
    result = np.cross(a, b)
    return _normalize_vector(result, name)


def _ensure_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)

    if rotation.shape != (3, 3):
        raise RuntimeError("rotation matrix must be 3x3")

    if not np.all(np.isfinite(rotation)):
        raise RuntimeError("rotation matrix contains non-finite values")

    det = float(np.linalg.det(rotation))
    if det < 0:
        # 避免左手座標系
        rotation[:, 2] *= -1.0
        det = float(np.linalg.det(rotation))

    if abs(det - 1.0) > 0.08:
        raise RuntimeError(f"invalid rotation matrix determinant: {det}")

    return rotation


def _rotation_matrix_to_rvec(rotation: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
    return rvec.reshape(3, 1)


# ============================================================
# Point cloud extraction
# ============================================================

def _extract_point_cloud_from_detection(
    target: Dict[str, Any],
    *,
    point_cloud_key_candidates: Optional[List[str]] = None,
) -> np.ndarray:
    """
    從 checktarget.py 回傳的 best_target 取出 segmentation point cloud。

    支援兩種輸入：
    1. target 本身含 object_points / object_point_cloud / segmentation_point_cloud
    2. target["detection"] 中含上述欄位

    預期格式：
        [
            [x, y, z],
            [x, y, z],
            ...
        ]

    單位：
        meter, camera frame
    """

    if point_cloud_key_candidates is None:
        point_cloud_key_candidates = [
            "segmentation_point_cloud",
            "object_point_cloud",
            "object_points",
            "points_3d",
            "point_cloud",
        ]

    containers = [
        target,
        target.get("detection") if isinstance(target.get("detection"), dict) else None,
    ]

    for container in containers:
        if not isinstance(container, dict):
            continue

        for key in point_cloud_key_candidates:
            value = container.get(key)
            if value is None:
                continue

            points = np.asarray(value, dtype=float)

            if points.ndim == 2 and points.shape[1] >= 3:
                return points[:, :3].astype(float)

            if points.ndim == 2 and points.shape[0] >= 3:
                # 防止有人傳成 3xN
                transposed = points.T
                if transposed.ndim == 2 and transposed.shape[1] >= 3:
                    return transposed[:, :3].astype(float)

    raise RuntimeError(
        "segmentation point cloud unavailable in target/detection. "
        "Expected one of: segmentation_point_cloud, object_point_cloud, "
        "object_points, points_3d, point_cloud"
    )


def _clean_point_cloud(
    points: np.ndarray,
    *,
    min_points: int = 80,
    max_points: int = 5000,
    mad_scale: float = 3.5,
    min_mad_m: float = 0.003,
) -> np.ndarray:
    """
    清理 point cloud：
    1. 移除 NaN / Inf
    2. 移除全零點
    3. 使用 median absolute deviation 移除離群點
    4. 點太多時均勻抽樣
    """

    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError("point cloud must have shape [N, 3]")

    finite_mask = np.all(np.isfinite(points), axis=1)
    points = points[finite_mask]

    nonzero_mask = np.linalg.norm(points, axis=1) > 1e-9
    points = points[nonzero_mask]

    if len(points) < int(min_points):
        raise RuntimeError(
            f"not enough valid point cloud points: {len(points)} < {min_points}"
        )

    center = np.median(points, axis=0)
    distance = np.linalg.norm(points - center.reshape(1, 3), axis=1)

    median_distance = float(np.median(distance))
    mad = float(np.median(np.abs(distance - median_distance)))
    threshold = median_distance + float(mad_scale) * max(float(min_mad_m), mad)

    inlier_mask = distance <= threshold
    points = points[inlier_mask]

    if len(points) < int(min_points):
        raise RuntimeError(
            f"not enough inlier point cloud points: {len(points)} < {min_points}"
        )

    if max_points is not None and len(points) > int(max_points):
        indices = np.linspace(0, len(points) - 1, int(max_points)).astype(int)
        points = points[indices]

    return points.astype(float)


# ============================================================
# Geometry estimation
# ============================================================

def _estimate_centroid(
    points: np.ndarray,
    fallback_center: Optional[Any] = None,
) -> np.ndarray:
    """
    優先使用 point cloud centroid。
    若 target 有 camera_xyz，也會在 debug 裡可做比對；
    但這裡仍以 point cloud median 作為較穩定中心。
    """

    if len(points) == 0:
        if _valid_xyz(fallback_center):
            return np.asarray(fallback_center, dtype=float).reshape(3, 1)

        raise RuntimeError("cannot estimate centroid from empty points")

    centroid = np.median(points, axis=0).reshape(3, 1)

    if not np.all(np.isfinite(centroid)):
        raise RuntimeError("centroid contains non-finite values")

    return centroid


def _estimate_pca_axes(points: np.ndarray) -> Dict[str, Any]:
    """
    使用 3D PCA 估計主軸。

    回傳：
        eigenvalues descending
        eigenvectors columns:
            v0: 最大變異方向
            v1: 次大變異方向
            v2: 最小變異方向
    """

    points = np.asarray(points, dtype=float)
    mean = np.mean(points, axis=0)
    centered = points - mean.reshape(1, 3)

    covariance = (centered.T @ centered) / max(1, len(points) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for index in range(3):
        eigenvectors[:, index] = _normalize_vector(
            eigenvectors[:, index],
            f"pca_axis_{index}",
        )

    major = eigenvectors[:, 0]
    minor = eigenvectors[:, 1]
    normal_from_pca = eigenvectors[:, 2]

    major_value = float(eigenvalues[0])
    minor_value = float(eigenvalues[1])
    normal_value = float(eigenvalues[2])

    axis_ratio = major_value / max(minor_value, 1e-12)
    planarity_ratio = minor_value / max(normal_value, 1e-12)

    return {
        "center": mean,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "major_axis": major,
        "minor_axis": minor,
        "normal_axis": normal_from_pca,
        "axis_ratio": float(axis_ratio),
        "planarity_ratio": float(planarity_ratio),
    }


def _fit_plane_normal_svd(points: np.ndarray) -> Dict[str, Any]:
    """
    以 SVD 擬合平面。

    normal = 最小 singular value 對應方向。
    """

    points = np.asarray(points, dtype=float)

    centroid = np.mean(points, axis=0)
    centered = points - centroid.reshape(1, 3)

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    normal = vh[-1, :]
    normal = _normalize_vector(normal, "plane_normal")

    residuals = np.abs(centered @ normal.reshape(3, 1)).reshape(-1)
    mean_residual = float(np.mean(residuals))
    median_residual = float(np.median(residuals))
    max_residual = float(np.max(residuals))

    return {
        "normal": normal,
        "centroid": centroid,
        "singular_values": singular_values,
        "mean_residual_m": mean_residual,
        "median_residual_m": median_residual,
        "max_residual_m": max_residual,
    }


def _orient_normal_toward_camera(normal: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Camera frame 中，物件 centroid 通常 z > 0。

    為了讓 target Z 軸方向穩定，將 normal 朝向 camera。
    從物件看向 camera origin 的方向為 -centroid。
    """

    normal = _normalize_vector(normal, "normal")
    centroid = np.asarray(centroid, dtype=float).reshape(3)

    to_camera = -centroid
    if np.linalg.norm(to_camera) < 1e-9:
        return normal

    if float(np.dot(normal, to_camera)) < 0.0:
        normal = -normal

    return normal


def _make_rotation_from_axes(
    *,
    x_axis_hint: np.ndarray,
    z_axis_hint: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """
    組成 target frame：

        Z_target: 平面 normal，方向朝 camera
        X_target: PCA 主軸投影到與 Z 垂直的平面
        Y_target: Z x X

    OpenCV / robotics 常用 rotation matrix 欄向量代表 target axes in camera frame：

        R_target2cam = [x_cam, y_cam, z_cam]
    """

    z_axis = _orient_normal_toward_camera(
        z_axis_hint,
        centroid=centroid,
    )

    x_hint = _normalize_vector(x_axis_hint, "x_axis_hint")

    # 將 x 軸投影到 normal 平面上，避免 x 與 z 不正交。
    x_axis = x_hint - float(np.dot(x_hint, z_axis)) * z_axis

    if float(np.linalg.norm(x_axis)) < 1e-6:
        raise RuntimeError("principal axis is parallel to plane normal")

    x_axis = _normalize_vector(x_axis, "x_axis")

    y_axis = _safe_cross(z_axis, x_axis, "y_axis")

    # 重新修正 x，確保正交。
    x_axis = _safe_cross(y_axis, z_axis, "x_axis_recomputed")

    rotation = np.column_stack([
        x_axis,
        y_axis,
        z_axis,
    ])

    return _ensure_rotation_matrix(rotation)


# ============================================================
# Public API
# ============================================================

def estimate_object_target_pose(
    target: Dict[str, Any],
    *,
    min_points: int = 80,
    max_points: int = 5000,
    mad_scale: float = 3.5,
    min_mad_m: float = 0.003,
    min_planarity_ratio: float = 4.0,
    min_axis_ratio: float = 1.10,
) -> Dict[str, Any]:
    """
    由任意物件 segmentation point cloud 估測 target pose。

    input:
        target:
            checktarget.py 回傳的 best_target

    output:
        {
            "R_target2cam": np.ndarray(3, 3),
            "t_target2cam": np.ndarray(3, 1),
            "rvec_target2cam": np.ndarray(3, 1),
            "target_type": "arbitrary_object_point_cloud",
            ...
        }

    注意：
        這個 target pose 是「物件幾何座標系」。
        它不是 checkerboard 的真實已知 CAD 座標系。
        因此後續要靠多次觀測與驗證確認穩定性。
    """

    if not isinstance(target, dict):
        raise ValueError("target must be a dictionary")

    if not target.get("eligible", False):
        raise RuntimeError("target is not eligible for object target pose estimation")

    if not target.get("has_segmentation_point_cloud", False):
        # 為了兼容舊版 checktarget.py，也會嘗試實際取點雲；
        # 但語意上要求必須有 segmentation point cloud。
        usable_features = target.get("usable_features") or []
        if "segmentation_point_cloud" not in usable_features:
            raise RuntimeError("target requires segmentation_point_cloud")

    raw_points = _extract_point_cloud_from_detection(target)

    points = _clean_point_cloud(
        raw_points,
        min_points=min_points,
        max_points=max_points,
        mad_scale=mad_scale,
        min_mad_m=min_mad_m,
    )

    centroid = _estimate_centroid(
        points,
        fallback_center=target.get("camera_xyz"),
    )

    pca = _estimate_pca_axes(points)
    plane = _fit_plane_normal_svd(points)

    axis_ratio = float(pca["axis_ratio"])
    planarity_ratio = float(pca["planarity_ratio"])

    warnings: List[str] = []

    if axis_ratio < float(min_axis_ratio):
        warnings.append("weak_3d_principal_axis")

    if planarity_ratio < float(min_planarity_ratio):
        warnings.append("weak_planarity")

    R_target2cam = _make_rotation_from_axes(
        x_axis_hint=pca["major_axis"],
        z_axis_hint=plane["normal"],
        centroid=centroid.reshape(3),
    )

    t_target2cam = centroid.reshape(3, 1)
    rvec_target2cam = _rotation_matrix_to_rvec(R_target2cam)

    class_name = target.get("class_name")
    detection = target.get("detection") or {}

    return {
        "target_type": "arbitrary_object_point_cloud",
        "class_name": class_name,
        "R_target2cam": R_target2cam,
        "t_target2cam": t_target2cam,
        "rvec_target2cam": rvec_target2cam,
        "point_count_raw": int(len(raw_points)),
        "point_count_used": int(len(points)),
        "center_camera_xyz": t_target2cam.reshape(3).tolist(),
        "rotation_det": float(np.linalg.det(R_target2cam)),
        "warnings": warnings,
        "quality": {
            "axis_ratio": axis_ratio,
            "planarity_ratio": planarity_ratio,
            "plane_mean_residual_m": float(plane["mean_residual_m"]),
            "plane_median_residual_m": float(plane["median_residual_m"]),
            "plane_max_residual_m": float(plane["max_residual_m"]),
            "pca_eigenvalues": [
                float(value)
                for value in pca["eigenvalues"]
            ],
        },
        "features": {
            "x_axis_camera": R_target2cam[:, 0].reshape(3).tolist(),
            "y_axis_camera": R_target2cam[:, 1].reshape(3).tolist(),
            "z_axis_camera": R_target2cam[:, 2].reshape(3).tolist(),
            "plane_normal_camera": plane["normal"].reshape(3).tolist(),
            "pca_major_axis_camera": pca["major_axis"].reshape(3).tolist(),
            "pca_minor_axis_camera": pca["minor_axis"].reshape(3).tolist(),
            "pca_normal_axis_camera": pca["normal_axis"].reshape(3).tolist(),
        },
        "source": {
            "confidence": target.get("confidence"),
            "distance_m": target.get("distance_m"),
            "camera_xyz": target.get("camera_xyz"),
            "robot_xyz": target.get("robot_xyz"),
            "object_point_count": target.get("object_point_count"),
            "yaw_deg_2d": target.get("yaw_deg"),
            "axis_ratio_2d": target.get("axis_ratio"),
            "depth_method": target.get("depth_method"),
            "depth_status": target.get("depth_status"),
            "box": detection.get("box"),
        },
    }


def estimate_object_target_pose_from_best_target(
    checktarget_result: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    方便直接吃 find_calibration_targets() 的回傳結果。
    """

    if not isinstance(checktarget_result, dict):
        raise ValueError("checktarget_result must be a dictionary")

    best_target = checktarget_result.get("best_target")

    if not isinstance(best_target, dict):
        raise RuntimeError("best_target unavailable")

    return estimate_object_target_pose(
        best_target,
        **kwargs,
    )


def object_target_pose_to_serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    將 numpy array 轉成 list，方便 response / print / json。
    """

    output = {}

    for key, value in result.items():
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, dict):
            nested = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, np.ndarray):
                    nested[nested_key] = nested_value.tolist()
                else:
                    nested[nested_key] = nested_value
            output[key] = nested
        else:
            output[key] = value

    return output