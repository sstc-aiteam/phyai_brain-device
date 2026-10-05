import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from services import perception_service


DEFAULT_REJECTED_CLASSES = {
    "person",
    "hand",
    "face",
    "human",
}


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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_frame_shape(camera_rgb: Any) -> Tuple[int, int]:
    """Return (height, width)."""

    shape = getattr(camera_rgb, "shape", None)
    if not shape or len(shape) < 2:
        raise RuntimeError("camera_rgb shape unavailable")

    return int(shape[0]), int(shape[1])


def _box_area_ratio(
    detection: Dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> float:
    box = detection.get("box") or {}

    x1 = _as_float(box.get("x1"), 0.0)
    y1 = _as_float(box.get("y1"), 0.0)
    x2 = _as_float(box.get("x2"), 0.0)
    y2 = _as_float(box.get("y2"), 0.0)

    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))

    frame_area = max(1.0, float(frame_width * frame_height))
    return float((width * height) / frame_area)


def _touches_image_edge(
    detection: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    edge_margin_px: int,
) -> bool:
    box = detection.get("box") or {}

    x1 = _as_float(box.get("x1"), 0.0)
    y1 = _as_float(box.get("y1"), 0.0)
    x2 = _as_float(box.get("x2"), 0.0)
    y2 = _as_float(box.get("y2"), 0.0)

    return (
        x1 <= edge_margin_px
        or y1 <= edge_margin_px
        or x2 >= frame_width - 1 - edge_margin_px
        or y2 >= frame_height - 1 - edge_margin_px
    )


def _valid_xyz(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False

    for item in value:
        parsed = _as_float(item)
        if parsed is None:
            return False

    return True


def _normalize_class_set(values: Optional[Iterable[str]]) -> Optional[set]:
    if values is None:
        return None

    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }


# ============================================================
# Target quality evaluation
# ============================================================

def evaluate_detection_as_target(
    detection: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    *,
    allowed_classes: Optional[Iterable[str]] = None,
    rejected_classes: Optional[Iterable[str]] = None,
    min_confidence: float = 0.60,
    min_point_count: int = 80,
    min_box_area_ratio: float = 0.005,
    min_distance_m: float = 0.15,
    max_distance_m: float = 2.00,
    edge_margin_px: int = 8,
) -> Dict[str, Any]:
    """
    判斷單一 detection 是否適合作為任意物件手眼標定候選物。

    設計原則：
    1. 任意物件標定需要 3D 幾何特徵。
    2. 只有 center pixel + depth 不足以建立穩定 target pose。
    3. 因此必須同時具備 segmentation polygon 與 segmentation point cloud。
    """

    allowed_classes = _normalize_class_set(allowed_classes)
    rejected_classes = (
        _normalize_class_set(rejected_classes)
        or set(DEFAULT_REJECTED_CLASSES)
    )

    reasons: List[str] = []
    warnings: List[str] = []
    score = 100.0

    class_name = str(detection.get("class_name") or "").strip().lower()
    confidence = _as_float(detection.get("confidence"), 0.0) or 0.0
    distance_m = _as_float(detection.get("distance_m"))
    object_point_count = _as_int(detection.get("object_point_count"), 0)

    segmentation_points = detection.get("segmentation_points") or []
    has_segmentation = len(segmentation_points) >= 3
    has_object_point_cloud = object_point_count >= int(min_point_count)
    has_segmentation_point_cloud = has_segmentation and has_object_point_cloud

    box_area_ratio = _box_area_ratio(
        detection=detection,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    axis_ratio = _as_float(
        detection.get("pca_axis_ratio"),
        _as_float((detection.get("orientation_debug") or {}).get("axis_ratio")),
    )

    yaw_deg = _as_float(detection.get("yaw_deg"))

    # -----------------------------
    # Hard checks
    # -----------------------------
    if not class_name:
        reasons.append("missing_class_name")

    if allowed_classes is not None and class_name not in allowed_classes:
        reasons.append("class_not_allowed")

    if class_name in rejected_classes:
        reasons.append("class_rejected")

    if confidence < float(min_confidence):
        reasons.append("low_confidence")
        score -= 25.0

    if detection.get("depth_status") != "available":
        reasons.append("depth_unavailable")
        score -= 35.0

    if not _valid_xyz(detection.get("camera_xyz")):
        reasons.append("camera_xyz_unavailable")
        score -= 35.0

    if distance_m is None:
        reasons.append("distance_unavailable")
        score -= 20.0
    elif distance_m < min_distance_m or distance_m > max_distance_m:
        reasons.append("distance_out_of_range")
        score -= 20.0

    # --------------------------------------------------------
    # 任意物件手眼標定必要條件：
    # 必須有 segmentation polygon + 足夠 object point cloud。
    # 若沒有，只靠 detection box / center pixel / single depth
    # 不足以建立穩定的 R_target2cam。
    # --------------------------------------------------------
    if not has_segmentation:
        reasons.append("segmentation_unavailable")
        score -= 35.0

    if not has_object_point_cloud:
        reasons.append("not_enough_depth_points")
        score -= 30.0

    if not has_segmentation_point_cloud:
        reasons.append("segmentation_point_cloud_unavailable")
        score -= 35.0

    if box_area_ratio < float(min_box_area_ratio):
        reasons.append("object_too_small_in_image")
        score -= 15.0

    if _touches_image_edge(
        detection=detection,
        frame_width=frame_width,
        frame_height=frame_height,
        edge_margin_px=int(edge_margin_px),
    ):
        reasons.append("object_touches_image_edge")
        score -= 15.0

    # -----------------------------
    # Soft quality checks
    # -----------------------------
    if yaw_deg is None:
        warnings.append("yaw_unavailable")
        score -= 8.0

    if axis_ratio is None:
        warnings.append("axis_ratio_unavailable")
        score -= 5.0
    elif axis_ratio < 1.2:
        warnings.append("weak_principal_axis")
        score -= 8.0

    usable_features = []

    if _valid_xyz(detection.get("camera_xyz")):
        usable_features.append("center_3d")

    if yaw_deg is not None:
        usable_features.append("image_yaw")

    if axis_ratio is not None and axis_ratio >= 1.2:
        usable_features.append("principal_axis")

    if has_segmentation_point_cloud:
        usable_features.append("segmentation_point_cloud")

    score = max(0.0, min(100.0, score))

    eligible = (
        len(reasons) == 0
        and score >= 70.0
        and has_segmentation_point_cloud
    )

    return {
        "eligible": bool(eligible),
        "score": float(round(score, 2)),
        "class_name": class_name,
        "confidence": float(confidence),
        "distance_m": distance_m,
        "camera_xyz": detection.get("camera_xyz"),
        "robot_xyz": detection.get("robot_xyz"),
        "depth_status": detection.get("depth_status"),
        "depth_method": detection.get("depth_method"),
        "object_point_count": object_point_count,
        "has_segmentation": bool(has_segmentation),
        "has_object_point_cloud": bool(has_object_point_cloud),
        "has_segmentation_point_cloud": bool(has_segmentation_point_cloud),
        "box_area_ratio": float(round(box_area_ratio, 6)),
        "yaw_deg": yaw_deg,
        "axis_ratio": axis_ratio,
        "usable_features": usable_features,
        "reasons": reasons,
        "warnings": warnings,
        "detection": detection,
    }


def sort_target_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    排序策略：
    1. 合格物件優先
    2. 分數高者優先
    3. 點雲數多者優先
    4. 信心分數高者優先
    """

    return sorted(
        candidates,
        key=lambda item: (
            bool(item.get("eligible")),
            float(item.get("score") or 0.0),
            int(item.get("object_point_count") or 0),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )


# ============================================================
# Public API
# ============================================================

def find_calibration_targets(
    camera_name: str,
    *,
    model_name: str = "object_detector",
    allowed_classes: Optional[Iterable[str]] = None,
    rejected_classes: Optional[Iterable[str]] = None,
    min_confidence: float = 0.60,
    min_point_count: int = 80,
    min_box_area_ratio: float = 0.005,
    min_distance_m: float = 0.15,
    max_distance_m: float = 2.00,
    edge_margin_px: int = 8,
    max_candidates: int = 5,
    include_robot_xyz: bool = True,
) -> Dict[str, Any]:
    """
    呼叫 perception_service，回傳可作為任意物件標定物的候選清單。

    return:
        {
            "camera_name": str,
            "timestamp": float,
            "target_found": bool,
            "best_target": dict | None,
            "candidates": [dict, ...],
            "eligible_count": int,
            "rejected_count": int,
            "recommended_mode": str,
            "multi_target_ready": bool,
            "criteria": dict,
        }
    """

    result = perception_service.detect_objects_value(
        camera_name=camera_name,
        model_name=model_name,
        draw=False,
        include_robot_xyz=include_robot_xyz,
    )

    frame_height, frame_width = _get_frame_shape(result.get("camera_rgb"))
    detections = result.get("detections") or []

    candidates = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue

        candidates.append(
            evaluate_detection_as_target(
                detection=detection,
                frame_width=frame_width,
                frame_height=frame_height,
                allowed_classes=allowed_classes,
                rejected_classes=rejected_classes,
                min_confidence=min_confidence,
                min_point_count=min_point_count,
                min_box_area_ratio=min_box_area_ratio,
                min_distance_m=min_distance_m,
                max_distance_m=max_distance_m,
                edge_margin_px=edge_margin_px,
            )
        )

    candidates = sort_target_candidates(candidates)
    eligible_candidates = [
        item for item in candidates
        if item.get("eligible")
    ]

    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]

    best_target = eligible_candidates[0] if eligible_candidates else None

    return {
        "camera_name": result.get("camera_name", camera_name),
        "timestamp": result.get("timestamp"),
        "target_found": best_target is not None,
        "best_target": best_target,
        "candidates": candidates,
        "eligible_count": len(eligible_candidates),
        "rejected_count": max(0, len(detections) - len(eligible_candidates)),
        "recommended_mode": "single_best_target",
        "multi_target_ready": len(eligible_candidates) >= 2,
        "criteria": {
            "min_confidence": float(min_confidence),
            "min_point_count": int(min_point_count),
            "min_box_area_ratio": float(min_box_area_ratio),
            "min_distance_m": float(min_distance_m),
            "max_distance_m": float(max_distance_m),
            "edge_margin_px": int(edge_margin_px),
            "requires_segmentation_point_cloud": True,
            "allowed_classes": sorted(_normalize_class_set(allowed_classes) or []),
            "rejected_classes": sorted(
                _normalize_class_set(rejected_classes)
                or set(DEFAULT_REJECTED_CLASSES)
            ),
        },
    }


def select_calibration_target(
    camera_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    取得最佳標定候選物。

    若找不到任何具備 segmentation point cloud 的合格物件，
    直接 raise，方便 workflow pre-check 使用。
    """

    result = find_calibration_targets(
        camera_name=camera_name,
        **kwargs,
    )

    if not result.get("target_found"):
        raise RuntimeError(
            "no suitable arbitrary calibration target found: "
            "requires segmentation point cloud"
        )

    return result["best_target"]