from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
)

from services import vision_service


vision_bp = Blueprint(
    "vision",
    __name__,
    url_prefix="/api/vision",
)


# ============================================================
# DETECTIONS
# ============================================================

@vision_bp.route(
    "/get_detections",
    methods=["GET"],
)
def get_detections():
    return jsonify(
        vision_service.get_detections(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
            model_name=
                request.args.get(
                    "model_name",
                    "object_detector",
                ),
            include_robot_xyz=
                request.args.get(
                    "include_robot_xyz",
                    "true",
                ),
        )
    )


# ============================================================
# CAMERA RGB
# ============================================================

@vision_bp.route(
    "/camera_rgb",
    methods=["GET"],
)
def camera_rgb():
    return Response(
        vision_service.get_camera_rgb_jpeg(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
        ),
        mimetype="image/jpeg",
    )


# ============================================================
# YOLO IMAGE
# ============================================================

@vision_bp.route(
    "/yolo_image",
    methods=["GET"],
)
def yolo_image():
    return Response(
        vision_service.get_yolo_image_jpeg(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
            model_name=
                request.args.get(
                    "model_name",
                    "object_detector",
                ),
            include_robot_xyz=
                request.args.get(
                    "include_robot_xyz",
                    "true",
                ),
        ),
        mimetype="image/jpeg",
    )


# ============================================================
# VLA OBSERVATION
# ============================================================

@vision_bp.route(
    "/get_vla_observation",
    methods=["GET"],
)
def get_vla_observation():
    return jsonify(
        vision_service.get_vla_observation_summary(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
            model_name=
                request.args.get(
                    "model_name",
                    "object_detector",
                ),
            run_yolo=
                request.args.get(
                    "run_yolo",
                    "true",
                ),
            include_point_cloud=
                request.args.get(
                    "include_point_cloud",
                    "true",
                ),
        )
    )


# ============================================================
# CAMERA STREAM
# ============================================================

@vision_bp.route(
    "/camera_stream",
    methods=["GET"],
)
def camera_stream():
    return Response(
        vision_service.camera_stream(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
        ),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


# ============================================================
# YOLO STREAM
# ============================================================

@vision_bp.route(
    "/yolo_stream",
    methods=["GET"],
)
def yolo_stream():
    return Response(
        vision_service.yolo_stream(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
            model_name=
                request.args.get(
                    "model_name",
                    "object_detector",
                ),
            include_robot_xyz=
                request.args.get(
                    "include_robot_xyz",
                    "true",
                ),
        ),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )