from flask import Blueprint, request, Response, jsonify

from services import perception_service


perception_bp = Blueprint(
    "perception",
    __name__,
    url_prefix="/api/perception",
)


# ============================================================
# Image
# ============================================================

@perception_bp.route("/get_camera_image", methods=["GET"])
def get_camera_image():
    camera_name = request.args.get("camera_name")
    quality = request.args.get("quality", 90)

    image = perception_service.get_camera_image(
        camera_name=camera_name,
        quality=quality,
    )

    return Response(
        image,
        mimetype="image/jpeg",
    )


@perception_bp.route("/get_detection_image", methods=["GET"])
def get_detection_image():
    camera_name = request.args.get("camera_name")
    model_name = request.args.get(
        "model_name",
        "object_detector",
    )
    include_robot_xyz = request.args.get(
        "include_robot_xyz",
        True,
    )
    quality = request.args.get(
        "quality",
        90,
    )

    image = perception_service.get_detection_image(
        camera_name=camera_name,
        model_name=model_name,
        include_robot_xyz=include_robot_xyz,
        quality=quality,
    )

    return Response(
        image,
        mimetype="image/jpeg",
    )


# ============================================================
# Stream
# ============================================================

@perception_bp.route("/get_camera_stream", methods=["GET"])
def get_camera_stream():
    camera_name = request.args.get("camera_name")
    interval_sec = request.args.get(
        "interval_sec",
        0.03,
    )
    quality = request.args.get(
        "quality",
        85,
    )

    stream = perception_service.get_camera_stream(
        camera_name=camera_name,
        interval_sec=interval_sec,
        quality=quality,
    )

    return Response(
        stream,
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@perception_bp.route("/get_detection_stream", methods=["GET"])
def get_detection_stream():
    camera_name = request.args.get("camera_name")
    model_name = request.args.get(
        "model_name",
        "object_detector",
    )
    include_robot_xyz = request.args.get(
        "include_robot_xyz",
        True,
    )
    interval_sec = request.args.get(
        "interval_sec",
        0.10,
    )
    quality = request.args.get(
        "quality",
        85,
    )

    stream = perception_service.get_detection_stream(
        camera_name=camera_name,
        model_name=model_name,
        include_robot_xyz=include_robot_xyz,
        interval_sec=interval_sec,
        quality=quality,
    )

    return Response(
        stream,
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# Detections
# ============================================================

@perception_bp.route("/get_detections", methods=["GET"])
def get_detections():
    camera_name = request.args.get("camera_name")
    model_name = request.args.get(
        "model_name",
        "object_detector",
    )
    include_robot_xyz = request.args.get(
        "include_robot_xyz",
        True,
    )

    result = perception_service.get_detections(
        camera_name=camera_name,
        model_name=model_name,
        include_robot_xyz=include_robot_xyz,
    )

    return jsonify(result)