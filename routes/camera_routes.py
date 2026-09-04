from flask import (
    Blueprint,
    jsonify,
    request,
)

from services import camera_service


camera_bp = Blueprint(
    "camera",
    __name__,
    url_prefix="/api/camera",
)


# ============================================================
# Helpers
# ============================================================

def _get_json():
    return (
        request.get_json(
            silent=True
        )
        or {}
    )


# ============================================================
# STATUS
# ============================================================

@camera_bp.route(
    "/get_camera_status",
    methods=["GET"],
)
def get_camera_status():
    return jsonify(
        camera_service.get_camera_status(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
        )
    )


# ============================================================
# START
# ============================================================

@camera_bp.route(
    "/start_camera",
    methods=["POST"],
)
def start_camera():
    data = _get_json()

    return jsonify(
        camera_service.start_camera(
            camera_name=
                data.get(
                    "camera_name"
                ),
            width=
                data.get(
                    "width"
                ),
            height=
                data.get(
                    "height"
                ),
            fps=
                data.get(
                    "fps"
                ),
        )
    )


# ============================================================
# STOP
# ============================================================

@camera_bp.route(
    "/stop_camera",
    methods=["POST"],
)
def stop_camera():
    data = _get_json()

    return jsonify(
        camera_service.stop_camera(
            camera_name=
                data.get(
                    "camera_name"
                ),
        )
    )


# ============================================================
# INTRINSICS
# ============================================================

@camera_bp.route(
    "/get_intrinsics",
    methods=["GET"],
)
def get_intrinsics():
    return jsonify(
        camera_service.get_intrinsics(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
        )
    )


# ============================================================
# DISTANCE
# ============================================================

@camera_bp.route(
    "/get_distance",
    methods=["GET"],
)
def get_distance():
    return jsonify(
        camera_service.get_distance(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
            x=
                request.args.get(
                    "x"
                ),
            y=
                request.args.get(
                    "y"
                ),
        )
    )


# ============================================================
# DEPROJECT
# ============================================================

@camera_bp.route(
    "/deproject_pixel_to_point",
    methods=["GET"],
)
def deproject_pixel_to_point():
    return jsonify(
        camera_service.deproject_pixel_to_point(
            camera_name=
                request.args.get(
                    "camera_name"
                ),
            x=
                request.args.get(
                    "x"
                ),
            y=
                request.args.get(
                    "y"
                ),
            depth=
                request.args.get(
                    "depth"
                ),
        )
    )