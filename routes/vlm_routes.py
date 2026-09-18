from __future__ import annotations
from flask import Blueprint, jsonify, request
from services import vlm_narrator_service


vlm_bp = Blueprint(
    "vlm",
    __name__,
    url_prefix="/api/vlm",
)


# ============================================================
# Scene Description
# ============================================================

@vlm_bp.post("/describe_scene")
def describe_scene():
    """
    單張 RGB 場景理解。

    Request:
        {
            "camera_name": "left",
            "vlm_backend": "remote"
        }

    camera_name 未指定時，
    由 VLM capability 使用其預設值。

    vlm_backend:
        "remote" -> PC Qwen3-VL
        其他 / 未指定 -> local Qwen
    """

    data = request.get_json(
        silent=True
    )

    if data is None:
        data = {}

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "module": "vlm",
            "action": "describe_scene",
            "result": False,
            "message": (
                "request body 必須是 JSON object"
            ),
        }), 400

    result = (
        vlm_narrator_service
        .describe_scene(
            camera_name=data.get(
                "camera_name"
            ),

            vlm_backend=data.get(
                "vlm_backend"
            ),
        )
    )

    status_code = (
        200
        if result.get("status") == "success"
        else 400
    )

    return jsonify(
        result
    ), status_code

#============================================================
# Object state
#============================================================

@vlm_bp.post("/get_object_state")
def get_object_state():
    data = request.get_json(
        silent=True
    )

    if data is None:
        data = {}

    if not isinstance(
        data,
        dict,
    ):
        return jsonify({
            "status": "error",
            "module": "vlm",
            "action": "get_object_state",
            "result": False,
            "message":
                "request body 必須是 JSON object",
        }), 400

    try:
        result = (
            vlm_narrator_service
            .get_object_state(
                camera_names=tuple(
                    data.get(
                        "camera_names",
                        ("left", "right"),
                    )
                ),
            )
        )

        return jsonify(
            result
        ), 200

    except Exception as exc:
        return jsonify({
            "status": "error",
            "module": "vlm",
            "action": "get_object_state",
            "result": False,
            "message":
                str(exc),
            "error_type":
                type(exc).__name__,
        }), 400


# ============================================================
# Object Relations
# ============================================================

@vlm_bp.post("/get_object_relations")
def get_object_relations():
    """
    重新觀察目前場景，
    回傳精簡 object-object relations。

    Request:
        {
            "camera_names": ["left"]
        }
    """

    data = request.get_json(
        silent=True
    )

    if data is None:
        data = {}

    if not isinstance(
        data,
        dict,
    ):
        return jsonify({
            "status": "error",
            "module": "vlm",
            "action": "get_object_relation",
            "result": False,
            "message":
                "request body 必須是 JSON object",
        }), 400

    try:
        result = (
            vlm_narrator_service
            .get_object_relation(
                camera_names=tuple(
                    data.get(
                        "camera_names",
                        ("left", "right"),
                    )
                ),
            )
        )

        status_code = (
            200
            if result.get(
                "status"
            )
            == "success"
            else 400
        )

        return jsonify(
            result
        ), status_code

    except Exception as exc:
        return jsonify({
            "status": "error",
            "module": "vlm",
            "action": "get_object_relation",
            "result": False,
            "message":
                str(
                    exc
                ),
            "error_type":
                type(
                    exc
                ).__name__,
        }), 400

# ============================================================
# Semantic Relations
# ============================================================

@vlm_bp.post("/relations")
def infer_relations():
    """
    根據單張 RGB 與已知 entities，
    判斷 entities 之間的 semantic relations。

    Request:
        {
            "camera_name": "left",
            "entities": [...]
        }
    """

    data = request.get_json(
        silent=True
    )

    if data is None:
        data = {}

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "module": "vlm",
            "action": "infer_relations",
            "result": False,
            "message":
                "request body 必須是 JSON object",
        }), 400

    result = (
        vlm_narrator_service
        .infer_relations(
            camera_name=data.get(
                "camera_name"
            ),
            entities=data.get(
                "entities"
            ),
        )
    )

    status_code = (
        200
        if result.get("status") == "success"
        else 400
    )

    return jsonify(
        result
    ), status_code


# ============================================================
# Health
# ============================================================

@vlm_bp.get("/health")
def get_vlm_health():
    """
    回傳目前公開的 VLM capabilities
    與 implementation metadata。
    """

    return jsonify(
        vlm_narrator_service.get_health()
    )
# ============================================================
# World State
# ============================================================

@vlm_bp.post("/world_state")
def get_world_state():
    """
    建立目前 live canonical WorldState。
    """

    data = request.get_json(
        silent=True
    )

    if data is None:
        data = {}

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "module": "vlm",
            "action": "world_state",
            "result": False,
            "message":
                "request body 必須是 JSON object",
        }), 400

    try:
        world = (
            vlm_narrator_service
            .build_world_state(
                context=data.get(
                    "context"
                ),
                include_robot=data.get(
                    "include_robot",
                    True,
                ),
                include_cameras=data.get(
                    "include_cameras",
                    True,
                ),
                include_detections=data.get(
                    "include_detections",
                    True,
                ),
                camera_names=tuple(
                    data.get(
                        "camera_names",
                        ("left", "right"),
                    )
                ),
            )
        )

        return jsonify({
            "status": "success",
            "module": "vlm",
            "action": "world_state",
            "result": True,
            "data":
                world.snapshot(),
        }), 200

    except Exception as exc:
        return jsonify({
            "status": "error",
            "module": "vlm",
            "action": "world_state",
            "result": False,
            "message": str(exc),
            "error_type":
                type(exc).__name__,
        }), 400