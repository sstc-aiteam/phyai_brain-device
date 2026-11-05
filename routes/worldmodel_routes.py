
from __future__ import annotations

from flask import Blueprint, jsonify, request
from services import worldmodel_service

worldmodel_bp = Blueprint("worldmodel", __name__, url_prefix="/api/worldmodel")


# ============================================================
# Helpers
# ============================================================

def _get_json():
    data = request.get_json(silent=True)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError("request body 必須是 JSON object")

    return data


def _get_camera_names(data):
    camera_names = data.get("camera_names")

    if camera_names is None:
        return None

    if not isinstance(camera_names, (list, tuple)) or not all(isinstance(name, str) and name.strip() for name in camera_names):
        raise ValueError("camera_names 必須是有效的相機名稱陣列")

    return tuple(dict.fromkeys(name.strip() for name in camera_names))


def _respond(action, callback):
    try:
        result = callback()
        status_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), status_code

    except Exception as exc:
        return jsonify({
            "status": "error",
            "module": "worldmodel",
            "action": action,
            "result": False,
            "message": str(exc),
            "error_type": type(exc).__name__
        }), 400


# ============================================================
# Robot State
# ============================================================

@worldmodel_bp.get("/get_arm_status")
def get_arm_status():
    return _respond("get_arm_status", lambda: worldmodel_service.get_arm_status(arm_name=request.args.get("arm_name")))


@worldmodel_bp.get("/get_arm_pose")
def get_arm_pose():
    return _respond("get_arm_pose", lambda: worldmodel_service.get_arm_pose(arm_name=request.args.get("arm_name")))


@worldmodel_bp.get("/get_arm_joints")
def get_arm_joints():
    return _respond("get_arm_joints", lambda: worldmodel_service.get_arm_joints(arm_name=request.args.get("arm_name")))

# ============================================================
# Scene Description
# ============================================================

@worldmodel_bp.post("/describe_scene")
def describe_scene():
    def execute():
        data = _get_json()

        return worldmodel_service.describe_scene(
            camera_name=data.get("camera_name"),
            vlm_backend=data.get("vlm_backend")
        )

    return _respond("describe_scene", execute)


# ============================================================
# Object State
# ============================================================

@worldmodel_bp.post("/get_object_state")
def get_object_state():
    def execute():
        data = _get_json()

        return worldmodel_service.get_object_state(camera_names=_get_camera_names(data))

    return _respond("get_object_state", execute)


# ============================================================
# Object Relations
# ============================================================

@worldmodel_bp.post("/get_object_relation")
def get_object_relation():
    def execute():
        data = _get_json()
        return worldmodel_service.get_object_relation(camera_names=_get_camera_names(data))

    return _respond("get_object_relation", execute)


# ============================================================
# World State
# ============================================================

@worldmodel_bp.post("/world_state")
def get_world_state():
    try:
        data = _get_json()

        world = worldmodel_service.build_world_state(
            context=data.get("context"),
            active_region_id=data.get("active_region_id"),
            include_robot=data.get("include_robot", True),
            include_cameras=data.get("include_cameras", True),
            include_detections=data.get("include_detections", True),
            include_materials=data.get("include_materials", False),
            include_relations=data.get("include_relations", True),
            include_grasp=data.get("include_grasp", True),
            camera_names=_get_camera_names(data)
        )

        return jsonify({
            "status": "success",
            "module": "worldmodel",
            "action": "world_state",
            "result": True,
            "data": world.snapshot()
        }), 200

    except Exception as exc:
        return jsonify({
            "status": "error",
            "module": "worldmodel",
            "action": "world_state",
            "result": False,
            "message": str(exc),
            "error_type": type(exc).__name__
        }), 400


# ============================================================
# Object Grasp
# ============================================================

@worldmodel_bp.post("/get_object_grasp")
def get_object_grasp():
    def execute():
        data = _get_json()

        return worldmodel_service.get_object_grasp(camera_names=_get_camera_names(data))

    return _respond("get_object_grasp", execute)


# ============================================================
# Smoke / Fire State
# ============================================================

@worldmodel_bp.post("/get_smoke_state")
def get_smoke_state():
    def execute():
        data = _get_json()

        return worldmodel_service.get_smoke_state(
            camera_name=data.get("camera_name"),
            observe_seconds=data.get("observe_seconds", 4.0)
        )

    return _respond("get_smoke_state", execute)
