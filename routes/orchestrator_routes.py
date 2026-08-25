"""HTTP routes for guarded two-arm planning and execution on port 5001."""

from flask import Blueprint, jsonify, request

from services import orchestrator_service
from agent.ollama_client import OllamaClientError


orchestrator_bp = Blueprint(
    "orchestrator",
    __name__,
    url_prefix="/api/orchestrator",
)


@orchestrator_bp.post("/probe")
def probe_cells():
    return jsonify({"status": "success", "data": orchestrator_service.probe_cells()})


@orchestrator_bp.post("/plan")
def prepare_plan():
    body = request.get_json(silent=True)
    try:
        plan = orchestrator_service.prepare_plan(body)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    return jsonify({"status": "success", "data": plan})


@orchestrator_bp.post("/agent/plan")
def plan_from_text():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"status": "error", "message": "request body must be JSON"}), 400
    try:
        result = orchestrator_service.plan_from_text(body.get("user_text"))
    except OllamaClientError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    return jsonify({"status": "success", "data": result})


@orchestrator_bp.post("/execute")
def execute_plan():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"status": "error", "message": "request body must be JSON"}), 400
    try:
        result = orchestrator_service.execute_plan(
            body.get("plan_id"),
            body.get("confirmation"),
            request.headers.get("X-Orchestrator-Execution-Token"),
        )
    except PermissionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 403
    except (ValueError, RuntimeError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 409
    return jsonify({"status": "success", "data": result})


@orchestrator_bp.get("/execution-policy")
def execution_policy():
    return jsonify({"status": "success", "data": orchestrator_service.execution_policy()})
