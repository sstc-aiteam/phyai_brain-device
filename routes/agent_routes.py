"""Agent routes for the existing /api/agent Blueprint.

No separate Flask process. Only the Workflow task-input API and health remain.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from services import agent_service

agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")


def _error(action: str, message: str, error_type: str = "ValueError") -> dict:
    return {
        "status": "error", "module": "agent", "action": action,
        "result": False, "data": None,
        "message": message, "error_type": error_type,
    }


def _ok(action: str, data: dict) -> dict:
    return {
        "status": "success", "module": "agent", "action": action,
        "result": True, "data": data,
    }


@agent_bp.get("/health")
def health():
    """No model calls and no hardware calls."""
    return jsonify(_ok("workflow_health", {
        "api": "workflow_real_minimal_v2",
        "supported_providers": ["openai", "local"],
        "default_mode": "plan",
        "check_default": False,
        "busy": agent_service.is_busy(),
        "real_dispatch_api": "enabled; hardware checks delegated to Action Gateway",
        "process_local": True,
    })), 200


@agent_bp.post("/workflow/run")
def workflow_run_http():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(_error("workflow_run", "request body 必須是 JSON object")), 400
    allowed = {"user_text", "provider", "camera_name", "mode", "check", "model"}
    extra = set(body) - allowed
    if extra:
        return jsonify(_error("workflow_run", "不支援欄位：" + ", ".join(sorted(extra)))), 400
    try:
        job = agent_service.create_workflow(
            user_text=body.get("user_text"),
            provider=body.get("provider", "openai"),
            camera_name=body.get("camera_name", "left"),
            mode=body.get("mode", "plan"),
            check=body.get("check", False),
            model=body.get("model"),
        )
    except ValueError as exc:
        return jsonify(_error("workflow_run", str(exc), "ValueError")), 400
    except agent_service.AgentBusy as exc:
        return jsonify(_error("workflow_run", str(exc), "AgentBusy")), 409
    except Exception as exc:
        # Do not expose internal tracebacks or credentials.
        return jsonify(_error("workflow_run", f"無法建立 Workflow：{type(exc).__name__}",
                              type(exc).__name__)), 500
    return jsonify(_ok("workflow_run", job)), 202


@agent_bp.get("/workflow/<job_id>")
def workflow_status_http(job_id):
    try:
        job = agent_service.get_workflow(job_id)
    except LookupError:
        return jsonify(_error("workflow_status", "找不到 Workflow Job", "NotFound")), 404
    return jsonify(_ok("workflow_status", job)), 200


@agent_bp.post("/workflow/<job_id>/decision")
def workflow_decision_http(job_id):
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(_error("workflow_decision", "request body 必須是 JSON object")), 400
    extra = set(body) - {"gate_id", "decision"}
    if extra:
        return jsonify(_error("workflow_decision", "不支援欄位：" + ", ".join(sorted(extra)))), 400
    try:
        job = agent_service.decide_workflow(
            job_id, gate_id=body.get("gate_id"), decision=body.get("decision"))
    except LookupError:
        return jsonify(_error("workflow_decision", "找不到 Workflow Job", "NotFound")), 404
    except ValueError as exc:
        return jsonify(_error("workflow_decision", str(exc), "ValueError")), 400
    except RuntimeError as exc:
        return jsonify(_error("workflow_decision", str(exc), "Conflict")), 409
    return jsonify(_ok("workflow_decision", job)), 200
