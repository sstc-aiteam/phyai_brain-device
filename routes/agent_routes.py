from __future__ import annotations

from flask import Blueprint, Response, jsonify, request, stream_with_context
from services import agent_service


agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")


# ============================================================
# Run Agent
# ============================================================

@agent_bp.post("/workflow/run-stream")
def workflow_run_stream_http():
    body = request.get_json(silent=True)

    if "output" in request.args:
        return jsonify(
            agent_service.workflow_api_error(
                "workflow_run_stream",
                "請把 output 放在 JSON body，勿使用 URL Query",
            )
        ), 400

    result, status_code, is_stream = agent_service.run_workflow_request(body)

    if is_stream:
        response = Response(
            stream_with_context(result),
            content_type="text/plain; charset=utf-8",
        )
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    return jsonify(result), status_code
