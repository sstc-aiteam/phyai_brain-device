from flask import Blueprint, jsonify, request
from services import agent_service

agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")


@agent_bp.post("/action")
def action_command():
    data = request.get_json(silent=True)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        return jsonify({
            "module": "agent",
            "action": "action_command",
            "status": "error",
            "result": False,
            "data": None,
            "message": "request body 必須是 JSON object",
            "error_type": "ValueError",
        }), 400

    response = agent_service.action_command(
        user_text=data.get("user_text"),
    )

    status_code = (
        200
        if response.get("status") == "success"
        else 400
    )

    return jsonify(response), status_code