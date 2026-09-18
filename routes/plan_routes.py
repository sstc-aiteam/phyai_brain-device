"""
Plan API routes.
"""

from flask import Blueprint, jsonify, request

from services import plan_service


plan_bp = Blueprint(
    "plan",
    __name__,
    url_prefix="/api/plan",
)


@plan_bp.post("/command")
def plan_command():
    """
    建立 plan，不執行實體動作。

    Request:
        {
            "user_text": "...",
            "planning_options": {
                "provider": "local",
                "model": null
            }
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
            "status":
                "error",

            "module":
                "plan",

            "action":
                "plan_command",

            "result":
                False,

            "message":
                "request body 必須是 JSON object",
        }), 400

    user_text = data.get(
        "user_text"
    )

    planning_options = data.get(
        "planning_options"
    )

    result = (
        plan_service
        .plan_command(
            user_text,
            planning_options=
                planning_options,
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