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


@plan_bp.post("/runtime/bootstrap")
def runtime_bootstrap():
    """
    Build the initial multi-region Runtime WorldState.

    Must run inside the main Flask process so it shares the already-started
    camera / D405 instance.
    """

    try:
        world = (
            plan_service
            .build_runtime_initial_world()
        )

        return jsonify({
            "status":
                "success",

            "module":
                "plan",

            "action":
                "runtime_bootstrap",

            "result":
                True,

            "data":
                world.snapshot(),
        }), 200

    except Exception as exc:
        return jsonify({
            "status":
                "error",

            "module":
                "plan",

            "action":
                "runtime_bootstrap",

            "result":
                False,

            "error_type":
                type(
                    exc
                ).__name__,

            "message":
                str(
                    exc
                ),
        }), 500



@plan_bp.post("/runtime/manual/start")
def runtime_manual_start():
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
                "runtime_manual_start",
            "result":
                False,
            "message":
                "request body 必須是 JSON object",
        }), 400

    try:
        session = (
            plan_service
            .start_manual_runtime(
                data.get(
                    "user_text"
                ),

                planning_options=
                    data.get(
                        "planning_options"
                    ),

                openai_api_key=
                    data.get(
                        "openai_api_key"
                    ),

                max_iterations=
                    data.get(
                        "max_iterations",
                        10,
                    ),

                force_device_id=
                    data.get(
                        "force_device_id"
                    ),
            )
        )

        return jsonify({
            "status":
                "success",
            "module":
                "plan",
            "action":
                "runtime_manual_start",
            "result":
                True,
            "data":
                session,
        }), 202

    except (
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        return jsonify({
            "status":
                "error",
            "module":
                "plan",
            "action":
                "runtime_manual_start",
            "result":
                False,
            "error_type":
                type(
                    exc
                ).__name__,
            "message":
                str(
                    exc
                ),
        }), 400


@plan_bp.get("/runtime/manual/status")
def runtime_manual_status():
    try:
        session = (
            plan_service
            .get_manual_runtime_status(
                request.args.get(
                    "session_id"
                )
            )
        )

        return jsonify({
            "status":
                "success",
            "module":
                "plan",
            "action":
                "runtime_manual_status",
            "result":
                True,
            "data":
                session,
        }), 200

    except (
        KeyError,
        RuntimeError,
    ) as exc:
        return jsonify({
            "status":
                "error",
            "module":
                "plan",
            "action":
                "runtime_manual_status",
            "result":
                False,
            "error_type":
                type(
                    exc
                ).__name__,
            "message":
                str(
                    exc
                ),
        }), 404


@plan_bp.post("/runtime/manual/confirm")
def runtime_manual_confirm():
    data = request.get_json(
        silent=True
    ) or {}

    try:
        session = (
            plan_service
            .confirm_manual_runtime_step(
                str(
                    data.get(
                        "session_id"
                    )
                    or ""
                )
            )
        )

        return jsonify({
            "status":
                "success",
            "module":
                "plan",
            "action":
                "runtime_manual_confirm",
            "result":
                True,
            "data":
                session,
        }), 200

    except (
        KeyError,
        RuntimeError,
    ) as exc:
        return jsonify({
            "status":
                "error",
            "module":
                "plan",
            "action":
                "runtime_manual_confirm",
            "result":
                False,
            "error_type":
                type(
                    exc
                ).__name__,
            "message":
                str(
                    exc
                ),
        }), 400


@plan_bp.post("/runtime/manual/cancel")
def runtime_manual_cancel():
    data = request.get_json(
        silent=True
    ) or {}

    try:
        session = (
            plan_service
            .cancel_manual_runtime(
                str(
                    data.get(
                        "session_id"
                    )
                    or ""
                )
            )
        )

        return jsonify({
            "status":
                "success",
            "module":
                "plan",
            "action":
                "runtime_manual_cancel",
            "result":
                True,
            "data":
                session,
        }), 200

    except (
        KeyError,
        RuntimeError,
    ) as exc:
        return jsonify({
            "status":
                "error",
            "module":
                "plan",
            "action":
                "runtime_manual_cancel",
            "result":
                False,
            "error_type":
                type(
                    exc
                ).__name__,
            "message":
                str(
                    exc
                ),
        }), 400
