"""
VLM 即時描述 API Routes。
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from services import vlm_narrator_service


vlm_bp = Blueprint(
    "vlm",
    __name__,
    url_prefix="/api/vlm",
)

@vlm_bp.route("/state", methods=["GET"])
def get_vlm_state():
    return jsonify(vlm_narrator_service.get_state())


@vlm_bp.route("/narrate_once", methods=["POST"])
def narrate_once():
    return jsonify(vlm_narrator_service.narrate_once())


@vlm_bp.route("/monitor/start", methods=["POST"])
def start_monitor():
    return jsonify(vlm_narrator_service.start_runtime())


@vlm_bp.route("/monitor/stop", methods=["POST"])
def stop_monitor():
    return jsonify(vlm_narrator_service.stop_runtime())


@vlm_bp.route("/reobserve", methods=["POST"])
def reobserve_vlm():
    data = request.get_json(silent=True) or {}

    reason = str(
        data.get("reason")
        or "main_ui_request"
    ).strip()

    return jsonify(
        vlm_narrator_service.request_reobserve(
            reason=reason,
        )
    )


@vlm_bp.route("/health", methods=["GET"])
def get_vlm_health():
    return jsonify(vlm_narrator_service.get_health())
