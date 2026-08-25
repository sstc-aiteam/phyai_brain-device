"""
VLM 即時描述 API Routes。
"""

from __future__ import annotations

import atexit
import os
import threading

from flask import Blueprint, jsonify, request

from services import vlm_narrator_service


vlm_bp = Blueprint(
    "vlm",
    __name__,
    url_prefix="/api/vlm",
)

VLM_SESSION_ID = (
    os.getenv("VLM_SESSION_ID", "d405-main").strip()
    or "d405-main"
)

# 預設啟用背景 VLM 場景／物件特性描述。
# 要停用時，在啟動 Flask 前設定 VLM_NARRATOR_ENABLED=0 即可，
# 不影響 YOLO、座標轉換或 task_service 的按需 VLM 安全檢查。
VLM_NARRATOR_ENABLED = str(
    os.getenv("VLM_NARRATOR_ENABLED", "1")
).strip().lower() in {"1", "true", "yes", "on"}

_runtime_lock = threading.Lock()
_runtime_started = False
_shutdown_registered = False


def _stop_vlm_runtime():
    global _runtime_started

    with _runtime_lock:
        if not _runtime_started:
            return

        try:
            vlm_narrator_service.stop_runtime()
        finally:
            _runtime_started = False


@vlm_bp.record_once
def _start_vlm_runtime_on_register(state):
    """
    main.py 註冊 vlm_bp 時，自動啟動同 process 的 Narrator。
    """
    global _runtime_started
    global _shutdown_registered

    if not VLM_NARRATOR_ENABLED:
        print(
            "[vlm_routes] narrator auto-start disabled "
            "(set VLM_NARRATOR_ENABLED=1 to enable)",
            flush=True,
        )
        return

    with _runtime_lock:
        if not _runtime_started:
            result = vlm_narrator_service.start_runtime()
            _runtime_started = True

            print(
                f"[vlm_routes] narrator started: {result}",
                flush=True,
            )

        if not _shutdown_registered:
            atexit.register(_stop_vlm_runtime)
            _shutdown_registered = True


@vlm_bp.route("/state", methods=["GET"])
def get_vlm_state():
    state = vlm_narrator_service.get_state(
        session_id=VLM_SESSION_ID,
    )
    state["narrator_enabled"] = VLM_NARRATOR_ENABLED
    return jsonify(state)


@vlm_bp.route("/reobserve", methods=["POST"])
def reobserve_vlm():
    data = request.get_json(silent=True) or {}

    reason = str(
        data.get("reason")
        or "main_ui_request"
    ).strip()

    return jsonify(
        vlm_narrator_service.request_reobserve(
            session_id=VLM_SESSION_ID,
            reason=reason,
        )
    )


@vlm_bp.route("/health", methods=["GET"])
def get_vlm_health():
    health = vlm_narrator_service.get_health()
    health["narrator_enabled"] = VLM_NARRATOR_ENABLED
    return jsonify(health)
