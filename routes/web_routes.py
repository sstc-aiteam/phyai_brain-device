import os
from flask import Blueprint, send_from_directory

web_bp = Blueprint("web", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
RECORDING_FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "recording_training_replay",
    "frontend",
)

@web_bp.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")

@web_bp.route("/perception")
def vision():
    return send_from_directory(WEB_DIR, "includes/component/perception.html")

@web_bp.route("/worldmodel")
def worldmodel():
    return send_from_directory(WEB_DIR, "includes/component/worldmodel.html")

@web_bp.route("/action")
def action():
    return send_from_directory(WEB_DIR, "includes/component/action.html")


@web_bp.route("/web/<path:filename>")
def web_static(filename):
    return send_from_directory(WEB_DIR, filename)

@web_bp.route("/assets/<path:filename>")
def assets_static(filename):
    return send_from_directory(os.path.join(WEB_DIR, "assets"), filename)