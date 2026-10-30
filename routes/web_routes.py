import os
from flask import Blueprint, send_from_directory

web_bp = Blueprint("web", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
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

@web_bp.route("/plan")
def plan():
    return send_from_directory(WEB_DIR,"includes/component/plan.html",)

@web_bp.route("/training")
def training():
    return send_from_directory(WEB_DIR, "includes/component/training.html")


@web_bp.route("/web/<path:filename>")
def web_static(filename):
    return send_from_directory(WEB_DIR, filename)

@web_bp.route("/assets/<path:filename>")
def assets_static(filename):
    return send_from_directory(os.path.join(WEB_DIR, "assets"), filename)