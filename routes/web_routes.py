import os
from flask import Blueprint, send_from_directory

web_bp = Blueprint("web", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")

@web_bp.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")

@web_bp.route("/vision")
def vision():
    return send_from_directory(WEB_DIR, "example/ex_vision.html")

@web_bp.route("/agent-test")
def agent_test():
    return send_from_directory(WEB_DIR, "agent_test.html")

@web_bp.route("/record")
def record():
    return send_from_directory(WEB_DIR, "record.html")

@web_bp.route("/web/<path:filename>")
def web_static(filename):
    return send_from_directory(WEB_DIR, filename)

@web_bp.route("/assets/<path:filename>")
def assets_static(filename):
    return send_from_directory(os.path.join(WEB_DIR, "assets"), filename)