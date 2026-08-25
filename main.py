from flask import Flask, jsonify, request

from routes.web_routes import web_bp
from routes.arm_routes import arm_bp
from routes.gripper_routes import gripper_bp
from routes.camera_routes import camera_bp
from routes.vision_routes import vision_bp
from routes.record_routes import record_bp

app = Flask(__name__)

app.register_blueprint(web_bp)
app.register_blueprint(arm_bp)
app.register_blueprint(gripper_bp)
app.register_blueprint(camera_bp)
app.register_blueprint(vision_bp)
app.register_blueprint(record_bp)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )
