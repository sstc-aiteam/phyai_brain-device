from flask import Flask, jsonify, request

from routes.web_routes import web_bp
from routes.arm_routes import arm_bp
from routes.gripper_routes import gripper_bp
from routes.camera_routes import camera_bp
from routes.perception_routes import perception_bp
from routes.worldmodel_routes import worldmodel_bp
from routes.plan_routes import plan_bp
from routes.action_routes import action_bp
from routes.agent_routes import agent_bp
from routes.recording_training_replay_routes import BLUEPRINTS as recording_training_replay_blueprints

app = Flask(__name__)

app.register_blueprint(web_bp)
app.register_blueprint(arm_bp)
app.register_blueprint(gripper_bp)
app.register_blueprint(camera_bp)
app.register_blueprint(perception_bp)
app.register_blueprint(worldmodel_bp)
app.register_blueprint(plan_bp)
app.register_blueprint(action_bp)
app.register_blueprint(agent_bp)
for blueprint in recording_training_replay_blueprints:
    app.register_blueprint(blueprint)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )