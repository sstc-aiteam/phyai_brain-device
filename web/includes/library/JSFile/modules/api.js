import * as Core from "./core.js";
import * as Agent from "./agent.js";
import * as Camera from "./camera.js";
import * as Perception from "./perception.js";
import * as Arm from "./arm.js";
import * as Gripper from "./gripper.js";
import * as WorldModel from "./worldmodel.js";
import * as Plan from "./plan.js";
import * as Action from "./action.js";
import * as Calibration from "./calibration.js";
import * as Recording from "./recording.js";
import * as Replay from "./replay.js";
import * as Training from "./training.js";

const RobotAPI = {
    ...Core,
    ...Agent,
    ...Camera,
    ...Perception,
    ...WorldModel,
    ...Arm,
    ...Gripper,
    ...Plan,
    ...Action,
    ...Calibration,
    ...Recording,
    ...Replay,
    ...Training,
};


window.RobotAPI = RobotAPI;


// =========================
// Core
// =========================

window.postJson = Core.postJson;
window.sleep = Core.sleep;
window.getElement = Core.getElement;
window.scrollToBottom = Core.scrollToBottom;
window.updateStatus = Core.updateStatus;
window.showReturn = Core.showReturn;
window.updateMessage = Core.updateMessage;
window.updateStreamMessage = Core.updateStreamMessage;
window.toggleHidden = Core.toggleHidden;


// =========================
// Perception
// =========================

window.get_perception_cameras = Perception.get_perception_cameras;
window.get_camera_image = Perception.get_camera_image;
window.get_detection_image = Perception.get_detection_image;
window.get_camera_stream = Perception.get_camera_stream;
window.get_detection_stream = Perception.get_detection_stream;
window.get_detections = Perception.get_detections;


// ============================================================
// WORLD MODEL
// ============================================================

window.get_arm_status = WorldModel.get_arm_status;
window.get_arm_pose = WorldModel.get_arm_pose;
window.get_arm_joints = WorldModel.get_arm_joints;
window.describe_scene = WorldModel.describe_scene;
window.get_object_state = WorldModel.get_object_state;
window.get_object_relation = WorldModel.get_object_relation;
window.get_world_state = WorldModel.get_world_state;
window.get_object_grasp = WorldModel.get_object_grasp;
window.get_smoke_state = WorldModel.get_smoke_state;

// ============================================================
// Plan
// ============================================================
window.plan_command = Plan.plan_command;

// ============================================================
// Action
// ============================================================
window.take_object = Action.take_object;
window.open_drawer = Action.open_drawer;

// ============================================================
// Calibration
// ============================================================
window.calibrate_eye_in_hand = Calibration.calibrate_eye_in_hand;
window.calibrate_hand_eye_board = Calibration.calibrate_hand_eye_board;

// ============================================================
// Recording / Replay / Training
// ============================================================
Object.assign(window, Recording, Replay, Training);

export {
    RobotAPI
};



// =========================
// Agent
// =========================

window.sendAI = Agent.sendAI;
window.createSTT = Agent.createSTT;
window.createTTS = Agent.createTTS;


// =========================
// Camera
// =========================

window.get_camera_status = Camera.get_camera_status;
window.start_camera = Camera.start_camera;
window.stop_camera = Camera.stop_camera;
window.get_intrinsics = Camera.get_intrinsics;
window.get_distance = Camera.get_distance;
window.deproject_pixel_to_point = Camera.deproject_pixel_to_point;


// ============================================================
// ARM
// ============================================================

window.get_arm_status = Arm.get_arm_status;
window.get_arm_pose = Arm.get_arm_pose;
window.get_arm_joints = Arm.get_arm_joints;
window.reconnect_arm = Arm.reconnect_arm;
window.move_arm_default = Arm.move_arm_default;
window.move_arm_pose = Arm.move_arm_pose;
window.move_arm_xyz = Arm.move_arm_xyz;
window.move_arm_joints = Arm.move_arm_joints;
window.move_arm_pose_trajectory = Arm.move_arm_pose_trajectory;
window.move_arm_joint_trajectory = Arm.move_arm_joint_trajectory;
window.move_arm_step = Arm.move_arm_step;
window.move_arm_rotate = Arm.move_arm_rotate;
window.start_arm_jog = Arm.start_arm_jog;
window.stop_arm_jog = Arm.stop_arm_jog;
window.start_arm_freedrive = Arm.start_arm_freedrive;
window.stop_arm_freedrive = Arm.stop_arm_freedrive;
window.stop_arm = Arm.stop_arm;


// ============================================================
// GRIPPER
// ============================================================

window.get_gripper_status = Gripper.get_gripper_status;
window.move_gripper = Gripper.move_gripper;
window.open_gripper = Gripper.open_gripper;
window.close_gripper = Gripper.close_gripper;
window.stop_gripper = Gripper.stop_gripper;