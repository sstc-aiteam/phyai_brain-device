import * as Core from "./core.js";
import * as Agent from "./agent.js";
import * as Vision from "./vision.js";
import * as Arm from "./arm.js";
import * as Gripper from "./gripper.js";
import * as VLM from "./vlm.js";
import * as Demo from "./demo.js";



const RobotAPI = {
    ...Core,
    ...Agent,
    ...Vision,
    ...Arm,
    ...Gripper,
    ...VLM,
    ...Demo
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
// Agent
// =========================

window.sendAI = Agent.sendAI;
window.createSTT = Agent.createSTT;
window.createTTS = Agent.createTTS;

// =========================
// Vision
// =========================

window.start_camera = Vision.start_camera;
window.stop_camera = Vision.stop_camera;
window.get_camera_status = Vision.get_camera_status;

window.start_vision_runtime = Vision.start_vision_runtime;
window.stop_vision_runtime = Vision.stop_vision_runtime;
window.get_vision_runtime_status = Vision.get_vision_runtime_status;

window.get_latest_detections = Vision.get_latest_detections;

window.camera_feed = Vision.camera_feed;
window.vision_feed = Vision.vision_feed;


// =========================
// Arm
// =========================

window.start_arm_jog = Arm.start_arm_jog;
window.stop_arm_jog = Arm.stop_arm_jog;
window.stop_arm = Arm.stop_arm;
window.move_arm_point_to_xyz = Arm.move_arm_point_to_xyz;
window.open_trash_can = Arm.open_trash_can;
window.close_trash_can = Arm.close_trash_can;
window.open_top_cabinet = Arm.open_top_cabinet;
window.close_top_cabinet = Arm.close_top_cabinet;
window.open_second_drawer = Arm.open_second_drawer;
window.close_second_drawer = Arm.close_second_drawer;
window.place_object_in_trash_can = Arm.place_object_in_trash_can;
window.place_object_in_top_cabinet = Arm.place_object_in_top_cabinet;
window.place_object_in_second_drawer = Arm.place_object_in_second_drawer;


// =========================
// Gripper
// =========================

window.action_gripper_open = Gripper.action_gripper_open;
window.action_gripper_close = Gripper.action_gripper_close;


// =========================
// VLM
// =========================

window.initVlmUI = VLM.initVlmUI;
window.startVlmPolling = VLM.startVlmPolling;
window.stopVlmPolling = VLM.stopVlmPolling;
window.pollVlmState = VLM.pollVlmState;
window.clearUnderstanding = VLM.clearUnderstanding;
window.requestVlmReobserve = VLM.requestVlmReobserve;

VLM.initVlmUI();

// =========================
// DEMO
// =========================

window.run_full_demo = Demo.runFullDemo;
window.runFullDemo2 = Demo.runFullDemo2;
window.run_demo_top_cabinet = Demo.runDemoTopCabinet;
window.run_demo_second_drawer = Demo.runDemoSecondDrawer;
window.run_demo_trash_can = Demo.runDemoTrashCan;

export {
    RobotAPI
};
