/** API boundary for the recording section. */
import { getJson, postJson } from "/web/includes/library/JSFile/modules/core.js";

export const getTaskRegistry = () => getJson("/api/record/get_task_registry");
export const registerTask = (task) => postJson("/api/record/register_task", { task });
export const getRecordingStatus = (format) => getJson(
    `/api/record/get_robot_recording_status?dataset_format=${encodeURIComponent(format)}`
);
export const startRecording = (payload) => postJson("/api/record/start_robot_recording", payload);
export const stopRecording = (payload) => postJson("/api/record/stop_robot_recording", payload);
export const moveRecordingGripper = (action, payload) => postJson(
    `/api/record/${action === "open" ? "open_recording_gripper" : "close_recording_gripper"}`,
    payload,
);
