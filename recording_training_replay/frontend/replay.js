/** API boundary for the replay section. */
import { getJson, postJson } from "/web/includes/library/JSFile/modules/core.js";

export const getReplayCatalog = () => getJson("/api/record/get_replay_catalog");
export const startPlayback = (payload) => postJson("/api/record/start_robot_playback", payload);
export const stopPlayback = (payload) => postJson("/api/record/stop_robot_playback", payload);
