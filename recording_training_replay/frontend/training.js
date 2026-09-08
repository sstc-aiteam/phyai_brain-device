/** API boundary for the training-server section. */
import { getJson, postJson } from "/web/includes/library/JSFile/modules/core.js";

export const getTrainingUploadStatus = () => getJson("/api/training/datasets/status");
export const submitTrainingDataset = (payload) => postJson("/api/training/datasets", payload);
