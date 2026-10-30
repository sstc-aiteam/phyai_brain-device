/** Local training prototype; no cloud credentials or network uploads. */
import { getJson, postJson, resolveApiUrl } from "./core.js";
let token = sessionStorage.getItem("trainingToken") || "";
let account = null;
try { account = JSON.parse(sessionStorage.getItem("trainingAccount") || "null"); } catch { /* Recover invalid cached display data. */ }
export const getTrainingAccount = () => account;
export const trainingSessionKey = () => token;
function clearSession() {
    token = ""; account = null;
    sessionStorage.removeItem("trainingToken");
    sessionStorage.removeItem("trainingAccount");
}
async function authenticatedGet(url) {
    const session = token;
    const response = await fetch(resolveApiUrl(url), { headers: options().headers, cache: "no-store" });
    if (response.status === 401 && token === session) clearSession();
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || `HTTP ${response.status}`);
    return result;
}
const options = () => ({ headers: { Authorization: `Bearer ${token}` } });
export const hasTrainingSession = () => Boolean(token);
export async function loginTraining(username, password) {
    const result = await postJson("/api/training/auth/login", { username, password });
    token = result.data.token;
    account = result.data.user;
    sessionStorage.setItem("trainingAccount", JSON.stringify(account));
    sessionStorage.setItem("trainingToken", token);
    return result;
}
export async function logoutTraining() {
    try { return await postJson("/api/training/auth/logout", {}, options()); }
    finally { clearSession(); }
}
export const getTrainingUploadStatus = () => authenticatedGet("/api/training/datasets/status");
export const submitTrainingDataset = (payload) => postJson("/api/training/datasets", payload, options());

export const registerTraining = (username, password) => postJson("/api/training/auth/register", { username, password });
export const getTrainingJobs = (query = "") => authenticatedGet(`/api/training/jobs?q=${encodeURIComponent(query)}`);
