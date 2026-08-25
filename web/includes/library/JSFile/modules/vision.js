import { getJson, postJson, resolveApiUrl } from "./core.js";

function resolveDirectStreamUrl(path, params)
{
    const streamConfig = (
        typeof window !== "undefined"
        && window.ROBOT_DIRECT_STREAM
        && typeof window.ROBOT_DIRECT_STREAM === "object"
    )
        ? window.ROBOT_DIRECT_STREAM
        : null;

    const basePath = typeof streamConfig?.basePath === "string"
        ? streamConfig.basePath.trim()
        : "";

    if (!basePath) {
        return null;
    }

    const normalizedPath = "/" + String(path || "").replace(/^\/+/, "");
    const url = new URL(
        `${basePath.replace(/\/+$/, "")}${normalizedPath}`,
        window.location.origin,
    );

    const authQuery = normalizedPath === "/api/camera/camera_feed"
        ? String(streamConfig?.cameraFeedAuthQuery || "")
        : String(streamConfig?.visionFeedAuthQuery || "");

    if (authQuery) {
        const authParams = new URLSearchParams(authQuery);
        for (const [key, value] of authParams.entries()) {
            url.searchParams.set(key, value);
        }
    }

    for (const [key, value] of params.entries()) {
        url.searchParams.set(key, value);
    }

    return url.toString();
}

export function start_camera()
{
    return postJson("/api/camera/start_camera");
}


export function stop_camera()
{
    return postJson("/api/camera/stop_camera");
}


export function get_camera_status()
{
    return getJson("/api/camera/get_camera_status");
}


// =========================
// Vision Runtime
// =========================

export function start_vision_runtime(fps = 10)
{
    return postJson("/api/vision/start_vision_runtime",{ fps: fps });
}


export function stop_vision_runtime()
{
    return postJson("/api/vision/stop_runtime");
}


export function get_vision_runtime_status()
{
    return getJson("/api/vision/get_vision_runtime_status");
}


// =========================
// Detection Results
// =========================

export function get_latest_detections(include_robot_xyz = true, mode = null)
{
    const data = { include_robot_xyz: include_robot_xyz};

    if (mode !== null)
    {
        data.mode = mode;
    }

    return postJson("/api/vision/get_latest_detections", data);
}



// =========================
// Stream URLs
// =========================

export function camera_feed(fps = 15, quality = 80)
{
    const params = new URLSearchParams({fps: String(fps), quality: String(quality), t: String(Date.now())});
    const directStreamUrl = resolveDirectStreamUrl(
        "/api/camera/camera_feed",
        params,
    );

    if (directStreamUrl) {
        return directStreamUrl;
    }

    return resolveApiUrl(`/api/camera/camera_feed?${params.toString()}`);
}


export function vision_feed(fps = 10, quality = 80)
{
    const params = new URLSearchParams({fps: String(fps), quality: String(quality), t: String(Date.now())});
    const directStreamUrl = resolveDirectStreamUrl(
        "/api/vision/video_feed",
        params,
    );

    if (directStreamUrl) {
        return directStreamUrl;
    }

    return resolveApiUrl(`/api/vision/video_feed?${params.toString()}`);
}
