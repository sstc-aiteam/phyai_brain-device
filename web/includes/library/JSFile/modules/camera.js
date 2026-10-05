import { getJson, postJson } from "./core.js";


export function get_camera_status(camera_name = null)
{
    const params = new URLSearchParams();

    if (camera_name !== null)
    {
        params.append("camera_name", camera_name);
    }

    const query = params.toString();

    return getJson(
        `/api/camera/get_camera_status${query ? `?${query}` : ""}`
    );
}


export function start_camera(camera_name, options = {})
{
    return postJson(
        "/api/camera/start_camera",
        {
            camera_name: camera_name,
            width: options.width,
            height: options.height,
            fps: options.fps
        }
    );
}


export function stop_camera(camera_name)
{
    return postJson(
        "/api/camera/stop_camera",
        {
            camera_name: camera_name
        }
    );
}


export function get_intrinsics(camera_name)
{
    const params = new URLSearchParams({
        camera_name: camera_name
    });

    return getJson(
        `/api/camera/get_intrinsics?${params.toString()}`
    );
}


export function get_distance(camera_name, x, y)
{
    const params = new URLSearchParams({
        camera_name: camera_name,
        x: x,
        y: y
    });

    return getJson(
        `/api/camera/get_distance?${params.toString()}`
    );
}


export function deproject_pixel_to_point(
    camera_name,
    x,
    y,
    depth = null
)
{
    const params = new URLSearchParams({
        camera_name: camera_name,
        x: x,
        y: y
    });

    if (depth !== null)
    {
        params.append("depth", depth);
    }

    return getJson(
        `/api/camera/deproject_pixel_to_point?${params.toString()}`
    );
}