function buildQuery(params = {})
{
    const query = new URLSearchParams();

    for (const [key, value] of Object.entries(params))
    {
        if (value !== undefined && value !== null)
        {
            query.append(key, value);
        }
    }

    return query.toString();
}


// ============================================================
// Image
// ============================================================

export function get_camera_image(camera_name, options = {})
{
    const query = buildQuery({
        camera_name: camera_name,
        quality: options.quality ?? 90,
    });

    return `/api/perception/get_camera_image?${query}`;
}


export function get_detection_image(camera_name, options = {})
{
    const query = buildQuery({
        camera_name: camera_name,
        model_name: options.model_name ?? "object_detector",
        include_robot_xyz: options.include_robot_xyz ?? true,
        quality: options.quality ?? 90,
    });

    return `/api/perception/get_detection_image?${query}`;
}


// ============================================================
// Stream
// ============================================================

export function get_camera_stream(camera_name, options = {})
{
    const query = buildQuery({
        camera_name: camera_name,
        interval_sec: options.interval_sec ?? 0.03,
        quality: options.quality ?? 85,
    });

    return `/api/perception/get_camera_stream?${query}`;
}


export function get_detection_stream(camera_name, options = {})
{
    const query = buildQuery({
        camera_name: camera_name,
        model_name: options.model_name ?? "object_detector",
        include_robot_xyz: options.include_robot_xyz ?? true,
        interval_sec: options.interval_sec ?? 0.10,
        quality: options.quality ?? 85,
    });

    return `/api/perception/get_detection_stream?${query}`;
}


// ============================================================
// Detections
// ============================================================

export function get_detections(camera_name, options = {})
{
    const query = buildQuery({
        camera_name: camera_name,
        model_name: options.model_name ?? "object_detector",
        include_robot_xyz: options.include_robot_xyz ?? true,
    });

    return fetch(`/api/perception/get_detections?${query}`)
        .then(response => response.json());
}