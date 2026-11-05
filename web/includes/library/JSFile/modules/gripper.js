import { getJson, postJson } from "./core.js";


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
// GET STATUS
// ============================================================

export function get_gripper_status(
    gripper_name = null
)
{
    const query = buildQuery({
        gripper_name: gripper_name,
    });

    return getJson(
        `/api/gripper/get_gripper_status${query ? `?${query}` : ""}`
    );
}


// ============================================================
// MOVE
// ============================================================

export function move_gripper(
    gripper_name,
    position,
    options = {}
)
{
    return postJson(
        "/api/gripper/move_gripper",
        {
            gripper_name: gripper_name,
            position: position,
            speed: options.speed,
            force: options.force,
            wait: options.wait,
            timeout: options.timeout,
        }
    );
}


// ============================================================
// OPEN
// ============================================================

export function open_gripper(
    gripper_name,
    options = {}
)
{
    return postJson(
        "/api/gripper/open_gripper",
        {
            gripper_name: gripper_name,
            speed: options.speed,
            force: options.force,
            wait: options.wait,
            timeout: options.timeout,
        }
    );
}


// ============================================================
// CLOSE
// ============================================================

export function close_gripper(
    gripper_name,
    options = {}
)
{
    return postJson(
        "/api/gripper/close_gripper",
        {
            gripper_name: gripper_name,
            speed: options.speed,
            force: options.force,
            wait: options.wait,
            timeout: options.timeout,
        }
    );
}


// ============================================================
// STOP
// ============================================================

export function stop_gripper(gripper_name)
{
    return postJson(
        "/api/gripper/stop_gripper",
        {
            gripper_name: gripper_name,
        }
    );
}