import { postJson } from "./core.js";


export function start_arm_jog(direction)
{
    return postJson("/api/arm/start_arm_jog",{direction: direction});
}


export function stop_arm_jog()
{
    return postJson("/api/arm/stop_arm_jog");
}


export function stop_arm()
{
    return postJson("/api/arm/stop_arm");
}


export function start_arm_freedrive(arm_name)
{
    return postJson("/api/arm/start_arm_freedrive", {arm_name: arm_name});
}


export function stop_arm_freedrive(arm_name)
{
    return postJson("/api/arm/stop_arm_freedrive", {arm_name: arm_name});
}

export function move_arm_point_to_xyz(x, y, z, options = {})
{
    return postJson("/api/arm/move_arm_point_to_xyz", {x: x, y: y, z: z, speed: options.speed ?? 0.1, acceleration: options.acceleration ?? 0.1, wait: options.wait ?? true});
}

export function open_trash_can(options = {})
{
    return postJson("/api/arm/open_trash_can", options);
}

export function close_trash_can(options = {})
{
    return postJson("/api/arm/close_trash_can", options);
}

export function open_top_cabinet(options = {})
{
    return postJson("/api/arm/open_top_cabinet", options);
}

export function close_top_cabinet(options = {})
{
    return postJson("/api/arm/close_top_cabinet", options);
}

export function open_second_drawer(options = {})
{
    return postJson("/api/arm/open_second_drawer", options);
}

export function close_second_drawer(options = {})
{
    return postJson("/api/arm/close_second_drawer", options);
}

export function place_object_in_trash_can(input, options = {})
{
    const payload = {};

    if (Array.isArray(input)) {
        const isPointXYZ = input.length === 3 && input.every(item => Number.isFinite(Number(item)));
        if (isPointXYZ) {
            payload.point_xyz = input;
        } else {
            payload.joint_trajectory = input;
        }
    } else if (input && typeof input === "object") {
        Object.assign(payload, input);
    }

    return postJson("/api/arm/place_object_in_trash_can", {
        ...payload,
        yaw_deg: options.yaw_deg ?? payload.yaw_deg ?? 0.0,
        lula_url: options.lula_url ?? payload.lula_url,
        timeout: options.timeout ?? payload.timeout ?? 300,
        speed: options.speed ?? 0.1,
        acceleration: options.acceleration ?? 0.1,
        wait: options.wait ?? true,
        dt: options.dt ?? 0.1,
        lookahead_time: options.lookahead_time ?? payload.lookahead_time ?? 0.1,
        gain: options.gain ?? 300,
        move_to_start: options.move_to_start ?? true,
    });
}

export function place_object_in_top_cabinet(input, options = {})
{
    const payload = {};

    if (Array.isArray(input)) {
        const isPointXYZ = input.length === 3 && input.every(item => Number.isFinite(Number(item)));
        if (isPointXYZ) {
            payload.point_xyz = input;
        } else {
            payload.joint_trajectory = input;
        }
    } else if (input && typeof input === "object") {
        Object.assign(payload, input);
    }

    return postJson("/api/arm/place_object_in_top_cabinet", {
        ...payload,
        yaw_deg: options.yaw_deg ?? payload.yaw_deg ?? 0.0,
        lula_url: options.lula_url ?? payload.lula_url,
        timeout: options.timeout ?? payload.timeout ?? 300,
        speed: options.speed ?? 0.1,
        acceleration: options.acceleration ?? 0.1,
        wait: options.wait ?? true,
        dt: options.dt ?? 0.1,
        lookahead_time: options.lookahead_time ?? payload.lookahead_time ?? 0.1,
        gain: options.gain ?? 300,
        move_to_start: options.move_to_start ?? true,
    });
}

export function place_object_in_second_drawer(input, options = {})
{
    const payload = {};

    if (Array.isArray(input)) {
        const isPointXYZ = input.length === 3 && input.every(item => Number.isFinite(Number(item)));
        if (isPointXYZ) {
            payload.point_xyz = input;
        } else {
            payload.joint_trajectory = input;
        }
    } else if (input && typeof input === "object") {
        Object.assign(payload, input);
    }

    return postJson("/api/arm/place_object_in_second_drawer", {
        ...payload,
        yaw_deg: options.yaw_deg ?? payload.yaw_deg ?? 0.0,
        lula_url: options.lula_url ?? payload.lula_url,
        timeout: options.timeout ?? payload.timeout ?? 300,
        speed: options.speed ?? 0.1,
        acceleration: options.acceleration ?? 0.1,
        wait: options.wait ?? true,
        dt: options.dt ?? 0.1,
        lookahead_time: options.lookahead_time ?? payload.lookahead_time ?? 0.1,
        gain: options.gain ?? 300,
        move_to_start: options.move_to_start ?? true,
    });
}
