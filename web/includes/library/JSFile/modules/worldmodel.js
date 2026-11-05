
import { getJson, postJson } from "./core.js";


// ============================================================
// Robot State
// ============================================================

export function get_arm_status(arm_name = null)
{
    const query = new URLSearchParams();

    if (arm_name !== null)
    {
        query.set("arm_name", arm_name);
    }

    const suffix = query.toString();
    return getJson(`/api/worldmodel/get_arm_status${suffix ? `?${suffix}` : ""}`);
}


export function get_arm_pose(arm_name = null)
{
    const query = new URLSearchParams();

    if (arm_name !== null)
    {
        query.set("arm_name", arm_name);
    }

    const suffix = query.toString();
    return getJson(`/api/worldmodel/get_arm_pose${suffix ? `?${suffix}` : ""}`);
}


export function get_arm_joints(arm_name = null)
{
    const query = new URLSearchParams();

    if (arm_name !== null)
    {
        query.set("arm_name", arm_name);
    }

    const suffix = query.toString();
    return getJson(`/api/worldmodel/get_arm_joints${suffix ? `?${suffix}` : ""}`);
}


// ============================================================
// Scene Description
// ============================================================

export function describe_scene(camera_name = null, options = {})
{
    return postJson("/api/worldmodel/describe_scene", {
        camera_name: camera_name,
        vlm_backend: options.vlm_backend
    });
}


// ============================================================
// Object State
// ============================================================

export function get_object_state(camera_names = null)
{
    return postJson("/api/worldmodel/get_object_state", {
        camera_names: camera_names
    });
}


// ============================================================
// Object Relation
// ============================================================

export function get_object_relation(camera_names = null)
{
    return postJson("/api/worldmodel/get_object_relation", {
        camera_names: camera_names
    });
}


// ============================================================
// World State
// ============================================================

export function get_world_state(options = {})
{
    return postJson("/api/worldmodel/world_state", {
        context: options.context,
        active_region_id: options.active_region_id,
        include_robot: options.include_robot ?? true,
        include_cameras: options.include_cameras ?? true,
        include_detections: options.include_detections ?? true,
        include_materials: options.include_materials ?? false,
        include_relations: options.include_relations ?? true,
        include_grasp: options.include_grasp ?? true,
        camera_names: options.camera_names
    });
}


// ============================================================
// Object Grasp
// ============================================================

export function get_object_grasp(camera_names = null)
{
    return postJson("/api/worldmodel/get_object_grasp", {
        camera_names: camera_names
    });
}


// ============================================================
// Smoke / Fire State
// ============================================================

export function get_smoke_state(camera_name = null, observe_seconds = 4.0)
{
    return postJson("/api/worldmodel/get_smoke_state", {
        camera_name: camera_name,
        observe_seconds: observe_seconds
    });
}
