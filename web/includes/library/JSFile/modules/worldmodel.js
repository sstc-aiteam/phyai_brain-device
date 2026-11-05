import {
    getJson,
    postJson,
} from "./core.js";


// ============================================================
// DESCRIBE SCENE
// POST /api/worldmodel/describe_scene
// ============================================================

export function describe_scene(
    camera_name = null,
    options = {}
)
{
    return postJson(
        "/api/worldmodel/describe_scene",
        {
            camera_name:
                camera_name,

            vlm_backend:
                options.vlm_backend,
        }
    );
}


// ============================================================
// GET OBJECT STATE
// POST /api/worldmodel/get_object_state
// ============================================================

export function get_object_state(
    camera_names = null
)
{
    const payload = {};

    if (
        Array.isArray(
            camera_names
        )
    )
    {
        payload.camera_names =
            camera_names;
    }

    return postJson(
        "/api/worldmodel/get_object_state",
        payload
    );
}


// ============================================================
// GET OBJECT RELATIONS
// POST /api/worldmodel/get_object_relations
// ============================================================

export function get_object_relations(
    camera_names = null
)
{
    const payload = {};

    if (
        Array.isArray(
            camera_names
        )
    )
    {
        payload.camera_names =
            camera_names;
    }

    return postJson(
        "/api/worldmodel/get_object_relations",
        payload
    );
}


// ============================================================
// INFER RELATIONS
// POST /api/worldmodel/relations
// ============================================================

export function infer_relations(
    camera_name,
    entities
)
{
    return postJson(
        "/api/worldmodel/relations",
        {
            camera_name:
                camera_name,

            entities:
                entities,
        }
    );
}


// ============================================================
// HEALTH
// GET /api/worldmodel/health
// ============================================================

export function get_worldmodel_health()
{
    return getJson(
        "/api/worldmodel/health"
    );
}


// ============================================================
// WORLD STATE
// POST /api/worldmodel/world_state
// ============================================================

export function get_world_state(
    options = {}
)
{
    return postJson(
        "/api/worldmodel/world_state",
        {
            context:
                options.context,

            include_robot:
                options.include_robot
                ?? true,

            include_cameras:
                options.include_cameras
                ?? true,

            include_detections:
                options.include_detections
                ?? true,

            include_materials:
                options.include_materials
                ?? false,

            include_relations:
                options.include_relations
                ?? true,

            include_grasp:
                options.include_grasp
                ?? true,

            camera_names:
                options.camera_names,
        }
    );
}


// ============================================================
// GET OBJECT GRASP
// POST /api/worldmodel/get_object_grasp
// ============================================================

export function get_object_grasp(
    camera_names = null
)
{
    const payload = {};

    if (
        Array.isArray(
            camera_names
        )
    )
    {
        payload.camera_names =
            camera_names;
    }

    return postJson(
        "/api/worldmodel/get_object_grasp",
        payload
    );
}