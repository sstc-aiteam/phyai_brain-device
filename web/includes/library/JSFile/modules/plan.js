import {
    postJson,
} from "./core.js";


// ============================================================
// PLAN COMMAND
// POST /api/plan/command
// ============================================================

export function plan_command(
    user_text,
    planning_options = null
)
{
    const payload = {
        user_text:
            user_text,
    };

    if (
        planning_options
        && typeof planning_options
            === "object"
        && !Array.isArray(
            planning_options
        )
    )
    {
        payload.planning_options =
            planning_options;
    }

    return postJson(
        "/api/plan/command",
        payload
    );
}