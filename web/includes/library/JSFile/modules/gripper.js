import { postJson } from "./core.js";


export function action_gripper_open()
{
    return postJson("/api/gripper/action_gripper_open");
}


export function action_gripper_close()
{
    return postJson("/api/gripper/action_gripper_close");
}