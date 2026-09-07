"""Fixed robot task workflows built on top of hardware services."""

import config
from services import arm_service
from utils.response import error


MODULE = "task"
RIGHT_INITIAL_TASK = "move_right_arm_initial"


def move_right_arm_initial(speed=None, acceleration=None, wait=True):
    """Move only the right UR5 to its configured initial joint position."""
    right_config = config.ARMS.get("right") or {}
    if right_config.get("driver") != "ur5":
        return error(
            MODULE,
            RIGHT_INITIAL_TASK,
            message="move_right_arm_initial is restricted to the right UR5",
        )

    initial_joints = (right_config.get("poses") or {}).get("initial_joints")
    if initial_joints is None:
        return error(
            MODULE,
            RIGHT_INITIAL_TASK,
            message="poses.initial_joints is not configured for right arm",
        )

    result = arm_service.move_arm_joints(
        arm_name="right",
        joints=initial_joints,
        speed=speed,
        acceleration=acceleration,
        wait=wait,
    )
    if isinstance(result, dict):
        result = result.copy()
        result["module"] = MODULE
        result["action"] = RIGHT_INITIAL_TASK
    return result
