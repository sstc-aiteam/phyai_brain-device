import logging
import math
import config
from control import loader
from utils import response


MODULE = "arm"
logger = logging.getLogger(__name__)


# ============================================================
# Arm Context
# ============================================================

def _normalize_arm_name(arm_name):
    if not isinstance(arm_name, str):
        raise ValueError(
            "arm_name must be a string"
        )

    arm_name = arm_name.strip().lower()

    if not arm_name:
        raise ValueError(
            "arm_name must not be empty"
        )

    if arm_name not in config.ARMS:
        raise ValueError(
            f"Unsupported arm: {arm_name}. "
            f"Supported arms: "
            f"{', '.join(sorted(config.ARMS))}"
        )

    return arm_name


def _get_arm_context(arm_name):
    arm_name = _normalize_arm_name(
        arm_name
    )

    arm_config = config.ARMS[
        arm_name
    ]

    arm = loader.get_arm_driver(
        arm_name
    )

    driver = loader.get_arm_driver_name(
        arm_name
    )

    return (
        arm_name,
        arm,
        driver,
        arm_config,
    )


def _get_target_arm_names(
    arm_name=None,
):
    if arm_name is None:
        return list(
            config.ARMS.keys()
        )

    return [
        _normalize_arm_name(
            arm_name
        )
    ]


# ============================================================
# Response Helper
# ============================================================

def _execution_results(
    action,
    reached,
    data=None,
    driver=None,
    **kwargs,
):
    if not isinstance(
        reached,
        bool,
    ):
        raise RuntimeError(
            f"arm driver '{driver}' "
            f"回傳無效結果："
            f"預期 bool，實際為 "
            f"{type(reached).__name__}"
        )

    if not reached:
        return response.success(
            MODULE,
            action,
            result=False,
            data=data,
            driver=driver,
            message=(
                "motion interrupted "
                "or target not reached"
            ),
            **kwargs,
        )

    return response.success(
        MODULE,
        action,
        result=True,
        data=data,
        driver=driver,
        **kwargs,
    )


# ============================================================
# Validation Helpers
# ============================================================

def _normalize_number(
    name,
    value,
):
    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} 必須是數值"
        ) from exc

    if not math.isfinite(
        number
    ):
        raise ValueError(
            f"{name} 必須是有限數值"
        )

    return number


def _normalize_pose(
    pose,
):
    if (
        not isinstance(
            pose,
            (list, tuple),
        )
        or len(pose) != 6
    ):
        raise ValueError(
            "pose 必須包含 6 個數值："
            "[x, y, z, rx, ry, rz]"
        )

    return [
        _normalize_number(
            f"pose[{index}]",
            value,
        )
        for index, value
        in enumerate(pose)
    ]


def _normalize_joints(
    joints,
    arm_dof,
):
    if (
        not isinstance(
            arm_dof,
            int,
        )
        or arm_dof <= 0
    ):
        raise RuntimeError(
            "Driver ARM_DOF invalid"
        )

    if (
        not isinstance(
            joints,
            (list, tuple),
        )
        or len(joints) != arm_dof
    ):
        raise ValueError(
            f"joints 必須包含 "
            f"{arm_dof} 個數值"
        )

    return [
        _normalize_number(
            f"joints[{index}]",
            value,
        )
        for index, value
        in enumerate(joints)
    ]


# ============================================================
# Config Helpers
# ============================================================

def _get_motion_params(
    arm_config,
    speed=None,
    acceleration=None,
):
    motion = arm_config.get(
        "motion",
        {},
    )

    if speed is None:
        speed = motion.get(
            "speed"
        )

    if acceleration is None:
        acceleration = motion.get(
            "acceleration"
        )

    if speed is None:
        raise ValueError(
            "motion.speed is not configured"
        )

    if acceleration is None:
        raise ValueError(
            "motion.acceleration "
            "is not configured"
        )

    return (
        _normalize_number(
            "speed",
            speed,
        ),
        _normalize_number(
            "acceleration",
            acceleration,
        ),
    )


def _get_trajectory_dt(
    arm_config,
    dt=None,
):
    if dt is None:
        dt = (
            arm_config
            .get(
                "motion",
                {},
            )
            .get(
                "dt"
            )
        )

    if dt is None:
        raise ValueError(
            "motion.dt is not configured"
        )

    dt = _normalize_number(
        "dt",
        dt,
    )

    if dt <= 0:
        raise ValueError(
            "dt 必須大於 0"
        )

    return dt


def _get_jog_params(
    arm_config,
    linear_speed=None,
    angular_speed=None,
    acceleration=None,
    timeout=None,
):
    jog = arm_config.get(
        "jog",
        {},
    )

    if linear_speed is None:
        linear_speed = jog.get(
            "linear_speed"
        )

    if angular_speed is None:
        angular_speed = jog.get(
            "angular_speed"
        )

    if acceleration is None:
        acceleration = jog.get(
            "acceleration"
        )

    if timeout is None:
        timeout = jog.get(
            "timeout"
        )

    required = {
        "linear_speed":
            linear_speed,

        "angular_speed":
            angular_speed,

        "acceleration":
            acceleration,

        "timeout":
            timeout,
    }

    for name, value in (
        required.items()
    ):
        if value is None:
            raise ValueError(
                f"jog.{name} "
                f"is not configured"
            )

    return (
        _normalize_number(
            "linear_speed",
            linear_speed,
        ),
        _normalize_number(
            "angular_speed",
            angular_speed,
        ),
        _normalize_number(
            "jog_acceleration",
            acceleration,
        ),
        _normalize_number(
            "jog_timeout",
            timeout,
        ),
    )


# ============================================================
# Safety
# ============================================================

def _check_pose_safety(
    pose,
    arm_config,
):
    pose = _normalize_pose(
        pose
    )

    safety = arm_config.get(
        "safety",
        {},
    )

    ranges = {
        "x": (
            pose[0],
            safety.get(
                "x_range"
            ),
        ),
        "y": (
            pose[1],
            safety.get(
                "y_range"
            ),
        ),
        "z": (
            pose[2],
            safety.get(
                "z_range"
            ),
        ),
    }

    for axis, (
        value,
        axis_range,
    ) in ranges.items():

        if axis_range is None:
            continue

        if (
            not isinstance(
                axis_range,
                (list, tuple),
            )
            or len(axis_range) != 2
        ):
            raise ValueError(
                f"safety.{axis}_range "
                f"格式錯誤"
            )

        minimum = float(
            axis_range[0]
        )

        maximum = float(
            axis_range[1]
        )

        if not (
            minimum
            <= value
            <= maximum
        ):
            raise ValueError(
                f"{axis} 超出 "
                f"{axis} safety range："
                f"{value}，"
                f"允許範圍 "
                f"{minimum} ~ {maximum}"
            )

    return pose


# ============================================================
# Connection Helper
# ============================================================

def _get_connected_arm_context(
    arm_name,
):
    (
        arm_name,
        arm,
        driver,
        arm_config,
    ) = _get_arm_context(
        arm_name
    )

    status = arm.get_arm_status()

    if not isinstance(
        status,
        dict,
    ):
        raise RuntimeError(
            f"arm driver '{driver}' "
            f"get_arm_status() "
            f"must return dict"
        )

    if not status.get(
        "connected",
        False,
    ):
        return None

    return (
        arm_name,
        arm,
        driver,
        arm_config,
        status,
    )


# ============================================================
# GET STATUS
# ============================================================

# ============================================================
# GET STATUS
# ============================================================

def get_arm_status(
    arm_name=None,
):
    """
    取得所有手臂的狀態。
    """

    action = "get_arm_status"

    try:
        arms = []

        for name in (
            _get_target_arm_names(
                arm_name
            )
        ):
            driver = None

            try:
                (
                    current_name,
                    arm,
                    driver,
                    _,
                ) = _get_arm_context(
                    name
                )

                raw_status = (
                    arm.get_arm_status()
                )

                if not isinstance(
                    raw_status,
                    dict,
                ):
                    raise RuntimeError(
                        f"arm driver "
                        f"'{driver}' "
                        f"get_arm_status() "
                        f"must return dict"
                    )

                pose = raw_status.get(
                    "pose"
                )

                if pose is not None:
                    pose = (
                        _normalize_pose(
                            pose
                        )
                    )

                joints = raw_status.get(
                    "joints"
                )

                if joints is not None:
                    joints = (
                        _normalize_joints(
                            joints,
                            arm.ARM_DOF,
                        )
                    )

                status = {
                    "connected":
                        bool(
                            raw_status.get(
                                "connected",
                                False,
                            )
                        ),

                    "ready":
                        raw_status.get(
                            "ready"
                        ),

                    "moving":
                        raw_status.get(
                            "moving"
                        ),

                    "protective_stop":
                        raw_status.get(
                            "protective_stop"
                        ),

                    "emergency_stop":
                        raw_status.get(
                            "emergency_stop"
                        ),

                    "fault":
                        raw_status.get(
                            "fault"
                        ),

                    "program_running":
                        raw_status.get(
                            "program_running"
                        ),

                    "arm_mode":
                        raw_status.get(
                            "arm_mode"
                        ),

                    "safety_mode":
                        raw_status.get(
                            "safety_mode"
                        ),

                    "pose":
                        pose,

                    "joints":
                        joints,
                }

                arms.append(
                    {
                        "arm_name":
                            current_name,

                        "driver":
                            driver,

                        "status":
                            status,
                    }
                )

            except Exception as exc:
                logger.warning(
                    "arm status unavailable: "
                    "arm_name=%s, error=%s",
                    name,
                    exc,
                )

                arms.append(
                    {
                        "arm_name":
                            name,

                        "driver":
                            driver,

                        "status": {
                            "connected":
                                False,

                            "ready":
                                None,

                            "moving":
                                None,

                            "protective_stop":
                                None,

                            "emergency_stop":
                                None,

                            "fault":
                                None,

                            "program_running":
                                None,

                            "arm_mode":
                                None,

                            "safety_mode":
                                None,

                            "pose":
                                None,

                            "joints":
                                None,
                        },
                    }
                )

        return response.success(
            MODULE,
            action,
            result=True,
            data={
                "arms": arms,
            },
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# GET POSE
# ============================================================

def get_arm_pose(
    arm_name=None,
):
    action = "get_arm_pose"

    try:
        arms = []

        for name in (
            _get_target_arm_names(
                arm_name
            )
        ):
            try:
                context = (
                    _get_connected_arm_context(
                        name
                    )
                )

                if context is None:
                    continue

                (
                    current_name,
                    _,
                    driver,
                    _,
                    status,
                ) = context

                pose = status.get(
                    "pose"
                )

                if pose is None:
                    raise RuntimeError(
                        f"arm "
                        f"'{current_name}' "
                        f"pose unavailable"
                    )

                pose = _normalize_pose(
                    pose
                )

                arms.append(
                    {
                        "arm_name":
                            current_name,

                        "driver":
                            driver,

                        "pose":
                            pose,
                    }
                )

            except Exception as exc:
                if arm_name is not None:
                    return response.error(
                        MODULE,
                        action,
                        error=exc,
                        error_type=
                            type(exc).__name__,
                    )

                continue

        return response.success(
            MODULE,
            action,
            result=True,
            data={
                "arms": arms,
            },
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# GET JOINTS
# ============================================================

def get_arm_joints(
    arm_name=None,
):
    action = "get_arm_joints"

    try:
        arms = []

        for name in (
            _get_target_arm_names(
                arm_name
            )
        ):
            try:
                context = (
                    _get_connected_arm_context(
                        name
                    )
                )

                if context is None:
                    continue

                (
                    current_name,
                    arm,
                    driver,
                    _,
                    status,
                ) = context

                joints = status.get(
                    "joints"
                )

                if joints is None:
                    raise RuntimeError(
                        f"arm "
                        f"'{current_name}' "
                        f"joints unavailable"
                    )

                joints = _normalize_joints(
                    joints,
                    arm.ARM_DOF,
                )

                arms.append(
                    {
                        "arm_name":
                            current_name,

                        "driver":
                            driver,

                        "joints":
                            joints,
                    }
                )

            except Exception as exc:
                if arm_name is not None:
                    return response.error(
                        MODULE,
                        action,
                        error=exc,
                        error_type=
                            type(exc).__name__,
                    )

                continue

        return response.success(
            MODULE,
            action,
            result=True,
            data={
                "arms": arms,
            },
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# RECONNECT
# ============================================================

def reconnect_arm(
    arm_name,
):
    action = "reconnect_arm"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            _,
        ) = _get_arm_context(
            arm_name
        )

        if not callable(
            getattr(
                arm,
                "reconnect_arm",
                None,
            )
        ):
            raise NotImplementedError(
                f"arm driver '{driver}' "
                f"does not support reconnect_arm"
            )

        reached = arm.reconnect_arm()

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# MOVE DEFAULT
# ============================================================

def move_arm_default(
    arm_name,
    speed=None,
    acceleration=None,
    wait=True,
):
    action = "move_arm_default"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        (
            speed,
            acceleration,
        ) = _get_motion_params(
            arm_config,
            speed,
            acceleration,
        )

        poses = arm_config.get(
            "poses",
            {},
        )

        intermediate = poses.get(
            "intermediate_default_joints"
        )

        default = poses.get(
            "default_joints"
        )

        if intermediate is None:
            raise ValueError(
                "poses."
                "intermediate_default_joints "
                "is not configured"
            )

        if default is None:
            raise ValueError(
                "poses.default_joints "
                "is not configured"
            )

        intermediate = (
            _normalize_joints(
                intermediate,
                arm.ARM_DOF,
            )
        )

        default = (
            _normalize_joints(
                default,
                arm.ARM_DOF,
            )
        )

        reached = arm.move_arm_joints(
            joints=intermediate,
            speed=speed,
            acceleration=acceleration,
            wait=wait,
        )

        if not reached:
            return _execution_results(
                action,
                reached,
                data={
                    "arm_name":
                        arm_name,

                    "target_joints":
                        intermediate,
                },
                driver=driver,
            )

        reached = arm.move_arm_joints(
            joints=default,
            speed=speed,
            acceleration=acceleration,
            wait=wait,
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "target_joints":
                    default,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# MOVE POSE
# ============================================================

def move_arm_pose(
    arm_name,
    pose,
    speed=None,
    acceleration=None,
    wait=True,
):
    action = "move_arm_pose"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        (
            speed,
            acceleration,
        ) = _get_motion_params(
            arm_config,
            speed,
            acceleration,
        )

        target_pose = (
            _check_pose_safety(
                pose,
                arm_config,
            )
        )

        reached = arm.move_arm_pose(
            *target_pose,
            speed=speed,
            acceleration=acceleration,
            wait=wait,
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "target_pose":
                    target_pose,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# MOVE XYZ
# ============================================================

def move_arm_xyz(
    arm_name,
    x,
    y,
    z,
    speed=None,
    acceleration=None,
    wait=True,
):
    action = "move_arm_xyz"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        (
            speed,
            acceleration,
        ) = _get_motion_params(
            arm_config,
            speed,
            acceleration,
        )

        current_pose = (
            _normalize_pose(
                arm.get_arm_pose()
            )
        )

        target_pose = [
            _normalize_number(
                "x",
                x,
            ),
            _normalize_number(
                "y",
                y,
            ),
            _normalize_number(
                "z",
                z,
            ),
            current_pose[3],
            current_pose[4],
            current_pose[5],
        ]

        target_pose = (
            _check_pose_safety(
                target_pose,
                arm_config,
            )
        )

        reached = arm.move_arm_pose(
            *target_pose,
            speed=speed,
            acceleration=acceleration,
            wait=wait,
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "target_pose":
                    target_pose,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# MOVE JOINTS
# ============================================================

def move_arm_joints(
    arm_name,
    joints,
    speed=None,
    acceleration=None,
    wait=True,
):
    action = "move_arm_joints"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        (
            speed,
            acceleration,
        ) = _get_motion_params(
            arm_config,
            speed,
            acceleration,
        )

        target_joints = (
            _normalize_joints(
                joints,
                arm.ARM_DOF,
            )
        )

        reached = (
            arm.move_arm_joints(
                joints=
                    target_joints,
                speed=speed,
                acceleration=
                    acceleration,
                wait=wait,
            )
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "target_joints":
                    target_joints,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# TRAJECTORY
# ============================================================

def move_arm_joint_trajectory(
    arm_name,
    joint_trajectory,
    dt=None,
    speed=None,
    acceleration=None,
    wait=True,
    move_to_start=True,
    move_to_start_speed=None,
    move_to_start_acceleration=None,
):
    action = (
        "move_arm_joint_trajectory"
    )

    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        if not callable(
            getattr(
                arm,
                "move_arm_joint_trajectory",
                None,
            )
        ):
            raise NotImplementedError(
                f"arm driver '{driver}' "
                f"does not support "
                f"move_arm_joint_trajectory"
            )

        (
            speed,
            acceleration,
        ) = _get_motion_params(
            arm_config,
            speed,
            acceleration,
        )

        dt = _get_trajectory_dt(
            arm_config,
            dt,
        )

        if not isinstance(
            joint_trajectory,
            (list, tuple),
        ):
            raise ValueError(
                "joint_trajectory "
                "必須是 list 或 tuple"
            )

        if not joint_trajectory:
            raise ValueError(
                "joint_trajectory 不可為空"
            )

        target_trajectory = [
            _normalize_joints(
                joints,
                arm.ARM_DOF,
            )
            for joints
            in joint_trajectory
        ]

        reached = (
            arm.move_arm_joint_trajectory(
                joint_trajectory=
                    target_trajectory,

                dt=dt,

                speed=speed,

                acceleration=
                    acceleration,

                wait=wait,

                move_to_start=
                    move_to_start,

                move_to_start_speed=
                    move_to_start_speed,

                move_to_start_acceleration=
                    move_to_start_acceleration,
            )
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "trajectory_points":
                    len(
                        target_trajectory
                    ),
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


def move_arm_pose_trajectory(
    arm_name,
    pose_trajectory,
    dt=None,
    speed=None,
    acceleration=None,
    wait=True,
    move_to_start=True,
    move_to_start_speed=None,
    move_to_start_acceleration=None,
):
    action = (
        "move_arm_pose_trajectory"
    )

    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        if not callable(
            getattr(
                arm,
                "move_arm_pose_trajectory",
                None,
            )
        ):
            raise NotImplementedError(
                f"arm driver '{driver}' "
                f"does not support "
                f"move_arm_pose_trajectory"
            )

        (
            speed,
            acceleration,
        ) = _get_motion_params(
            arm_config,
            speed,
            acceleration,
        )

        dt = _get_trajectory_dt(
            arm_config,
            dt,
        )

        if not isinstance(
            pose_trajectory,
            (list, tuple),
        ):
            raise ValueError(
                "pose_trajectory "
                "必須是 list 或 tuple"
            )

        if not pose_trajectory:
            raise ValueError(
                "pose_trajectory 不可為空"
            )

        target_trajectory = [
            _check_pose_safety(
                pose,
                arm_config,
            )
            for pose
            in pose_trajectory
        ]

        reached = (
            arm.move_arm_pose_trajectory(
                pose_trajectory=
                    target_trajectory,

                dt=dt,

                speed=speed,

                acceleration=
                    acceleration,

                wait=wait,

                move_to_start=
                    move_to_start,

                move_to_start_speed=
                    move_to_start_speed,

                move_to_start_acceleration=
                    move_to_start_acceleration,
            )
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "trajectory_points":
                    len(
                        target_trajectory
                    ),
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# MOVE STEP
# ============================================================

def move_arm_step(
    arm_name,
    direction,
    distance=None,
    speed=None,
    acceleration=None,
    wait=True,
):
    action = "move_arm_step"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        (
            speed,
            acceleration,
        ) = _get_motion_params(
            arm_config,
            speed,
            acceleration,
        )

        if distance is None:
            distance = (
                arm_config
                .get(
                    "motion",
                    {},
                )
                .get(
                    "dx"
                )
            )

        if distance is None:
            raise ValueError(
                "motion.dx "
                "is not configured"
            )

        distance = _normalize_number(
            "distance",
            distance,
        )

        if distance < 0:
            raise ValueError(
                "distance 不可小於 0"
            )

        if not isinstance(
            direction,
            str,
        ):
            raise ValueError(
                "direction 必須是字串"
            )

        direction = (
            direction
            .strip()
            .lower()
        )

        direction_map = {
            "x+": (0, 1.0),
            "x-": (0, -1.0),
            "y+": (1, 1.0),
            "y-": (1, -1.0),
            "z+": (2, 1.0),
            "z-": (2, -1.0),
        }

        if direction not in (
            direction_map
        ):
            raise ValueError(
                f"Unsupported direction: "
                f"{direction}"
            )

        current_pose = (
            _normalize_pose(
                arm.get_arm_pose()
            )
        )

        target_pose = list(
            current_pose
        )

        axis, sign = (
            direction_map[
                direction
            ]
        )

        target_pose[
            axis
        ] += (
            distance
            * sign
        )

        target_pose = (
            _check_pose_safety(
                target_pose,
                arm_config,
            )
        )

        reached = (
            arm.move_arm_pose(
                *target_pose,
                speed=speed,
                acceleration=
                    acceleration,
                wait=wait,
            )
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "direction":
                    direction,

                "distance":
                    distance,

                "target_pose":
                    target_pose,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# MOVE ROTATE
# ============================================================

def move_arm_rotate(
    arm_name,
    direction,
    angle=None,
    speed=None,
    acceleration=None,
    wait=True,
):
    action = "move_arm_rotate"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        (
            speed,
            acceleration,
        ) = _get_motion_params(
            arm_config,
            speed,
            acceleration,
        )

        if angle is None:
            angle = (
                arm_config
                .get(
                    "motion",
                    {},
                )
                .get(
                    "dr"
                )
            )

        if angle is None:
            raise ValueError(
                "motion.dr "
                "is not configured"
            )

        angle = _normalize_number(
            "angle",
            angle,
        )

        if angle < 0:
            raise ValueError(
                "angle 不可小於 0"
            )

        if not isinstance(
            direction,
            str,
        ):
            raise ValueError(
                "direction 必須是字串"
            )

        direction = (
            direction
            .strip()
            .lower()
        )

        direction_map = {
            "rx+": (3, 1.0),
            "rx-": (3, -1.0),
            "ry+": (4, 1.0),
            "ry-": (4, -1.0),
            "rz+": (5, 1.0),
            "rz-": (5, -1.0),
        }

        if direction not in (
            direction_map
        ):
            raise ValueError(
                f"Unsupported direction: "
                f"{direction}"
            )

        current_pose = (
            _normalize_pose(
                arm.get_arm_pose()
            )
        )

        target_pose = list(
            current_pose
        )

        axis, sign = (
            direction_map[
                direction
            ]
        )

        target_pose[
            axis
        ] += (
            angle
            * sign
        )

        target_pose = (
            _check_pose_safety(
                target_pose,
                arm_config,
            )
        )

        reached = (
            arm.move_arm_pose(
                *target_pose,
                speed=speed,
                acceleration=
                    acceleration,
                wait=wait,
            )
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "direction":
                    direction,

                "angle":
                    angle,

                "target_pose":
                    target_pose,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# JOG
# ============================================================

def start_arm_jog(
    arm_name,
    direction,
    linear_speed=None,
    angular_speed=None,
    acceleration=None,
    timeout=None,
):
    action = "start_arm_jog"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        if not callable(
            getattr(
                arm,
                "start_arm_jog",
                None,
            )
        ):
            raise NotImplementedError(
                f"arm driver '{driver}' "
                f"does not support jog"
            )

        (
            linear_speed,
            angular_speed,
            acceleration,
            timeout,
        ) = _get_jog_params(
            arm_config,
            linear_speed,
            angular_speed,
            acceleration,
            timeout,
        )

        reached = (
            arm.start_arm_jog(
                direction=direction,

                linear_speed=
                    linear_speed,

                angular_speed=
                    angular_speed,

                acceleration=
                    acceleration,

                timeout=timeout,
            )
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "direction":
                    direction,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


def stop_arm_jog(
    arm_name,
):
    action = "stop_arm_jog"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            _,
        ) = _get_arm_context(
            arm_name
        )

        if not callable(
            getattr(
                arm,
                "stop_arm_jog",
                None,
            )
        ):
            raise NotImplementedError(
                f"arm driver '{driver}' "
                f"does not support jog"
            )

        reached = (
            arm.stop_arm_jog()
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# FREEDRIVE
# ============================================================

def start_arm_freedrive(
    arm_name,
):
    action = (
        "start_arm_freedrive"
    )

    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            _,
        ) = _get_arm_context(
            arm_name
        )

        if not callable(
            getattr(
                arm,
                "start_arm_freedrive",
                None,
            )
        ):
            raise NotImplementedError(
                f"arm driver '{driver}' "
                f"does not support freedrive"
            )

        reached = (
            arm.start_arm_freedrive()
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "freedrive":
                    True,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


def stop_arm_freedrive(
    arm_name,
):
    action = (
        "stop_arm_freedrive"
    )

    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            _,
        ) = _get_arm_context(
            arm_name
        )

        if not callable(
            getattr(
                arm,
                "stop_arm_freedrive",
                None,
            )
        ):
            raise NotImplementedError(
                f"arm driver '{driver}' "
                f"does not support freedrive"
            )

        reached = (
            arm.stop_arm_freedrive()
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,

                "freedrive":
                    False,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )


# ============================================================
# STOP
# ============================================================

def stop_arm(
    arm_name,
    acceleration=None,
):
    action = "stop_arm"
    driver = None

    try:
        (
            arm_name,
            arm,
            driver,
            arm_config,
        ) = _get_arm_context(
            arm_name
        )

        if not callable(
            getattr(
                arm,
                "stop_arm",
                None,
            )
        ):
            raise NotImplementedError(
                f"arm driver '{driver}' "
                f"does not support stop_arm"
            )

        if acceleration is None:
            acceleration = (
                arm_config
                .get(
                    "motion",
                    {},
                )
                .get(
                    "stop_acceleration"
                )
            )

        if acceleration is None:
            acceleration = (
                arm_config
                .get(
                    "jog",
                    {},
                )
                .get(
                    "acceleration"
                )
            )

        if acceleration is not None:
            acceleration = (
                _normalize_number(
                    "stop_acceleration",
                    acceleration,
                )
            )

        reached = (
            arm.stop_arm(
                acceleration=
                    acceleration
            )
        )

        return _execution_results(
            action,
            reached,
            data={
                "arm_name":
                    arm_name,
            },
            driver=driver,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=exc,
            driver=driver,
            error_type=
                type(exc).__name__,
        )