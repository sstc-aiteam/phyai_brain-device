from importlib import import_module
import threading
import config

# ============================================================
# ARM
# ============================================================

ARM_DRIVERS = {
    "ur5": {
        "module": "control.ur5",
        "class": "UR5Driver",
    },
    "ur7e": {
        "module": "control.ur7e",
        "class": "UR7eDriver",
    },
}

ARM_ATTRIBUTES = (
    "ARM_DOF",
)

ARM_FUNCTIONS = (
    "get_arm_status",
    "get_arm_pose",
    "get_arm_joints",
    "reconnect_arm",
    "move_arm_pose",
    "move_arm_joints",
    "move_arm_joint_trajectory",
    "start_arm_jog",
    "stop_arm_jog",
    "start_arm_freedrive",
    "stop_arm_freedrive",
    "stop_arm",
)

_ARM_INSTANCES = {}
_ARM_INSTANCE_LOCK = threading.RLock()

# ============================================================
# GRIPPER
# ============================================================

GRIPPER_DRIVERS = {
    "robotiq": {
        "module": "control.robotiq",
        "class": "RobotiqDriver",
    },
    "robotiq_eseries": {
        "module": "control.robotiq_eSeries",
        "class": "RobotiqESeriesDriver",
    },
}

GRIPPER_ATTRIBUTES = (
    "DRIVER_METADATA",
)

GRIPPER_FUNCTIONS = (
    "get_gripper_status",
    "move_gripper",
    "open_gripper",
    "close_gripper",
    "stop_gripper",
)

_GRIPPER_INSTANCES = {}
_GRIPPER_INSTANCE_LOCK = threading.RLock()

# ============================================================
# CAMERA
# ============================================================

CAMERA_DRIVERS = {
    "d405": {
        "module": "control.d405",
        "class": "D405Driver",
    },

    "usb_camera": {
        "module": "control.usb_camera",
        "class": "USBCameraDriver",
    },
}

CAMERA_ATTRIBUTES = (
    "DRIVER_METADATA",
)

CAMERA_FUNCTIONS = (
    "start_camera",
    "stop_camera",
    "get_camera_status",
    "get_frame",
    "get_distance",
    "deproject_pixel_to_point",
    "get_intrinsics",
    "get_point_cloud",
)

_CAMERA_INSTANCES = {}
_CAMERA_INSTANCE_LOCK = threading.RLock()

# ============================================================
# Internal Helpers
# ============================================================

def _load_driver(
    driver_name,
    driver_definition,
    driver_type,
    kwargs=None,
):
    """
    動態載入 Driver class 並建立 instance。
    """

    if not isinstance(
        driver_name,
        str,
    ) or not driver_name.strip():
        raise ValueError(
            f"{driver_type} "
            f"must be a non-empty string"
        )

    driver_name = (
        driver_name
        .strip()
        .lower()
    )

    if not isinstance(
        driver_definition,
        dict,
    ):
        raise RuntimeError(
            f"Invalid {driver_type} "
            f"definition: {driver_name}"
        )

    module_path = (
        driver_definition.get(
            "module"
        )
    )

    class_name = (
        driver_definition.get(
            "class"
        )
    )

    if not isinstance(
        module_path,
        str,
    ) or not module_path.strip():
        raise RuntimeError(
            f"Invalid {driver_type} "
            f"'{driver_name}': "
            f"module is missing"
        )

    if not isinstance(
        class_name,
        str,
    ) or not class_name.strip():
        raise RuntimeError(
            f"Invalid {driver_type} "
            f"'{driver_name}': "
            f"class is missing"
        )

    if kwargs is None:
        kwargs = {}

    if not isinstance(
        kwargs,
        dict,
    ):
        raise RuntimeError(
            f"{driver_type} kwargs "
            f"must be a dictionary"
        )

    try:
        module = import_module(
            module_path
        )

    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import "
            f"{driver_type} "
            f"'{driver_name}' "
            f"from '{module_path}': "
            f"{exc}"
        ) from exc

    driver_class = getattr(
        module,
        class_name,
        None,
    )

    if driver_class is None:
        raise RuntimeError(
            f"Class '{class_name}' "
            f"not found in "
            f"'{module_path}'"
        )

    if not isinstance(
        driver_class,
        type,
    ):
        raise RuntimeError(
            f"'{module_path}."
            f"{class_name}' "
            f"is not a class"
        )

    try:
        return driver_class(
            **kwargs
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to create "
            f"{driver_type} "
            f"'{driver_name}': "
            f"{exc}"
        ) from exc


def _validate_driver(
    driver,
    driver_name,
    driver_type,
    required_attributes=(),
    required_functions=(),
):
    """
    驗證 Driver interface contract。
    """

    missing_attributes = [
        name
        for name in required_attributes
        if not hasattr(
            driver,
            name,
        )
    ]

    missing_functions = [
        name
        for name in required_functions
        if not callable(
            getattr(
                driver,
                name,
                None,
            )
        )
    ]

    errors = []

    if missing_attributes:
        errors.append(
            "missing attributes: "
            + ", ".join(
                missing_attributes
            )
        )

    if missing_functions:
        errors.append(
            "missing functions: "
            + ", ".join(
                missing_functions
            )
        )

    if errors:
        raise RuntimeError(
            f"Invalid {driver_type} "
            f"'{driver_name}': "
            + "; ".join(
                errors
            )
        )


def _normalize_arm_name(
    arm_name,
):
    if not isinstance(
        arm_name,
        str,
    ) or not arm_name.strip():
        raise ValueError(
            "arm_name must be "
            "a non-empty string"
        )

    arm_name = (
        arm_name
        .strip()
        .lower()
    )

    if arm_name not in config.ARMS:
        raise ValueError(
            f"Unsupported arm: "
            f"{arm_name}. "
            f"Supported arms: "
            f"{', '.join(sorted(config.ARMS))}"
        )

    return arm_name

def _normalize_gripper_name(
    gripper_name,
):
    if not isinstance(
        gripper_name,
        str,
    ) or not gripper_name.strip():
        raise ValueError(
            "gripper_name must be "
            "a non-empty string"
        )

    gripper_name = (
        gripper_name
        .strip()
        .lower()
    )

    if gripper_name not in config.GRIPPERS:
        raise ValueError(
            f"Unsupported gripper: "
            f"{gripper_name}. "
            f"Supported grippers: "
            f"{', '.join(sorted(config.GRIPPERS))}"
        )

    return gripper_name

def _normalize_camera_name(
    camera_name,
):
    if not isinstance(
        camera_name,
        str,
    ) or not camera_name.strip():
        raise ValueError(
            "camera_name must be "
            "a non-empty string"
        )

    camera_name = (
        camera_name
        .strip()
        .lower()
    )

    if camera_name not in config.CAMERAS:
        raise ValueError(
            f"Unsupported camera: "
            f"{camera_name}. "
            f"Supported cameras: "
            f"{', '.join(sorted(config.CAMERAS))}"
        )

    return camera_name


# ============================================================
# Arm Driver
# ============================================================
def get_arm_driver(
    arm_name,
):
    """
    取得指定 arm_name 的 Driver instance。
    """

    arm_name = (
        _normalize_arm_name(
            arm_name
        )
    )

    with _ARM_INSTANCE_LOCK:

        if arm_name in (
            _ARM_INSTANCES
        ):
            return (
                _ARM_INSTANCES[
                    arm_name
                ]
            )

        arm_config = (
            config.ARMS[
                arm_name
            ]
        )

        driver_name = (
            arm_config.get(
                "driver"
            )
        )

        if not isinstance(
            driver_name,
            str,
        ) or not driver_name.strip():
            raise RuntimeError(
                f"Arm '{arm_name}' "
                f"driver is missing"
            )

        driver_name = (
            driver_name
            .strip()
            .lower()
        )

        driver_definition = (
            ARM_DRIVERS.get(
                driver_name
            )
        )

        if driver_definition is None:
            raise RuntimeError(
                f"Unsupported ARM driver: "
                f"{driver_name}. "
                f"Supported drivers: "
                f"{', '.join(sorted(ARM_DRIVERS))}"
            )

        kwargs = dict(
            arm_config.get(
                "kwargs",
                {},
            )
        )

        driver = (
            _load_driver(
                driver_name=
                    driver_name,
                driver_definition=
                    driver_definition,
                driver_type=
                    "ARM_DRIVER",
                kwargs=
                    kwargs,
            )
        )

        _validate_driver(
            driver=driver,
            driver_name=
                driver_name,
            driver_type=
                "ARM_DRIVER",
            required_attributes=
                ARM_ATTRIBUTES,
            required_functions=
                ARM_FUNCTIONS,
        )

        _ARM_INSTANCES[
            arm_name
        ] = driver

        return driver


def get_arm_driver_name(
    arm_name,
):
    arm_name = (
        _normalize_arm_name(
            arm_name
        )
    )

    driver_name = (
        config.ARMS[
            arm_name
        ].get(
            "driver"
        )
    )

    if not isinstance(
        driver_name,
        str,
    ) or not driver_name.strip():
        raise RuntimeError(
            f"Arm '{arm_name}' "
            f"driver is missing"
        )

    return (
        driver_name
        .strip()
        .lower()
    )

# ============================================================
# Gripper Driver
# ============================================================

def get_gripper_driver(
    gripper_name,
):
    gripper_name = (
        _normalize_gripper_name(
            gripper_name
        )
    )

    with _GRIPPER_INSTANCE_LOCK:

        if gripper_name in (
            _GRIPPER_INSTANCES
        ):
            return (
                _GRIPPER_INSTANCES[
                    gripper_name
                ]
            )

        gripper_config = (
            config.GRIPPERS[
                gripper_name
            ]
        )

        driver_name = (
            gripper_config.get(
                "driver"
            )
        )

        if not isinstance(
            driver_name,
            str,
        ) or not driver_name.strip():
            raise RuntimeError(
                f"Gripper "
                f"'{gripper_name}' "
                "driver is missing"
            )

        driver_name = (
            driver_name
            .strip()
            .lower()
        )

        driver_definition = (
            GRIPPER_DRIVERS.get(
                driver_name
            )
        )

        if driver_definition is None:
            raise RuntimeError(
                f"Unsupported GRIPPER driver: "
                f"{driver_name}. "
                f"Supported drivers: "
                f"{', '.join(sorted(GRIPPER_DRIVERS))}"
            )

        kwargs = dict(
            gripper_config.get(
                "kwargs",
                {},
            )
        )

        driver = (
            _load_driver(
                driver_name=
                    driver_name,

                driver_definition=
                    driver_definition,

                driver_type=
                    "GRIPPER_DRIVER",

                kwargs=
                    kwargs,
            )
        )

        _validate_driver(
            driver=
                driver,

            driver_name=
                driver_name,

            driver_type=
                "GRIPPER_DRIVER",

            required_attributes=
                GRIPPER_ATTRIBUTES,

            required_functions=
                GRIPPER_FUNCTIONS,
        )

        _GRIPPER_INSTANCES[
            gripper_name
        ] = driver

        return driver


def get_gripper_driver_name(
    gripper_name,
):
    gripper_name = (
        _normalize_gripper_name(
            gripper_name
        )
    )

    driver_name = (
        config.GRIPPERS[
            gripper_name
        ].get(
            "driver"
        )
    )

    if not isinstance(
        driver_name,
        str,
    ) or not driver_name.strip():
        raise RuntimeError(
            f"Gripper "
            f"'{gripper_name}' "
            "driver is missing"
        )

    return (
        driver_name
        .strip()
        .lower()
    )
    
def get_camera_driver(
    camera_name,
):
    camera_name = (
        _normalize_camera_name(
            camera_name
        )
    )

    with _CAMERA_INSTANCE_LOCK:

        if camera_name in (
            _CAMERA_INSTANCES
        ):
            return (
                _CAMERA_INSTANCES[
                    camera_name
                ]
            )

        camera_config = (
            config.CAMERAS[
                camera_name
            ]
        )

        driver_name = (
            camera_config.get(
                "driver"
            )
        )

        if not isinstance(
            driver_name,
            str,
        ) or not driver_name.strip():
            raise RuntimeError(
                f"Camera '{camera_name}' "
                "driver is missing"
            )

        driver_name = (
            driver_name
            .strip()
            .lower()
        )

        driver_definition = (
            CAMERA_DRIVERS.get(
                driver_name
            )
        )

        if driver_definition is None:
            raise RuntimeError(
                f"Unsupported CAMERA driver: "
                f"{driver_name}. "
                f"Supported drivers: "
                f"{', '.join(sorted(CAMERA_DRIVERS))}"
            )

        kwargs = dict(
            camera_config.get(
                "kwargs",
                {},
            )
        )

        driver = _load_driver(
            driver_name=
                driver_name,

            driver_definition=
                driver_definition,

            driver_type=
                "CAMERA_DRIVER",

            kwargs=
                kwargs,
        )

        _validate_driver(
            driver=driver,

            driver_name=
                driver_name,

            driver_type=
                "CAMERA_DRIVER",

            required_attributes=
                CAMERA_ATTRIBUTES,

            required_functions=
                CAMERA_FUNCTIONS,
        )

        _CAMERA_INSTANCES[
            camera_name
        ] = driver

        return driver

def get_camera_driver_name(
    camera_name,
):
    camera_name = (
        _normalize_camera_name(
            camera_name
        )
    )

    driver_name = (
        config.CAMERAS[
            camera_name
        ].get(
            "driver"
        )
    )

    if not isinstance(
        driver_name,
        str,
    ) or not driver_name.strip():
        raise RuntimeError(
            f"Camera '{camera_name}' "
            "driver is missing"
        )

    return (
        driver_name
        .strip()
        .lower()
    )