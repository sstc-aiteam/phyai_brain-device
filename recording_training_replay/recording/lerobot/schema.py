"""LeRobot feature schema and state/action conversion."""
import numpy as np

# Default names for the robot's arm, gripper, and camera components.
LEROBOT_STATE_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper",
]
LEROBOT_TCP_POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
LEROBOT_TCP_ACTION_NAMES = [
    "tcp_local_delta_x", "tcp_local_delta_y", "tcp_local_delta_z",
    "tcp_local_delta_rx", "tcp_local_delta_ry", "tcp_local_delta_rz", "gripper",
]
LEROBOT_TCP_BASE_ACTION_NAMES = [
    "tcp_base_delta_x", "tcp_base_delta_y", "tcp_base_delta_z",
    "tcp_base_delta_rx", "tcp_base_delta_ry", "tcp_base_delta_rz", "gripper",
]
LEROBOT_JOINT_ACTION_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper",
]


ARM_IMAGE_KEY = "observation.images.arm"


def _arm_camera_images(camera_frames):
    """Expose the selected arm camera without encoding its physical side."""
    if len(camera_frames) > 1:
        raise ValueError("observation.images.arm 僅支援單一手臂相機")
    return camera_frames.values()


def _lerobot_features(camera_frames):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_STATE_NAMES,
        },
        "observation.tcp_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_TCP_POSE_NAMES,
        },
        "action_tcp": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_TCP_ACTION_NAMES,
        },
        "action_tcp_base": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_TCP_BASE_ACTION_NAMES,
        },
        "action_joints": {
            "dtype": "float32",
            "shape": (7,),
            "names": LEROBOT_JOINT_ACTION_NAMES,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }
    for image in _arm_camera_images(camera_frames):
        height, width = image.shape[:2]
        features[ARM_IMAGE_KEY] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def tcp_local_delta(current_pose, next_pose):
    import cv2

    current = np.asarray(current_pose, dtype=np.float64)
    following = np.asarray(next_pose, dtype=np.float64)
    current_rotation, _ = cv2.Rodrigues(current[3:6])
    next_rotation, _ = cv2.Rodrigues(following[3:6])

    # Express both translation and rotation in the current TCP/tool frame.
    # Deployment must map local translation back with R_current @ delta_local.
    base_translation = following[:3] - current[:3]
    local_translation = current_rotation.T @ base_translation
    relative_rotation = current_rotation.T @ next_rotation
    rotation_delta, _ = cv2.Rodrigues(relative_rotation)
    return np.concatenate([
        local_translation,
        rotation_delta.reshape(3),
    ]).astype(np.float32)


def tcp_base_delta(current_pose, next_pose):
    import cv2

    current = np.asarray(current_pose, dtype=np.float64)
    following = np.asarray(next_pose, dtype=np.float64)
    current_rotation, _ = cv2.Rodrigues(current[3:6])
    next_rotation, _ = cv2.Rodrigues(following[3:6])

    # Express both translation and rotation in the robot base frame.
    base_translation = following[:3] - current[:3]
    relative_rotation = next_rotation @ current_rotation.T
    rotation_delta, _ = cv2.Rodrigues(relative_rotation)
    return np.concatenate([
        base_translation,
        rotation_delta.reshape(3),
    ]).astype(np.float32)


def sample_to_lerobot_frame(sample, next_sample, task, done=False):
    state = np.asarray(sample["joints"] + [sample["gripper"]], dtype=np.float32)
    tcp_pose = np.asarray(sample["tcp_pose"] + [sample["gripper"]], dtype=np.float32)
    delta = tcp_local_delta(sample["tcp_pose"], next_sample["tcp_pose"])
    action_tcp = np.concatenate([
        delta,
        np.asarray([next_sample["gripper"]], dtype=np.float32),
    ])
    base_delta = tcp_base_delta(sample["tcp_pose"], next_sample["tcp_pose"])
    action_tcp_base = np.concatenate([
        base_delta,
        np.asarray([next_sample["gripper"]], dtype=np.float32),
    ])
    action_joints = np.asarray(
        next_sample["joints"] + [next_sample["gripper"]],
        dtype=np.float32,
    )
    frame = {
        "observation.state": state,
        "observation.tcp_pose": tcp_pose,
        "action_tcp": action_tcp,
        "action_tcp_base": action_tcp_base,
        "action_joints": action_joints,
        "next.done": np.atleast_1d(np.bool_(done)),
        "task": task,
    }
    for image in _arm_camera_images(sample["images"]):
        # Camera drivers expose BGR; LeRobot policies expect RGB.
        frame[ARM_IMAGE_KEY] = image[:, :, ::-1].copy()
    return frame


