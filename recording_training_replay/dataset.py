"""Dataset metadata facade. Container owns disk I/O."""
from services import model_service

def read_dataset_info(dataset_path):
    return model_service.lerobot.request("info", dataset_path=str(dataset_path)).get("info")

def save_episode_devices(dataset_path, episode_index, arm_name, gripper_name, task):
    return model_service.lerobot.request("save_devices", dataset_path=str(dataset_path),
        episode_index=int(episode_index), arm_name=arm_name, gripper_name=gripper_name, task=task)

def episode_details(dataset_path):
    return model_service.lerobot.request("details", dataset_path=str(dataset_path))["episodes"]

def catalog():
    return model_service.lerobot.request("catalog")["datasets"]
