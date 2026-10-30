"""Replay catalog facade: all LeRobot dataset filesystem access occurs in the runtime."""
from recording_training_replay.storage import LEROBOT_DATASET_DIR as DEFAULT_DATASET_DIR
from recording_training_replay.dataset import read_dataset_info, episode_details, save_episode_devices, catalog
from utils.response import success, error
from services import model_service

MODULE = "recording_training_replay"

def _normalize_input_path(input_path):
    return model_service.lerobot.request("resolve", dataset_path=str(input_path))["dataset_path"]

def _parquet_episode_details(dataset_path, info=None):
    return episode_details(dataset_path)

def get_replay_catalog():
    try:
        import config
        datasets = catalog()
        for entry in datasets:
            entry["compatible_arms"] = [name for name, arm in config.ARMS.items()
                if arm.get("driver") == entry.get("robot_type")]
        return success(MODULE, "get_replay_catalog", data={
            "dataset_root": DEFAULT_DATASET_DIR, "datasets": datasets})
    except Exception as exc:
        return error(MODULE, "get_replay_catalog", error=exc,
                     error_type=type(exc).__name__)
