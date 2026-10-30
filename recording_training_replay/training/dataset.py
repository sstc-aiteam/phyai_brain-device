"""Training dataset discovery via the model runtime."""
from recording_training_replay.dataset import read_dataset_info, catalog

def training_dataset_catalog(dataset_root=None):
    return [{"dataset_path": entry["dataset_path"], "task": entry["task"],
             "format": "v3", "total_episodes": entry["total_episodes"]} for entry in catalog()]
