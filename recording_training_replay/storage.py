"""LeRobot dataset and task registry locations."""
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
LEROBOT_DATASET_DIR = str(Path(PROJECT_ROOT) / 'lerobot_datasets')
TASK_REGISTRY_PATH = str(Path(LEROBOT_DATASET_DIR) / 'task_registry.json')
