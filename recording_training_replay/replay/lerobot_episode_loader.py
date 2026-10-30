"""HTTP episode loader (PyArrow executes only in the model runtime)."""
from services import model_service

def load_lerobot_v3_episode(input_path, episode_index=0):
    result = model_service.lerobot.request("episode", dataset_path=str(input_path),
                                            episode_index=int(episode_index))
    return result["dataset_path"], result["interval"], result["trajectory"], result["events"]
