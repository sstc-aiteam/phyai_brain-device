"""HTTP LeRobot dataset writer; only model_service communicates with the container."""
import base64
import threading
import uuid
from types import SimpleNamespace
import numpy as np
from services import model_service

class HttpDatasetWriter:
    def __init__(self, dataset_path, options):
        self.path = str(dataset_path)
        self.writer_id = uuid.uuid4().hex
        self._closed = False
        self._lock = threading.RLock()
        answer = model_service.lerobot.request("create", writer_id=self.writer_id,
                                                dataset_path=self.path, options=options)
        self.meta = SimpleNamespace(total_episodes=answer["total_episodes"])

    def add_frame(self, frame):
        with self._lock:
            self._require_open()
            encoded = {}
            for key, value in frame.items():
                if isinstance(value, np.ndarray):
                    array = np.ascontiguousarray(value)
                    value = {"__array__": base64.b64encode(array.tobytes()).decode("ascii"),
                             "dtype": array.dtype.str, "shape": list(array.shape)}
                encoded[key] = value
            model_service.lerobot.request("add_frame", writer_id=self.writer_id, frame=encoded)

    def save_episode(self):
        with self._lock:
            self._require_open()
            return model_service.lerobot.request("save_episode", writer_id=self.writer_id)

    def finalize(self):
        with self._lock:
            if not self._closed:
                model_service.lerobot.request("finalize", writer_id=self.writer_id)
                self._closed = True

    def abort(self):
        with self._lock:
            if not self._closed:
                try:
                    model_service.lerobot.request("abort", writer_id=self.writer_id)
                finally:
                    self._closed = True

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Dataset writer is closed")

def warmup_status():
    return {"phase": "ready", "transport": "http"}
