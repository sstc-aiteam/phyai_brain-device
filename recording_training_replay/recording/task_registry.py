"""Recording task normalization and persistent registration."""

import json
import os
import threading

def normalize_dataset_mode(value):
    mode = str(value or "multi_task").strip().lower().replace("-", "_")
    if mode not in {"multi_task", "single_task"}:
        raise ValueError("dataset_mode 必須是 multi_task 或 single_task")
    return mode


def normalize_task(task):
    normalized = " ".join(str(task or "").split()).lower()
    if not normalized:
        raise ValueError("task 不可為空")
    if len(normalized) > 200:
        raise ValueError("task 長度不可超過 200 個字元")
    return normalized


class TaskRegistry:
    def __init__(self):
        self.lock = threading.RLock()

    def read(self, path):
        if not os.path.isfile(path):
            return {"version": 1, "tasks": []}
        with open(path, encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            raise ValueError("task_registry.json 格式錯誤")
        return payload

    def snapshot(self, path):
        with self.lock:
            return self.read(path)

    def require(self, path, task):
        with self.lock:
            if not any(item.get("task") == task for item in self.read(path)["tasks"]):
                raise ValueError(f"multi-task 尚未註冊 task：{task}")

    def register(self, path, task):
        task = normalize_task(task)
        with self.lock:
            registry = self.read(path)
            existing = next((item for item in registry["tasks"] if item.get("task") == task), None)
            if existing is None:
                index = max((int(item.get("task_index", -1)) for item in registry["tasks"]), default=-1) + 1
                existing = {"task_index": index, "task": task}
                registry["tasks"].append(existing)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                temporary = f"{path}.tmp"
                with open(temporary, "w", encoding="utf-8") as destination:
                    json.dump(registry, destination, ensure_ascii=False, indent=2)
                    destination.write("\n")
                os.replace(temporary, path)
            return existing

