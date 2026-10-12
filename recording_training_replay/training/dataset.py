"""Read training dataset metadata and discover v3 datasets."""

import json
import os

def read_dataset_info(dataset_path):
    path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)

def training_dataset_catalog(dataset_root):
    datasets = []
    for directory, _, filenames in os.walk(dataset_root):
        if os.path.basename(directory) != "meta" or "info.json" not in filenames:
            continue
        dataset_path = os.path.dirname(directory)
        info = read_dataset_info(dataset_path) or {}
        if info.get("codebase_version") != "v3.0":
            continue
        if not any(name.endswith(".parquet") for _, _, names in os.walk(os.path.join(dataset_path, "data")) for name in names):
            continue
        datasets.append({"dataset_path": dataset_path, "task": os.path.basename(dataset_path),
                         "format": "v3", "total_episodes": info.get("total_episodes", 0)})
    return sorted(datasets, key=lambda item: item["dataset_path"])


