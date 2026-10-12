"""Discover completed episodes and build the replay catalog."""

import json
import os
import re
import numpy as np

def raw_episode_details(dataset_path):
    details = []
    for name in sorted(os.listdir(dataset_path)):
        match = re.fullmatch(r"episode_(\d+)", name)
        if not match:
            continue
        episode_path = os.path.join(dataset_path, name)
        metadata_path = os.path.join(episode_path, "episode.json")
        samples_path = os.path.join(episode_path, "frames.csv")
        if not os.path.isfile(metadata_path):
            metadata_path = os.path.join(episode_path, "metadata.json")
            samples_path = os.path.join(episode_path, "samples.jsonl")
        if not os.path.isfile(metadata_path) or not os.path.isfile(samples_path):
            continue
        try:
            with open(metadata_path, encoding="utf-8") as source:
                metadata = json.load(source)
            if metadata.get("format") != "raw" or metadata.get("complete") is not True:
                continue
            if metadata.get("status", "completed") != "completed":
                continue
            count = int(metadata.get("sample_count", 0))
            interval = float(metadata.get("interval", 0))
            if count <= 0 or not np.isfinite(interval) or interval <= 0:
                continue
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        details.append({
            "episode_index": int(match.group(1)), "label": name,
            "task": metadata.get("task"), "episode_path": episode_path,
            "source_filename": name, "metadata": metadata,
        })
    return details



def replay_dataset_catalog(raw_root, training_root, available_arms):
    datasets = []
    from itertools import chain
    for directory, dirnames, filenames in chain.from_iterable(os.walk(root) for root in dict.fromkeys((raw_root, training_root))):
        episode_details = raw_episode_details(directory)
        if not episode_details:
            continue
        dataset_path = directory
        info = episode_details[0]["metadata"]
        dataset_format = "raw"
        robot_types = sorted({item["metadata"].get("robot_type") for item in episode_details
                              if item["metadata"].get("robot_type")})
        robot_type = robot_types[0] if len(robot_types) == 1 else " / ".join(robot_types)
        compatible_arms = [
            arm.get("arm_name") for arm in available_arms
            if arm.get("driver") in robot_types
        ]
        relative_path = os.path.relpath(dataset_path, raw_root)
        task = os.path.basename(dataset_path)
        datasets.append({
            "task_id": relative_path,
            "task": task,
            "dataset_path": dataset_path,
            "relative_path": relative_path,
            "format": dataset_format,
            "robot_type": robot_type,
            "compatible_arms": compatible_arms,
            "fps": 1.0 / float(info["interval"]),
            "total_episodes": len(episode_details),
            "dataset_total_episodes": len(episode_details),
            "total_frames": sum(int(item["metadata"]["sample_count"]) for item in episode_details),
            "episodes": [item["episode_index"] for item in episode_details],
            "episode_details": episode_details,
        })
        dirnames[:] = []
    datasets.sort(key=lambda item: item["relative_path"])
    return datasets
