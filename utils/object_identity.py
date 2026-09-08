"""物件識別欄位的共用工具。"""

import re
import threading
from copy import deepcopy


OBJECT_ID_KEYS = (
    "object_id",
    "track_id",
    "id",
    "instance_id",
)


def get_existing_object_id(object_data):
    """依共用優先順序取得上游已提供的物件 ID。"""
    if not isinstance(object_data, dict):
        return None

    for key in OBJECT_ID_KEYS:
        value = object_data.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized

    return None


def assign_request_object_ids(objects):
    """保留有效上游 ID；缺少或重複時配置帶類別名稱的 request ID。"""
    if not isinstance(objects, list):
        raise TypeError("objects 必須是 list")

    assigned = []
    used_ids = set()
    class_counts = {}

    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            assigned.append(obj)
            continue

        object_id = get_existing_object_id(obj)
        if object_id is None or object_id in used_ids:
            class_name = str(obj.get("class_name") or "object").strip().lower()
            base_name = re.sub(r"[^a-z0-9_]+", "_", class_name).strip("_")
            if not base_name:
                base_name = "object"

            suffix = class_counts.get(base_name, 0) + 1
            object_id = f"{base_name}_{suffix}"
            while object_id in used_ids:
                suffix += 1
                object_id = f"{base_name}_{suffix}"
            class_counts[base_name] = suffix

        used_ids.add(object_id)
        assigned.append({
            **obj,
            "object_id": object_id,
        })

    return assigned


def _class_base_name(obj):
    class_name = str(obj.get("class_name") or "object").strip().lower()
    return re.sub(r"[^a-z0-9_]+", "_", class_name).strip("_") or "object"


def _box_xyxy(obj):
    box = obj.get("box") or obj.get("bbox")
    if isinstance(box, dict):
        values = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        values = list(box[:4])
    else:
        return None
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    x1, y1, x2, y2 = [float(value) for value in values]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _iou(left, right):
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


class StableObjectIdentityAssigner:
    """Assign class-scoped IDs and retain them across nearby observations."""

    def __init__(self, *, iou_threshold=0.25, max_missed_observations=5):
        self.iou_threshold = float(iou_threshold)
        self.max_missed_observations = int(max_missed_observations)
        self._lock = threading.Lock()
        self._tracks = {}
        self._class_counters = {}

    def assign(self, objects):
        if not isinstance(objects, list):
            raise TypeError("objects 必須是 list")
        with self._lock:
            return self._assign_locked(objects)

    def _assign_locked(self, objects):
        for track in self._tracks.values():
            track["missed"] += 1

        assigned = []
        used_track_ids = set()
        for raw in objects:
            if not isinstance(raw, dict):
                assigned.append(raw)
                continue
            obj = deepcopy(raw)
            existing_id = get_existing_object_id(obj)
            box = _box_xyxy(obj)
            class_name = _class_base_name(obj)
            object_id = existing_id

            if object_id is None and box is not None:
                candidates = []
                for track_id, track in self._tracks.items():
                    if track_id in used_track_ids or track["class_name"] != class_name:
                        continue
                    score = _iou(box, track["box"])
                    if score >= self.iou_threshold:
                        candidates.append((score, track_id))
                if candidates:
                    _score, object_id = max(candidates, key=lambda item: (item[0], item[1]))

            if object_id is None:
                suffix = self._class_counters.get(class_name, 0) + 1
                object_id = f"{class_name}_{suffix}"
                while object_id in self._tracks:
                    suffix += 1
                    object_id = f"{class_name}_{suffix}"
                self._class_counters[class_name] = suffix

            if box is not None:
                self._tracks[object_id] = {
                    "class_name": class_name,
                    "box": box,
                    "missed": 0,
                }
                used_track_ids.add(object_id)
            obj["object_id"] = object_id
            assigned.append(obj)

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track["missed"] <= self.max_missed_observations
        }
        return assigned
