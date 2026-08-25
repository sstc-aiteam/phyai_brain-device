"""物件識別欄位的輕量共用工具。"""

import re


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
