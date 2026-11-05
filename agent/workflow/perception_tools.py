"""Independent Perception Tool registry (NOT action/tools.py).

Other teams may maintain or replace this separate registry without altering
Agent's action tool whitelist.
"""
from copy import deepcopy

TOOLS = {
    "get_detections_value": {
        "module": "perception",
        "description": (
            "以 YOLO 偵測指定相機目前可見的小物件，回傳物件類別、confidence、"
            "Robot Base XYZ。需要找到指定目標並取得抓取座標時使用。"
            "target_class 必須是 YOLO 實際 class_name（例如 disposable_mask 表示一次性口罩）。"
            "本工具不移動機械手臂，不呼叫 VLM。"
        ),
        "parameters": {
            "camera_name": {"type": "string", "description": "相機，例如 left。"},
            "target_class": {"type": "string", "description": "欲尋找的 YOLO class_name，例如 disposable_mask。"},
        },
        "required": ["camera_name", "target_class"],
    }
}


def get_tool(name=None):
    return deepcopy(TOOLS if name is None else TOOLS.get(name))
