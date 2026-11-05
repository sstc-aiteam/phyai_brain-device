from copy import deepcopy

TOOLS = {

    # ============================================================
    # OPEN DRAWER
    # ============================================================

    "open_drawer": {
        "module": "action",
        "description": (
            "先將手臂移動至預設的開抽屜初始姿態，"
            "接著透過相機影像、手臂姿態與夾爪狀態進行推論，"
            "自動產生手臂軌跡與夾爪動作，抓住黑色抽屜把手並拉開抽屜。"
            "當使用者要求開啟抽屜、拉開抽屜或打開抽屜時使用。"
        ),
        "parameters": {
            "arm_name": {
                "type": "string",
                "description": "執行動作的機器手臂名稱，例如 right。",
            },
            "gripper_name": {
                "type": "string",
                "description": "執行動作的夾爪名稱，例如 right。",
            },
            "camera_name": {
                "type": "string",
                "description": "SmolVLA 使用的相機名稱，例如 right。",
            },
        },
        "required": ["arm_name", "gripper_name", "camera_name"],
        "target_types": ["robot_arm", "gripper", "camera"],
        "required_capabilities": ["arm_motion", "gripper_control", "camera_capture", "smolvla_inference"],
        "required_zones": [],
    },

    # ============================================================
    # CLOSE DRAWER
    # ============================================================

    "close_drawer": {
        "module": "action",
        "description": (
            "先將手臂移動至預設的關抽屜初始姿態，"
            "接著透過相機影像、手臂姿態與夾爪狀態進行推論，"
            "抓住黑色抽屜把手，將抽屜部分推回，"
            "放開把手後再推動灰色抽屜面板，嘗試將抽屜完全關閉。"
            "當使用者要求關閉抽屜、推回抽屜或把抽屜關上時使用。"
            "此功能不負責將物體放入抽屜，也不保證能自動判斷抽屜是否完全關閉。"
        ),
        "parameters": {
            "arm_name": {
                "type": "string",
                "description": "執行動作的機器手臂名稱，例如 right。",
            },
            "gripper_name": {
                "type": "string",
                "description": "執行動作的夾爪名稱，例如 right。",
            },
            "camera_name": {
                "type": "string",
                "description": "SmolVLA 使用的相機名稱，例如 right。",
            },
        },
        "required": ["arm_name", "gripper_name", "camera_name"],
        "target_types": ["robot_arm", "gripper", "camera"],
        "required_capabilities": ["arm_motion", "gripper_control", "camera_capture", "smolvla_inference"],
        "required_zones": [],
    },

    # ============================================================
    # OBJECT
    # ============================================================

    "pick_object": {
        "module": "action",
        "description": (
            "使用機器手臂與夾爪夾取指定位置的物體。"
            "系統會先打開夾爪，再將夾爪移動至指定的 Robot Base XYZ 座標，"
            "接著關閉夾爪，並透過夾爪的 object_detected 狀態確認是否抓到物體。"
            "當使用者要求抓取、夾取或拿起指定位置的物體時使用。"
            "此功能不負責將夾取的物體搬運或放置到其他位置。"
        ),
        "parameters": {
            "arm_name": {
                "type": "string",
                "description": "執行動作的機器手臂名稱，例如 right。",
            },
            "gripper_name": {
                "type": "string",
                "description": "執行動作的夾爪名稱，例如 right。",
            },
            "x": {
                "type": "number",
                "description": "物體在 Robot Base 座標系下的 X 座標，單位為公尺。",
            },
            "y": {
                "type": "number",
                "description": "物體在 Robot Base 座標系下的 Y 座標，單位為公尺。",
            },
            "z": {
                "type": "number",
                "description": "物體在 Robot Base 座標系下的 Z 座標，單位為公尺。",
            },
        },
        "required": ["arm_name", "gripper_name", "x", "y", "z"],
        "target_types": ["robot_arm", "gripper"],
        "required_capabilities": ["arm_motion", "gripper_control"],
        "required_zones": [],
    },

    "place_object": {
        "module": "action",
        "description": (
            "使用機器手臂與夾爪，將目前夾持的物體放置到抽屜內。"
            "當使用者要求將物品放入抽屜、把手上的東西放到抽屜裡或將物體放置到抽屜內時使用。"
            "此不負責開啟或關閉抽屜，且不會自動確認物體是否成功放置。"
        ),
        "parameters": {
            "arm_name": {
                "type": "string",
                "description": "執行物體放置動作的機器手臂名稱，例如 left。"
            },
            "gripper_name": {
                "type": "string",
                "description": "執行物體釋放動作的夾爪名稱，例如 left。"
            }
        },
        "required": ["arm_name", "gripper_name"],
        "target_types": ["robot_arm", "gripper"],
        "required_capabilities": ["arm_motion", "gripper_control"],
        "required_zones": []
    },

    "move_arm_direction": {
        "module": "action",
        "description": (
            "控制指定機器手臂上的相機朝特定方向或目標區域觀察環境。"
            "系統會將手臂移動至預先設定的關節姿態，"
            "支援前、左、後、右、上、下六個方向，以及椅子觀察位置。"
            "當使用者要求觀察環境、往某個方向看、查看椅子、觀察椅子或查看周圍環境時使用。"
            "此功能只負責移動相機觀察姿態，不會自動辨識或分析影像內容。"
        ),
        "parameters": {
            "arm_name": {
                "type": "string",
                "description": "執行觀察動作的機器手臂名稱，例如 left 或 right。"
            },
            "direction": {
                "type": "string",
                "enum": ["front", "left", "back", "right", "up", "down", "chair"],
                "description": "觀察方向或目標區域：前、左、後、右、上、下、椅子（chair）。"
            }
        },
        "required": ["arm_name", "direction"],
        "target_types": ["robot_arm", "camera"],
        "required_capabilities": ["arm_motion", "camera_mount"],
        "required_zones": []
    },
}


# ============================================================
# PLANNING METADATA
# ============================================================

for _tool_definition in TOOLS.values():
    _tool_definition.setdefault("target_types", ["robot_arm"])
    _tool_definition.setdefault("required_capabilities", ["arm_motion"])
    _tool_definition.setdefault("required_zones", [])


# ============================================================
# GET TOOL
# ============================================================

def get_tool(name=None):

    if name is None:
        return deepcopy(TOOLS)

    if not isinstance(name, str):
        return None

    return deepcopy(TOOLS.get(name))


# ============================================================
# GET TOOLS FOR PROMPT
# ============================================================

def get_tools_for_prompt():

    result = []

    for function_name, definition in TOOLS.items():
        result.append({
            "api_function": function_name,
            "module": definition["module"],
            "description": definition["description"],
            "parameters": definition["parameters"],
            "required": definition["required"],
            "target_types": definition["target_types"],
            "required_capabilities": definition["required_capabilities"],
            "required_zones": definition["required_zones"],
        })

    return result
