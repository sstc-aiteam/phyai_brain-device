"""
AI Agent 可使用的工具白名單。
"""

from copy import deepcopy

TOOLS = {

    # -------------------------
    # Arm motion
    # -------------------------
    "move_right_arm_initial": {
        "module": "arm",
        "description": (
            "只將右側 UR5 機器手臂移動到系統設定的初始位置。"
            "使用者要求右手臂或 UR5 回到初始位置時使用；"
            "不可用於左側 UR7e。"
            "不可用於沿 X、Y、Z 軸移動指定距離；"
            "沿軸移動必須使用 move_arm_step。"
        ),
        "parameters": {},
        "required": [],
    },

    "move_arm_default": {
        "module": "arm",
        "description": (
            "只將機器手臂移動到系統設定的預設姿態。"
            "僅在使用者明確要求回到預設位置、預設姿態或待命姿態時使用。"
            "不可用於沿 X、Y、Z 軸移動，也不可用於移動指定距離；"
            "沿軸移動指定距離必須使用 move_arm_step。"
        ),
        "parameters": {},
        "required": [],
    },

    "move_arm_step": {
        "module": "arm",
        "description": (
            "讓機器手臂沿 X、Y 或 Z 軸移動指定距離。"
            "使用者提到 x+、x-、y+、y-、z+、z-，或要求沿某一軸"
            "移動幾公尺／公分時，必須使用此工具。"
            "每一個獨立移動步驟都必須各自提供 direction；"
            "不可改用 move_arm_default 或 move_right_arm_initial。"
        ),
        "parameters": {
            "direction": {
                "type": "string",
                "enum": ["x+", "x-", "y+", "y-", "z+", "z-"],
                "description": "移動方向。",
            },
            "distance": {
                "type": "number",
                "minimum": 0.001,
                "maximum": 0.5,
                "default": 0.01,
                "description": "移動距離，單位為公尺；可省略，預設為0.01",
            },
            "speed": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 0.25,
                "default": 0.1,
                "description": "移動速度，單位為 m/s；可省略，預設為 0.1。",
            },
        },
        "required": ["direction"]
    },

    # -------------------------
    # tasks
    # -------------------------
    "place_object_in_trash_can": {
        "module": "task",
        "description": (
            "把已選定的物件丟到垃圾桶中時，使用這個功能。"
            "此功能只負責夾取與放置，不負責打開或關閉垃圾桶蓋。"
            "輸入物件在 Robot Base 座標系下的位置 point_xyz。"
            "手臂會自動產生關節軌跡、夾取物件並放置至垃圾桶內。"
        ),
        "parameters": {
            "point_xyz": {
                "type": "array",
                "items": {
                    "type": "number",
                },
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "物件在 Robot Base 座標系下的 XYZ 座標，"
                    "應直接從物件辨識結果中的 robot_xyz 讀取。"
                ),
            },
            "yaw_deg": {
                "type": "number",
                "minimum": -180.0,
                "maximum": 180.0,
                "default": 0.0,
                "description": (
                    "物件擺放的角度，"
                    "應直接從物件辨識結果中的 yaw_deg 讀取。"
                ),
            },
        },
        "required": ["point_xyz", "yaw_deg"],
    },

    "place_object_in_top_cabinet": {
        "module": "task",
        "description": (
            "把已選定的物件放置到櫃子時，使用這個功能。"
            "此功能只負責夾取與放置，不負責打開或關閉櫃門。"
            "輸入物件在 Robot Base 座標系下的 XYZ 座標。"
            "手臂會自動產生關節軌跡、夾取物件並放置至櫃子內。"
        ),
        "parameters": {
            "point_xyz": {
                "type": "array",
                "items": {
                    "type": "number",
                },
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "物件在 Robot Base 座標系下的 XYZ 座標，"
                    "應直接從物件辨識結果中的 robot_xyz 讀取。"
                ),
            },
            "yaw_deg": {
                "type": "number",
                "minimum": -180.0,
                "maximum": 180.0,
                "default": 0.0,
                "description": (
                    "物件擺放的角度，"
                    "應直接從物件辨識結果中的 yaw_deg 讀取。"
                ),
            },
        },
        "required": ["point_xyz", "yaw_deg"],
    },

    "place_object_in_second_drawer": {
        "module": "task",
        "description": (
            "把已選定的物件放置到抽屜時，使用這個功能。"
            "此功能只負責夾取與放置，不負責打開或關閉抽屜。"
            "輸入物件在 Robot Base 座標系下的 XYZ 座標。"
            "手臂會自動產生關節軌跡、夾取物件並放置至抽屜內。"
        ),
        "parameters": {
            "point_xyz": {
                "type": "array",
                "items": {
                    "type": "number",
                },
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "物件在 Robot Base 座標系下的 XYZ 座標，"
                    "應直接從物件辨識結果中的 robot_xyz 讀取。"
                ),
            },
            "yaw_deg": {
                "type": "number",
                "minimum": -180.0,
                "maximum": 180.0,
                "default": 0.0,
                "description": (
                    "物件擺放的角度，"
                    "應直接從物件辨識結果中的 yaw_deg 讀取。"
                ),
            },
        },
        "required": ["point_xyz", "yaw_deg"],
    },

    "open_trash_can": {
        "module": "task",
        "description": "直接打開垃圾桶，不需要放置物件時，執行此功能",
        "parameters": {},
        "required": [],
    },

    "close_trash_can": {
        "module": "task",
        "description": "直接關閉垃圾桶，不需要放置物件時，執行此功能",
        "parameters": {},
        "required": [],
    },

    "open_top_cabinet": {
        "module": "task",
        "description": (
            "直接開門板，不需要放置物件時執行。"
            "此工具不需要任何物件辨識資訊，parameters 為空。"
        ),
        "parameters": {},
        "required": [],
    },

    "close_top_cabinet": {
        "module": "task",
        "description": (
            "直接關閉門板，不需要放置物件時執行。"
            "此工具不需要任何物件辨識資訊，parameters 為空。"
        ),
        "parameters": {},
        "required": [],
    },

    "open_second_drawer": {
        "module": "task",
        "description": "直接打開抽屜，不需要放置物件時，執行此功能",
        "parameters": {},
        "required": [],
    },

    "close_second_drawer": {
        "module": "task",
        "description": "直接關閉抽屜，不需要放置物件時，執行此功能",
        "parameters": {},
        "required": [],
    },
}


# Planning metadata is kept next to the existing execution whitelist so the
# validator, allocator and LLM all read the same tool catalog. Current tools
# are arm operations. Reachable zones remain empty until calibrated workspace
# data is available; no left/right reachability is guessed here.
for _tool_definition in TOOLS.values():
    _tool_definition.setdefault("target_types", ["robot_arm"])
    _tool_definition.setdefault("required_capabilities", ["arm_motion"])
    _tool_definition.setdefault("required_zones", [])


def get_tool(name=None):
    """
    name 為 None：回傳全部工具定義副本。
    name 為有效字串且工具存在：回傳指定工具定義副本。
    name 格式錯誤或工具不存在：回傳 None。
    """
    if name is None:
        return deepcopy(TOOLS)

    if not isinstance(name, str):
        return None

    return deepcopy(TOOLS.get(name))


def get_tools_for_prompt():
    """
    產生提供給 LLM 的精簡工具描述。

    不把 Python callable 或內部實作暴露給模型。
    """
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
