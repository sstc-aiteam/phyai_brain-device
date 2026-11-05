"""Agent capability catalog.

Metadata only: importing this module will not call any service, model, or robot.
The Agent may select capabilities here. Actual dispatch is implemented separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    description: str
    service_module: str
    service_function: str
    parameters_schema: dict[str, Any]
    side_effect: str  # 'none' or 'may_move_robot'
    execution_enabled: bool = True

    def routing_view(self) -> dict[str, Any]:
        """The metadata the LLM may see; no callable imports are exposed."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters_schema": self.parameters_schema,
            "side_effect": self.side_effect,
            "execution_enabled": self.execution_enabled,
        }


def _object_schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_CAPABILITIES: dict[str, CapabilitySpec] = {
    "describe_scene": CapabilitySpec(
        name="describe_scene",
        description=(
            "使用相機與 VLM 描述目前畫面、場景或可見事物。"
            "用於『你現在看到什麼』等開放式視覺描述問題。"
        ),
        service_module="services.worldmodel_service",
        service_function="describe_scene",
        parameters_schema=_object_schema({
            "camera_name": {"type": "string", "description": "指定相機名稱；未指定則使用 Service 預設值。"},
            "vlm_backend": {"type": "string", "description": "指定場景描述後端；未指定則使用 Service 預設值。"},
        }),
        side_effect="none",
    ),
    "get_object_state": CapabilitySpec(
        name="get_object_state",
        description=(
            "取得目前偵測到的物件清單與物件本身狀態，包括位置、辨識框、"
            "方向、信心度與材質；這個 Service 會啟用材質分析。"
        ),
        service_module="services.worldmodel_service",
        service_function="get_object_state",
        parameters_schema=_object_schema({
            "camera_names": {
                "type": "array", "items": {"type": "string"}, "minItems": 1,
                "uniqueItems": True,
                "description": "相機名稱清單；在 Executor 轉為 Python tuple。未指定時沿用 Service 預設。",
            },
        }),
        side_effect="none",
    ),
    "get_arm_status": CapabilitySpec(
        name="get_arm_status",
        description=(
            "讀取手臂目前狀態。未指定 arm_name 時取得所有設定的手臂，"
            "可由 data.arms[].status.connected 統計已連線數量，"
            "並讀取 ready、moving 等資訊。"
        ),
        service_module="services.arm_service",
        service_function="get_arm_status",
        parameters_schema=_object_schema({
            "arm_name": {"type": "string", "description": "只查指定手臂；未指定則查全部。"},
        }),
        side_effect="none",
    ),
    "get_camera_status": CapabilitySpec(
        name="get_camera_status",
        description=(
            "查詢設定的相機清單、數量、連線與串流狀態。"
            "camera_name 未指定時查全部相機；這不會發出 start_camera/stop_camera 指令。"
            "適用『有幾台相機』『左相機連線嗎』『哪台相機正在運作』。"
        ),
        service_module="services.camera_service",
        service_function="get_camera_status",
        parameters_schema=_object_schema({
            "camera_name": {"type": "string", "minLength": 1,
                            "description": "指定相機名稱；未指定時查詢全部。"},
        }),
        side_effect="none",
    ),
    "get_gripper_status": CapabilitySpec(
        name="get_gripper_status",
        description=(
            "查詢設定的夾爪清單、連線、ready、位置與物件夾持偵測（可能為 null）。"
            "gripper_name 未指定時查全部夾爪。不會開關夾爪。"
        ),
        service_module="services.gripper_service",
        service_function="get_gripper_status",
        parameters_schema=_object_schema({
            "gripper_name": {"type": "string", "minLength": 1,
                             "description": "指定夾爪名稱；未指定時查詢全部。"},
        }),
        side_effect="none",
    ),
    "get_arm_pose": CapabilitySpec(
        name="get_arm_pose",
        description=(
            "讀取機械手臂當前 TCP pose；未指定 arm_name 時，"
            "Service 只回傳能取得姿態的手臂，不代表全部設定的手臂。"
            "不會移動手臂。"
        ),
        service_module="services.arm_service",
        service_function="get_arm_pose",
        parameters_schema=_object_schema({
            "arm_name": {"type": "string", "minLength": 1,
                         "description": "指定手臂名稱；未指定時取得所有可讀取姿態的手臂。"},
        }),
        side_effect="none",
    ),
    "get_arm_joints": CapabilitySpec(
        name="get_arm_joints",
        description=(
            "讀取機械手臂目前的關節值；未指定 arm_name 時，"
            "Service 只回傳能取得關節值的手臂。不會移動手臂。"
        ),
        service_module="services.arm_service",
        service_function="get_arm_joints",
        parameters_schema=_object_schema({
            "arm_name": {"type": "string", "minLength": 1,
                         "description": "指定手臂名稱；未指定時取得所有可讀取關節值的手臂。"},
        }),
        side_effect="none",
    ),
    "plan_task": CapabilitySpec(
        name="plan_task",
        description=(
            "由 Agent 發起動作任務（例如開抽屜、手臂回預設位置、搬運、抓取等）。"
            "Plan Service 會自行透過 WorldModel 取得場景資訊；Agent 不必先呼叫 VLM。"
            "目前動作執行僅支援 open_drawer / move_arm_default ；"
            "其他任務由 Plan 明確拒絕，不會執行實機。"
        ),
        service_module="services.plan_service",
        service_function="plan_task",
        parameters_schema=_object_schema({}),
        side_effect="none",  # Plan Service itself is read-only.
        execution_enabled=True,  # Dedicated Agent runner enforces mock-only Action.
    ),
    "run_manual_runtime": CapabilitySpec(
        name="run_manual_runtime",
        description=(
            "啟動閉環機器人任務，Runtime 自行建立初始 WorldState、"
            "理解任務並以 Manual Executor 進行後續動作流程。"
            "可能會移動觀察手臂；不是一般資訊查詢。"
        ),
        service_module="services.plan_service",
        service_function="start_manual_runtime",
        parameters_schema=_object_schema({
            "user_text": {"type": "string", "minLength": 1, "description": "使用者原始任務指令。"},
            "planning_options": {"type": "object", "description": "保留原有 Runtime Planning Options。"},
            "max_iterations": {"type": "integer", "minimum": 1},
            "force_device_id": {"type": "string"},
        }, required=("user_text",)),
        side_effect="may_move_robot",
        execution_enabled=False,  # Selection only until explicit integration and safety checks.
    ),
}


def list_capabilities() -> list[CapabilitySpec]:
    return list(_CAPABILITIES.values())


def get_capability(name: str) -> CapabilitySpec:
    return _CAPABILITIES[name]


def routing_catalog() -> list[dict[str, Any]]:
    return [item.routing_view() for item in list_capabilities() if item.name != "run_manual_runtime"]


__all__ = ["CapabilitySpec", "get_capability", "list_capabilities", "routing_catalog"]
