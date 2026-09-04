"""Agent 可使用的設備註冊與公開能力描述。

本模組可將現有 config.ARMS 轉成通用設備資料，也支援測試或
未來 AGV、IO controller 等設備直接注入。

公開給 LLM 的資料不包含 IP、憑證或 Python callable。
Runtime status 與設備能力定義分離：
- enabled: config 是否允許 planner 使用
- connected: runtime 是否實際連線
- available: 現在是否可被 allocator 派任務
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import config


DEFAULT_ARM_CAPABILITIES = {
    "arm_motion",
}

PUBLIC_DEVICE_FIELDS = {
    "id",
    "type",
    "enabled",
    "capabilities",
    "reachable_zones",
    "end_effector",
}


class DeviceRegistryError(ValueError):
    """Raised when a device registry definition is invalid."""


def _read_robot_arm_status(
    device: dict[str, Any],
) -> dict[str, Any]:
    """Read one configured arm status without issuing motion."""
    from services import arm_service

    backend_name = device.get("backend_name")

    if (
        not isinstance(backend_name, str)
        or not backend_name.strip()
    ):
        raise DeviceRegistryError(
            f"device {device.get('id')} 缺少 backend_name"
        )

    backend_name = backend_name.strip()

    response = arm_service.get_arm_status(
        arm_name=backend_name,
    )

    if not isinstance(response, dict):
        raise DeviceRegistryError(
            "arm service 回傳格式錯誤"
        )

    # 新格式：
    #
    # {
    #     "arms": [...]
    # }
    #
    # 舊格式相容：
    #
    # {
    #     "data": {
    #         "arms": [...]
    #     }
    # }

    arms = response.get("arms")

    if not isinstance(arms, list):
        data = response.get("data")

        if isinstance(data, dict):
            arms = data.get("arms")

    if not isinstance(arms, list):
        arms = []

    # 不可使用 arms[0]。
    #
    # 即使 get_arm_status(arm_name=...)
    # 未來改成回傳所有 arm，也必須用 backend_name
    # 精確找到對應設備。
    entry = next(
        (
            row
            for row in arms
            if (
                isinstance(row, dict)
                and row.get("arm_name")
                == backend_name
            )
        ),
        None,
    )

    if entry is None:
        return {
            "connected": False,
            "available": False,
            "ready": None,
            "moving": None,
            "protective_stop": None,
            "emergency_stop": None,
            "fault": None,
            "program_running": None,
            "arm_mode": None,
            "safety_mode": None,
            "pose": None,
            "joints": None,
            "error": (
                f"arm {backend_name!r} "
                "not found in status response"
            ),
        }

    status = entry.get("status")

    if not isinstance(status, dict):
        return {
            "connected": False,
            "available": False,
            "ready": None,
            "moving": None,
            "protective_stop": None,
            "emergency_stop": None,
            "fault": None,
            "program_running": None,
            "arm_mode": None,
            "safety_mode": None,
            "pose": None,
            "joints": None,
            "error": "arm status unavailable",
        }

    connected = (
        status.get("connected") is True
    )

    ready = (
        status.get("ready") is True
    )

    protective_stop = (
        status.get("protective_stop") is True
    )

    emergency_stop = (
        status.get("emergency_stop") is True
    )

    fault = (
        status.get("fault") is True
    )

    enabled = (
        device.get("enabled") is True
    )

    # Planner 是否能實際派任務。
    #
    # moving 不放進 available 判定：
    # moving/busy 應由 scheduler/runtime occupancy
    # 額外處理，而不是把設備定義成 unavailable。
    available = (
        enabled
        and connected
        and ready
        and not protective_stop
        and not emergency_stop
        and not fault
    )

    return {
        "connected": connected,
        "available": available,
        "ready": status.get("ready"),
        "moving": status.get("moving"),
        "protective_stop": status.get(
            "protective_stop"
        ),
        "emergency_stop": status.get(
            "emergency_stop"
        ),
        "fault": status.get("fault"),
        "program_running": status.get(
            "program_running"
        ),
        "arm_mode": status.get(
            "arm_mode"
        ),
        "safety_mode": status.get(
            "safety_mode"
        ),
        "pose": deepcopy(
            status.get("pose")
        ),
        "joints": deepcopy(
            status.get("joints")
        ),
    }


DEFAULT_STATUS_READERS = {
    "robot_arm": _read_robot_arm_status,
}


class DeviceRegistry:
    def __init__(
        self,
        devices: dict[str, dict[str, Any]],
    ):
        if not isinstance(devices, dict):
            raise DeviceRegistryError(
                "devices 必須是 dict"
            )

        normalized = {}

        for raw_id, raw_device in devices.items():
            if (
                not isinstance(raw_id, str)
                or not raw_id.strip()
            ):
                raise DeviceRegistryError(
                    "device id 必須是非空字串"
                )

            device_id = raw_id.strip()

            if not isinstance(raw_device, dict):
                raise DeviceRegistryError(
                    f"device {device_id} 必須是 dict"
                )

            device = deepcopy(raw_device)

            device_type = device.get("type")

            if (
                not isinstance(device_type, str)
                or not device_type.strip()
            ):
                raise DeviceRegistryError(
                    f"device {device_id} "
                    "缺少有效 type"
                )

            capabilities = device.get(
                "capabilities",
                [],
            )

            zones = device.get(
                "reachable_zones",
                [],
            )

            if not isinstance(
                capabilities,
                (list, set, tuple),
            ):
                raise DeviceRegistryError(
                    f"device {device_id}."
                    "capabilities 格式錯誤"
                )

            if not isinstance(
                zones,
                (list, set, tuple),
            ):
                raise DeviceRegistryError(
                    f"device {device_id}."
                    "reachable_zones 格式錯誤"
                )

            device.update({
                "id": device_id,
                "type": device_type.strip(),
                "enabled": (
                    device.get(
                        "enabled",
                        True,
                    )
                    is True
                ),
                "capabilities": sorted(
                    set(capabilities)
                ),
                "reachable_zones": sorted(
                    set(zones)
                ),
            })

            normalized[device_id] = device

        self._devices = normalized

    @classmethod
    def from_arm_config(
        cls,
        arm_configs: (
            dict[str, dict[str, Any]]
            | None
        ) = None,
        capability_overrides: (
            dict[
                str,
                set[str] | list[str],
            ]
            | None
        ) = None,
    ) -> "DeviceRegistry":

        arm_configs = (
            config.ARMS
            if arm_configs is None
            else arm_configs
        )

        capability_overrides = (
            capability_overrides or {}
        )

        devices = {}

        for (
            arm_name,
            arm_config,
        ) in arm_configs.items():

            device_id = (
                f"{arm_name}_arm"
            )

            capabilities = (
                capability_overrides.get(
                    device_id,
                    DEFAULT_ARM_CAPABILITIES,
                )
            )

            devices[device_id] = {
                "type": "robot_arm",

                # backend_name 是內部欄位，
                # 不會透過 for_prompt 給 LLM。
                "backend_name": arm_name,

                "driver": arm_config.get(
                    "driver"
                ),

                "enabled": arm_config.get(
                    "enabled",
                    True,
                ),

                "capabilities": capabilities,

                "reachable_zones": [],

                "end_effector": (
                    (
                        arm_config.get(
                            "tool"
                        )
                        or {}
                    ).get("type")
                ),
            }

        return cls(devices)

    def get(
        self,
        device_id: str,
    ) -> dict[str, Any] | None:

        device = self._devices.get(
            device_id
        )

        return (
            deepcopy(device)
            if device is not None
            else None
        )

    def list(
        self,
        device_type: str | None = None,
    ) -> list[dict[str, Any]]:

        devices = list(
            self._devices.values()
        )

        if device_type is not None:
            devices = [
                device
                for device in devices
                if (
                    device["type"]
                    == device_type
                )
            ]

        return deepcopy(
            sorted(
                devices,
                key=lambda item: item["id"],
            )
        )

    def for_prompt(
        self,
    ) -> list[dict[str, Any]]:
        """Return only safe static device metadata for planning."""

        return [
            {
                key: deepcopy(value)
                for key, value in device.items()
                if (
                    key
                    in PUBLIC_DEVICE_FIELDS
                    and value is not None
                )
            }
            for device in self.list()
        ]

    def probe_statuses(
        self,
        status_readers: (
            dict[str, Any]
            | None
        ) = None,
    ) -> dict[str, dict[str, Any]]:
        """Read all registered device states.

        所有 configured devices 都會出現在結果內。

        enabled=False 不代表 disconnected；
        因此 disabled device 仍會執行 read-only status
        probe，只是 available 一定為 False。

        Future IO、AGV、camera、gripper 可以透過
        device type 註冊新的 status reader。
        """

        readers = dict(
            DEFAULT_STATUS_READERS
        )

        if status_readers is not None:
            if not isinstance(
                status_readers,
                dict,
            ):
                raise DeviceRegistryError(
                    "status_readers 必須是 dict"
                )

            readers.update(
                status_readers
            )

        statuses = {}

        for device in self.list():
            device_id = device["id"]

            internal_device = self.get(
                device_id
            )

            if internal_device is None:
                statuses[device_id] = {
                    "enabled": False,
                    "connected": False,
                    "available": False,
                    "error": (
                        "device disappeared "
                        "from registry"
                    ),
                }
                continue

            enabled = (
                internal_device.get(
                    "enabled"
                )
                is True
            )

            reader = readers.get(
                internal_device["type"]
            )

            if not callable(reader):
                statuses[device_id] = {
                    "enabled": enabled,
                    "connected": False,
                    "available": False,
                    "error": (
                        "no status reader "
                        "for device type "
                        f"{internal_device['type']}"
                    ),
                }
                continue

            try:
                status = reader(
                    deepcopy(
                        internal_device
                    )
                )

                if not isinstance(
                    status,
                    dict,
                ):
                    raise DeviceRegistryError(
                        "status reader "
                        "必須回傳 dict"
                    )

                status = deepcopy(status)

                # Registry 層統一補 enabled。
                status["enabled"] = enabled

                # 無論 custom reader 怎麼判斷，
                # config disabled 都不允許 allocator 使用。
                if not enabled:
                    status["available"] = False

                # 保證 canonical 欄位存在。
                status.setdefault(
                    "connected",
                    False,
                )

                status.setdefault(
                    "available",
                    False,
                )

                statuses[device_id] = status

            except Exception as exc:
                statuses[device_id] = {
                    "enabled": enabled,
                    "connected": False,
                    "available": False,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

        return statuses