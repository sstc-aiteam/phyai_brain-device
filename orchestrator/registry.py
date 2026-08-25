"""Runtime registry that probes configured robot cells on demand."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any

BASE_CAPABILITIES = {
    "read_status": True,
    "move": False,
    "gripper": False,
    "camera": False,
}


class RobotCellRegistry:
    def __init__(
        self,
        cell_configs: dict[str, dict[str, Any]],
        client,
        execution_enabled: bool = False,
    ):
        self._cell_configs = deepcopy(cell_configs)
        for config in self._cell_configs.values():
            resources = config.setdefault("resources", {})
            resources.setdefault("arm", {
                "backend_name": config["backend_arm_name"],
                "verification": "confirmed",
            })
            resources.setdefault("gripper", {
                "backend_name": None,
                "verification": "unknown",
            })
            resources.setdefault("wrist_camera", {
                "backend_name": None,
                "verification": "unknown",
            })
        self._client = client
        self._capabilities = deepcopy(BASE_CAPABILITIES)
        self._capabilities["move"] = bool(execution_enabled)
        self._lock = RLock()
        self._snapshot: dict[str, Any] | None = None
        self._runtime = {
            cell_id: {
                "last_probe": None,
                "last_success_at": None,
                "consecutive_failures": 0,
            }
            for cell_id in self._cell_configs
        }

    def configured_cells(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._cell_view(cell_id) for cell_id in sorted(self._cell_configs)]

    def configured_cell(self, cell_id: str) -> dict[str, Any] | None:
        with self._lock:
            if cell_id not in self._cell_configs:
                return None
            return self._cell_view(cell_id)

    def _cell_view(self, cell_id: str) -> dict[str, Any]:
        return {
            "cell_id": cell_id,
            **deepcopy(self._cell_configs[cell_id]),
            "capabilities": deepcopy(self._capabilities),
            **deepcopy(self._runtime[cell_id]),
        }

    def last_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(self._snapshot)

    def safe_planning_observation(self, cell_id: str) -> dict[str, Any] | None:
        """Return the last safe arm observation without contacting the backend."""
        with self._lock:
            if cell_id not in self._cell_configs:
                return None
            last_probe = self._runtime[cell_id]["last_probe"]
            if not last_probe or last_probe.get("state") != "available":
                raise ValueError(
                    f"cell '{cell_id}' requires a successful available probe before planning"
                )
            status = last_probe.get("status") or {}
            if (
                status.get("connected") is not True
                or status.get("data_valid") is not True
                or status.get("data_stale") is not False
                or status.get("is_emergency_stopped") is True
                or status.get("is_protective_stopped") is True
            ):
                raise ValueError(f"cell '{cell_id}' last probe is not safe for planning")
            return deepcopy(status)

    def backend_arm_name(self, cell_id: str) -> str | None:
        config = self._cell_configs.get(cell_id)
        return None if config is None else config["backend_arm_name"]

    def probe_all(self) -> dict[str, Any]:
        candidates = [
            (cell_id, config)
            for cell_id, config in self._cell_configs.items()
            if config.get("enabled", True)
        ]
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max(1, min(8, len(candidates)))) as pool:
            futures = {
                pool.submit(
                    self._client.probe_arm,
                    cell_id,
                    config["backend_arm_name"],
                ): cell_id
                for cell_id, config in candidates
            }
            for future in as_completed(futures):
                cell_id = futures[future]
                try:
                    result = future.result().to_dict()
                except Exception as exc:
                    result = self._unexpected_failure(cell_id, exc)
                results.append(self._record_result(result))

        results.sort(key=lambda item: item["cell_id"])
        snapshot = {
            "probed_at": datetime.now(timezone.utc).isoformat(),
            "configured_count": len(self._cell_configs),
            "probed_count": len(results),
            "connected_count": sum(
                item["status"] is not None
                and item["status"].get("connected") is True
                for item in results
            ),
            "available_count": sum(item["available"] for item in results),
            "cells": results,
        }

        with self._lock:
            self._snapshot = deepcopy(snapshot)
        return snapshot

    def probe_one(self, cell_id: str) -> dict[str, Any] | None:
        config = self._cell_configs.get(cell_id)
        if config is None:
            return None
        if not config.get("enabled", True):
            result = {
                "cell_id": cell_id,
                "backend_arm_name": config["backend_arm_name"],
                "state": "disabled",
                "available": False,
                "status": None,
                "error": "cell is disabled",
            }
            return self._record_result(result)

        try:
            result = self._client.probe_arm(
                cell_id,
                config["backend_arm_name"],
            ).to_dict()
        except Exception as exc:
            result = self._unexpected_failure(cell_id, exc)
        return self._record_result(result)

    def _unexpected_failure(self, cell_id: str, exc: Exception) -> dict[str, Any]:
        return {
            "cell_id": cell_id,
            "backend_arm_name": self._cell_configs[cell_id]["backend_arm_name"],
            "state": "probe_failed",
            "available": False,
            "status": None,
            "error": str(exc),
        }

    def _record_result(self, result: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        cell_id = result["cell_id"]
        succeeded = result.get("status") is not None
        with self._lock:
            runtime = self._runtime[cell_id]
            if succeeded:
                runtime["last_success_at"] = now
                runtime["consecutive_failures"] = 0
            else:
                runtime["consecutive_failures"] += 1
            runtime["last_probe"] = {"probed_at": now, **deepcopy(result)}
            tracking = {
                "probed_at": now,
                "last_success_at": runtime["last_success_at"],
                "consecutive_failures": runtime["consecutive_failures"],
            }
        return {**result, **tracking}
