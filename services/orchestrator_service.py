"""Guarded in-process coordination for the left and right robot arms.

Imported by the existing Flask application on port 5001. Importing this module
does not connect to hardware; probes and moves only happen through explicit API
calls. Natural-language planning never receives execution authority.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from services import arm_service
from agent.ollama_client import OllamaClientError, OllamaStructuredClient
from agent.orchestrator_planner import NaturalLanguagePlanner
from orchestrator.executor import CONFIRMATION_PHRASE, TaskExecutor
from orchestrator.planner import DryRunPlanner, PLAN_TTL_SECONDS
from orchestrator.registry import RobotCellRegistry


CELL_CONFIGS = {
    "left": {"backend_arm_name": "left", "enabled": True},
    "right": {"backend_arm_name": "right", "enabled": True},
}


@dataclass(frozen=True)
class DirectProbeResult:
    cell_id: str
    backend_arm_name: str
    state: str
    available: bool
    status: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self):
        return {
            "cell_id": self.cell_id,
            "backend_arm_name": self.backend_arm_name,
            "state": self.state,
            "available": self.available,
            "status": self.status,
            "error": self.error,
        }


class DirectArmClient:
    """Adapter around the existing multi-arm service; no self-HTTP calls."""

    def probe_arm(self, cell_id, backend_arm_name):
        response = arm_service.get_arm_status(arm_name=backend_arm_name)
        if not isinstance(response, dict) or response.get("status") != "success":
            return DirectProbeResult(
                cell_id, backend_arm_name, "probe_failed", False,
                error=(response or {}).get("message", "arm status query failed"),
            )
        arms = (response.get("data") or {}).get("arms") or []
        entry = next(
            (item for item in arms if item.get("arm_name") == backend_arm_name),
            None,
        )
        if entry is None:
            return DirectProbeResult(
                cell_id, backend_arm_name, "disconnected", False,
                error="arm missing from status response",
            )
        status = entry.get("status") or {}
        available = (
            status.get("connected") is True
            and status.get("data_valid") is True
            and status.get("data_stale") is False
            and status.get("is_emergency_stopped") is not True
            and status.get("is_protective_stopped") is not True
        )
        if status.get("is_emergency_stopped") is True:
            state = "emergency_stopped"
        elif status.get("is_protective_stopped") is True:
            state = "protective_stopped"
        elif available:
            state = "available"
        elif status.get("connected") is True:
            state = "degraded"
        else:
            state = "disconnected"
        return DirectProbeResult(
            cell_id, backend_arm_name, state, available, status,
            status.get("receive_error"),
        )

    def move_arm_pose(self, backend_arm_name, pose, speed, acceleration):
        response = arm_service.move_arm_pose(
            arm_name=backend_arm_name,
            pose=pose,
            speed=speed,
            acceleration=acceleration,
            wait=True,
        )
        if (
            not isinstance(response, dict)
            or response.get("status") != "success"
            or response.get("result") is not True
        ):
            raise RuntimeError(
                (response or {}).get("message") or "arm did not confirm target reached"
            )
        return response


def _env_enabled(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


_execution_token = os.environ.get("ROBOT_ORCHESTRATOR_EXECUTION_TOKEN")
_execution_enabled = (
    _env_enabled("ROBOT_ORCHESTRATOR_EXECUTION_ENABLED")
    and bool(_execution_token)
)
_client = DirectArmClient()
_registry = RobotCellRegistry(
    CELL_CONFIGS,
    _client,
    execution_enabled=_execution_enabled,
)
_planner = DryRunPlanner(_registry)
_llm_client = OllamaStructuredClient(
    os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
    float(os.environ.get("OLLAMA_TIMEOUT", "60")),
)
_agent_planner = NaturalLanguagePlanner(_llm_client, _planner)
_executor = TaskExecutor(
    _registry,
    _client,
    _planner,
    _execution_enabled,
    _execution_token,
)


def probe_cells():
    return _registry.probe_all()


def prepare_plan(payload):
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    if "stages" in payload:
        return _planner.plan_parallel_stages(payload.get("stages"))
    return _planner.plan(payload.get("steps"))


def plan_from_text(user_text):
    return _agent_planner.plan(user_text)


def execute_plan(plan_id, confirmation, execution_token):
    return _executor.execute(plan_id, confirmation, execution_token)


def execution_policy():
    return {
        "execution_enabled": _execution_enabled,
        "execution_token_configured": bool(_execution_token),
        "confirmation_phrase": CONFIRMATION_PHRASE,
        "plan_ttl_seconds": PLAN_TTL_SECONDS,
        "speed": 0.03,
        "acceleration": 0.05,
        "cells": sorted(CELL_CONFIGS),
        "server_port": 5001,
    }
