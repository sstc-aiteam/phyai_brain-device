"""Deterministically derive task facts from planned robot actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_PLACE_DESTINATIONS = {
    "place_object_in_trash_can": "trash_can",
    "place_object_in_top_cabinet": "top_cabinet",
    "place_object_in_second_drawer": "second_drawer",
}


def _steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = plan.get("resolved_steps")
    if isinstance(rows, list):
        return rows
    rows = plan.get("steps")
    return rows if isinstance(rows, list) else []


def _function_name(step: dict[str, Any]) -> str:
    value = step.get("function_name")
    if not value and isinstance(step.get("action"), dict):
        value = step["action"].get("function_name")
    return str(value or "").strip()


def _arguments(step: dict[str, Any]) -> dict[str, Any]:
    value = step.get("arguments")
    if not isinstance(value, dict) and isinstance(step.get("action"), dict):
        value = step["action"].get("arguments")
    return value if isinstance(value, dict) else {}


def _target_binding(step: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    _ = context
    arguments = _arguments(step)
    object_id = arguments.get("object_id")
    if object_id:
        return {"status": "resolved", "object_id": str(object_id)}
    return {
        "status": "unresolved",
        "missing": "object_id",
        "reason": "step arguments/context 未提供 target object binding",
    }


def derive_perception_requirements(
    plan: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return only predicates required by the actions present in ``plan``."""
    if not isinstance(plan, dict):
        raise ValueError("plan 必須是 dict")
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise ValueError("context 必須是 dict")
    result = []
    for step in _steps(plan):
        if not isinstance(step, dict):
            continue
        function_name = _function_name(step)
        destination = _PLACE_DESTINATIONS.get(function_name)
        if destination is None:
            continue
        binding = _target_binding(step, context)
        target_subject = str(binding.get("object_id") or "target_object")
        target_fields = {"binding": binding} if binding["status"] == "unresolved" else {}
        requirements = [
            {"subject": target_subject, "predicate": "exists", "timing": "before_step", "freshness": "current", "camera_source": "left", **target_fields},
            {"subject": target_subject, "predicate": "position", "timing": "before_step", "freshness": "current", "camera_source": "left", **target_fields},
            {"subject": target_subject, "predicate": "reachable", "timing": "before_step", "freshness": "current", "camera_source": "left", **target_fields},
            {"subject": destination, "predicate": "exists", "timing": "before_step", "freshness": "current"},
            {"subject": destination, "predicate": "open_state", "timing": "before_step", "freshness": "current", "camera_source": "middle"},
        ]
        result.append({"step_id": str(step.get("id") or ""), "requirements": requirements})
    return deepcopy(result)
