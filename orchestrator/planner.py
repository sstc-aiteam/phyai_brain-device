"""Pure, read-only planning for coordinated robot-cell demonstrations."""

from __future__ import annotations

import math
import secrets
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any


MAX_RELATIVE_Z_METERS = 0.05
PLAN_TTL_SECONDS = 120


class DryRunPlanner:
    def __init__(self, registry):
        self._registry = registry
        self._lock = RLock()
        self._plans: dict[str, dict[str, Any]] = {}

    def plan(self, steps: Any) -> dict[str, Any]:
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps must be a non-empty list")

        starts: dict[str, list[float]] = {}
        simulated: dict[str, list[float]] = {}
        planned_steps = []

        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"steps[{index}] must be an object")

            cell_id = step.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id.strip():
                raise ValueError(f"steps[{index}].cell_id must be a non-empty string")
            cell_id = cell_id.strip().lower()

            if cell_id not in starts:
                observation = self._registry.safe_planning_observation(cell_id)
                if observation is None:
                    raise ValueError(f"unknown robot cell: {cell_id}")
                pose = observation.get("pose")
                if not isinstance(pose, list) or len(pose) != 6:
                    raise ValueError(f"cell '{cell_id}' has no valid probed pose")
                starts[cell_id] = [float(value) for value in pose]
                simulated[cell_id] = deepcopy(starts[cell_id])

            operation_keys = {
                key for key in ("relative_z", "return_to_start") if key in step
            }
            if len(operation_keys) != 1:
                raise ValueError(
                    f"steps[{index}] must contain exactly one operation: "
                    "relative_z or return_to_start"
                )
            allowed_keys = {"cell_id", *operation_keys}
            if set(step) != allowed_keys:
                raise ValueError(f"steps[{index}] contains unsupported fields")

            source_pose = deepcopy(simulated[cell_id])
            if "relative_z" in step:
                try:
                    relative_z = float(step["relative_z"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"steps[{index}].relative_z must be numeric") from exc
                if not math.isfinite(relative_z) or relative_z == 0:
                    raise ValueError(f"steps[{index}].relative_z must be finite and non-zero")
                if abs(relative_z) > MAX_RELATIVE_Z_METERS:
                    raise ValueError(
                        f"steps[{index}].relative_z exceeds "
                        f"{MAX_RELATIVE_Z_METERS:.2f} m dry-run limit"
                    )
                target_pose = deepcopy(source_pose)
                target_pose[2] += relative_z
                operation = {"type": "relative_z", "meters": relative_z}
            else:
                if step["return_to_start"] is not True:
                    raise ValueError(f"steps[{index}].return_to_start must be true")
                target_pose = deepcopy(starts[cell_id])
                operation = {"type": "return_to_start"}

            simulated[cell_id] = target_pose
            planned_steps.append({
                "sequence": index + 1,
                "cell_id": cell_id,
                "operation": operation,
                "source_pose": source_pose,
                "target_pose": deepcopy(target_pose),
            })

        plan = {
            "mode": "dry_run",
            "will_execute": False,
            "backend_calls_made": 0,
            "limits": {"max_relative_z_meters": MAX_RELATIVE_Z_METERS},
            "start_poses": starts,
            "steps": planned_steps,
        }
        now = datetime.now(timezone.utc)
        plan_id = secrets.token_urlsafe(18)
        plan["plan_id"] = plan_id
        plan["expires_at"] = (now + timedelta(seconds=PLAN_TTL_SECONDS)).isoformat()
        with self._lock:
            self._plans[plan_id] = {
                "expires_at": now + timedelta(seconds=PLAN_TTL_SECONDS),
                "consumed": False,
                "plan": deepcopy(plan),
            }
        return plan

    def plan_parallel_stages(self, stages: Any) -> dict[str, Any]:
        if not isinstance(stages, list) or not stages:
            raise ValueError("stages must be a non-empty list")

        flattened = []
        stage_sizes = []
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict) or not isinstance(stage.get("parallel"), list):
                raise ValueError(f"stages[{index}].parallel must be a non-empty list")
            if set(stage) != {"parallel"}:
                raise ValueError(f"stages[{index}] contains unsupported fields")
            parallel = stage["parallel"]
            if not parallel:
                raise ValueError(f"stages[{index}].parallel must be a non-empty list")
            cell_ids = [step.get("cell_id") for step in parallel if isinstance(step, dict)]
            if len(cell_ids) != len(set(cell_ids)):
                raise ValueError(f"stages[{index}] contains duplicate cell_id values")
            flattened.extend(parallel)
            stage_sizes.append(len(parallel))

        plan = self.plan(flattened)
        planned_stages = []
        offset = 0
        for index, size in enumerate(stage_sizes):
            planned_stages.append({
                "stage": index + 1,
                "parallel": deepcopy(plan["steps"][offset:offset + size]),
            })
            offset += size
        plan["execution_mode"] = "parallel_stages"
        plan["stages"] = planned_stages
        with self._lock:
            self._plans[plan["plan_id"]]["plan"] = deepcopy(plan)
        return plan

    def claim(self, plan_id: Any) -> dict[str, Any]:
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan_id must be a non-empty string")
        with self._lock:
            stored = self._plans.get(plan_id)
            if stored is None:
                raise ValueError("unknown plan_id")
            if stored["consumed"]:
                raise ValueError("plan_id has already been consumed")
            if datetime.now(timezone.utc) > stored["expires_at"]:
                raise ValueError("plan_id has expired; create a new dry-run")
            stored["consumed"] = True
            return deepcopy(stored["plan"])
