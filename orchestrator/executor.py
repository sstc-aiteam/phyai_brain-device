"""Guarded sequential execution of previously reviewed short-lived plans."""

from __future__ import annotations

import math
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from threading import Lock
from typing import Any


CONFIRMATION_PHRASE = "EXECUTE_CLEARED_TWO_ARM_PLAN"
MOVE_SPEED = 0.03
MOVE_ACCELERATION = 0.05
POSITION_TOLERANCE_METERS = 0.01
ROTATION_TOLERANCE_RADIANS = 0.05


class TaskExecutor:
    def __init__(
        self,
        registry,
        backend_client,
        planner,
        enabled: bool = False,
        execution_token: str | None = None,
    ):
        self._registry = registry
        self._backend_client = backend_client
        self._planner = planner
        self._enabled = bool(enabled)
        self._execution_token = execution_token
        self._execution_lock = Lock()

    def execute(
        self,
        plan_id: Any,
        confirmation: Any,
        execution_token: Any,
    ) -> dict[str, Any]:
        if not self._enabled:
            raise PermissionError("hardware execution is disabled by configuration")
        if confirmation != CONFIRMATION_PHRASE:
            raise PermissionError("invalid execution confirmation phrase")
        if (
            not isinstance(execution_token, str)
            or not secrets.compare_digest(execution_token, self._execution_token or "")
        ):
            raise PermissionError("invalid execution token")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("another coordinated task is already executing")

        try:
            plan = self._planner.claim(plan_id)
            completed = []
            if plan.get("execution_mode") == "parallel_stages":
                self._execute_parallel_stages(plan["stages"], completed)
            else:
                for step in plan["steps"]:
                    completed.append(self._execute_sequential_step(step))

            return {
                "mode": "executed",
                "plan_id": plan_id,
                "completed_count": len(completed),
                "completed_steps": completed,
                "execution_mode": plan.get("execution_mode", "sequential"),
                "speed": MOVE_SPEED,
                "acceleration": MOVE_ACCELERATION,
            }
        except Exception as exc:
            completed_count = len(locals().get("completed", []))
            raise RuntimeError(
                f"task aborted after {completed_count} completed step(s): {exc}"
            ) from exc
        finally:
            self._execution_lock.release()

    def _execute_sequential_step(self, step: dict[str, Any]) -> dict[str, Any]:
        cell_id = step["cell_id"]
        before = self._registry.probe_one(cell_id)
        self._require_safe_and_near(before, step["source_pose"], "before move")
        response = self._move_step(step)
        after = self._registry.probe_one(cell_id)
        self._require_safe_and_near(after, step["target_pose"], "after move")
        return self._completed_step(step, after, response)

    def _execute_parallel_stages(
        self,
        stages: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> None:
        for stage in stages:
            steps = stage["parallel"]

            # Barrier 1: every arm must be safe and still at the reviewed source.
            for step in steps:
                before = self._registry.probe_one(step["cell_id"])
                self._require_safe_and_near(
                    before,
                    step["source_pose"],
                    f"stage {stage['stage']} before move",
                )

            responses = {}
            with ThreadPoolExecutor(max_workers=len(steps)) as pool:
                futures = {pool.submit(self._move_step, step): step for step in steps}
                for future in as_completed(futures):
                    step = futures[future]
                    responses[step["sequence"]] = future.result()

            # Barrier 2: no later stage starts until every target is verified.
            stage_completed = []
            for step in steps:
                after = self._registry.probe_one(step["cell_id"])
                self._require_safe_and_near(
                    after,
                    step["target_pose"],
                    f"stage {stage['stage']} after move",
                )
                stage_completed.append(
                    self._completed_step(step, after, responses[step["sequence"]])
                )
            completed.extend(stage_completed)

    def _move_step(self, step: dict[str, Any]) -> dict[str, Any]:
        return self._backend_client.move_arm_pose(
            self._registry.backend_arm_name(step["cell_id"]),
            deepcopy(step["target_pose"]),
            MOVE_SPEED,
            MOVE_ACCELERATION,
        )

    @staticmethod
    def _completed_step(step, after, response):
        return {
            "sequence": step["sequence"],
            "cell_id": step["cell_id"],
            "target_pose": deepcopy(step["target_pose"]),
            "verified_pose": deepcopy(after["status"]["pose"]),
            "backend_response": response,
        }

    @staticmethod
    def _require_safe_and_near(result: dict[str, Any], expected: list[float], stage: str):
        if result.get("state") != "available" or result.get("available") is not True:
            raise RuntimeError(f"{stage}: arm is not available")
        status = result.get("status") or {}
        pose = status.get("pose")
        if not isinstance(pose, list) or len(pose) != 6:
            raise RuntimeError(f"{stage}: valid pose unavailable")
        if any(not math.isfinite(float(value)) for value in pose):
            raise RuntimeError(f"{stage}: pose contains a non-finite value")
        position_error = max(abs(float(pose[i]) - expected[i]) for i in range(3))
        rotation_error = max(abs(float(pose[i]) - expected[i]) for i in range(3, 6))
        if position_error > POSITION_TOLERANCE_METERS:
            raise RuntimeError(
                f"{stage}: position differs from plan by {position_error:.6f} m"
            )
        if rotation_error > ROTATION_TOLERANCE_RADIANS:
            raise RuntimeError(
                f"{stage}: rotation differs from plan by {rotation_error:.6f} rad"
            )
