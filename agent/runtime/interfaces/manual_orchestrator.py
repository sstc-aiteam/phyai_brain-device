"""
Manual Runtime construction and orchestration helpers.

This layer assembles the closed-loop Runtime used by the manual UI.
HTTP/service state remains outside this module.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path

from ..adapters.memory import ExecutionMemory
from ..adapters.runtime_observation import (
    build_runtime_dispatch_observer,
)
from ..decision.interpreter import (
    interpret_runtime_task,
    normalize_runtime_planning_options,
)
from ..decision.planner import RuntimeBrain
from ..domain.task import (
    TaskSpec,
    task_spec_to_dict,
)
from ..domain.world import WorldState
from ..engine.guards.binding import (
    StaticWorldDeviceBinder,
    apply_force_device_constraint,
)
from ..engine.runtime import RuntimeEngine
from ..tools.catalog import runtime_tools

from .manual_executor import ManualUIRuntimeExecutor
from .manual_session import _ManualRuntimeSession


@dataclass(frozen=True)
class ManualRuntimeSetup:
    task_spec: TaskSpec
    engine: RuntimeEngine


def prepare_manual_runtime(
    *,
    session: _ManualRuntimeSession,
    world: WorldState,
    runtime_data_dir: Path,
) -> ManualRuntimeSetup:

    task_spec = interpret_runtime_task(
        session.user_text,
        world=
            world,
        grounded_targets=
            None,
        planning_options=
            session.planning_options,
        openai_api_key=
            session.openai_api_key,
    )

    if task_spec.goals is None:
        raise RuntimeError(
            "Task Interpreter returned no goals"
        )

    tool_catalog = runtime_tools()

    task_spec, forced_functions = (
        apply_force_device_constraint(
            task_spec,
            world=
                world,
            tools=
                tool_catalog,
            force_device_id=
                session.force_device_id,
        )
    )

    if forced_functions:
        session.record(
            "force_device_applied",
            message=(
                "Manual Runtime hard device constraint applied: "
                f"{session.force_device_id}"
            ),
            data={
                "device_id":
                    session.force_device_id,
                "mode":
                    "required",
                "functions":
                    forced_functions,
            },
        )

    task_spec_dict = task_spec_to_dict(
        task_spec
    )

    session.update(
        task_spec=
            task_spec_dict,
        stage=
            "runtime_setup",
        message=
            "TaskSpec 完成，正在建立 Runtime Coordinator。",
    )
    session.record(
        "task_interpreted",
        data={
            "task_spec":
                task_spec_dict,
        },
    )

    normalized = normalize_runtime_planning_options(
        session.planning_options
    )
    brain_options = (
        normalized.get(
            "step_planning"
        )
        or {}
    )

    # Runtime Step Planning is an explicit user-facing phase.
    # Request configuration is authoritative so the UI cannot be
    # silently overridden by stale environment variables.
    brain_provider = str(
        brain_options.get(
            "provider",
            "local",
        )
        or "local"
    ).strip().lower()
    brain_model = brain_options.get(
        "model"
    )

    brain = RuntimeBrain(
        provider=
            brain_provider,
        model=
            brain_model,
        api_key=
            session.openai_api_key,
        temperature=
            0.1,
        max_tokens=
            1024,
        timeout=
            120,
    )

    session.record(
        "runtime_brain_configured",
        data={
            "phase":
                "step_planning",
            "provider":
                brain_provider,
            "model":
                brain_model,
            "api_key_source": (
                (
                    "request"
                    if session.openai_api_key
                    else (
                        "environment"
                        if os.environ.get(
                            "OPENAI_API_KEY"
                        )
                        else None
                    )
                )
                if brain_provider == "openai"
                else None
            ),
        },
    )

    executor = ManualUIRuntimeExecutor(
        session=
            session,
        observe_dispatch=
            build_runtime_dispatch_observer(),
    )

    execution_memory_path = (
        runtime_data_dir
        / session.session_id
        / "execution_memory.jsonl"
    )
    execution_memory = ExecutionMemory(
        execution_memory_path,
        max_entries=10,
    )
    session.update(
        execution_memory_path=
            str(execution_memory_path),
    )
    session.record(
        "execution_memory_ready",
        data={
            "path":
                str(execution_memory_path),
            "max_entries":
                10,
        },
    )

    engine = RuntimeEngine(
        brain=
            brain,
        binder=
            StaticWorldDeviceBinder(),
        executor=
            executor,
        tools=
            tool_catalog,
        execution_memory=
            execution_memory,
        max_parallel_actions=
            1,
        max_decision_attempts=
            5,
    )

    return ManualRuntimeSetup(
        task_spec=task_spec,
        engine=engine,
    )


__all__ = [
    "ManualRuntimeSetup",
    "prepare_manual_runtime",
]
