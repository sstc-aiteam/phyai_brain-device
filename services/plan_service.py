from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import RLock, Thread
from typing import Any
from uuid import uuid4

from agent.runtime.adapters.runtime_observation import (
    build_runtime_initial_world,
)
from agent.runtime.decision.interpreter import (
    RUNTIME_ALLOWED_PROVIDERS as _RUNTIME_ALLOWED_PROVIDERS,
    normalize_runtime_planning_options,
)
from agent.runtime.interfaces.manual_orchestrator import (
    prepare_manual_runtime,
)
from agent.runtime.interfaces.manual_session import (
    ManualRuntimeCancelled,
    _ManualRuntimeSession,
    _manual_short_error,
    _manual_world_summary,
)
from utils import response


MODULE = "plan"
_MANUAL_RUNTIME_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "runtime_data"
    / "manual"
)

# ============================================================
# Manual UI Runtime
# ============================================================


_MANUAL_RUNTIME_LOCK = RLock()
_MANUAL_RUNTIME_SESSION: _ManualRuntimeSession | None = None


def _manual_runtime_worker(
    session: _ManualRuntimeSession,
) -> None:
    try:
        session.update(
            status=
                "running",
            stage=
                "bootstrap",
            message=
                "正在建立 initial multi-region WorldState。",
        )
        session.record(
            "bootstrap_started"
        )

        world = build_runtime_initial_world()

        session.update(
            latest_world=
                world.snapshot(),
            stage=
                "task_interpretation",
            message=
                "Initial WorldState 完成，正在解析 Runtime task。",
        )
        session.record(
            "bootstrap_completed",
            data={
                "world":
                    _manual_world_summary(
                        world
                    ),
            },
        )

        setup = prepare_manual_runtime(
            session=session,
            world=world,
            runtime_data_dir=
                _MANUAL_RUNTIME_DATA_DIR,
        )

        task_spec = setup.task_spec
        coordinator = setup.engine

        session.update(
            stage=
                "runtime",
            message=
                "Manual closed-loop Runtime 已開始。",
        )
        session.record(
            "runtime_started"
        )

        run_result = coordinator.run_until_done(
            world=
                world,
            goals=
                task_spec.goals,
            coordination=
                task_spec.coordination,
            max_iterations=
                session.max_iterations,
        )

        result = {
            "success":
                run_result.success,
            "iterations":
                run_result.iterations,
            "final_world":
                run_result.world.snapshot(),
            "trace": [
                {
                    "iteration":
                        row.iteration,
                    "dispatch":
                        deepcopy(
                            row.dispatch
                        ),
                    "bound_dispatch":
                        deepcopy(
                            row.bound_dispatch
                        ),
                    "execution":
                        deepcopy(
                            row.execution
                        ),
                    "progress_feedback":
                        deepcopy(
                            row.progress_feedback
                        ),
                }
                for row
                in run_result.trace
            ],
        }

        session.update(
            status=(
                "complete"
                if run_result.success
                else "stopped"
            ),
            stage=(
                "complete"
                if run_result.success
                else "max_iterations"
            ),
            message=(
                "Runtime task 完成。"
                if run_result.success
                else "已到 max_iterations，Runtime 尚未完成 task。"
            ),
            instruction=
                None,
            pending_action=
                None,
            latest_world=
                run_result.world.snapshot(),
            result=
                result,
        )
        session.record(
            "runtime_finished",
            data={
                "success":
                    run_result.success,
                "iterations":
                    run_result.iterations,
            },
        )

    except ManualRuntimeCancelled as exc:
        session.update(
            status=
                "cancelled",
            stage=
                "cancelled",
            message=
                str(
                    exc
                ),
            instruction=
                None,
            pending_action=
                None,
        )
        session.record(
            "runtime_cancelled",
            message=
                str(
                    exc
                ),
        )

    except Exception as exc:
        full_message = str(
            exc
        )
        summary = _manual_short_error(
            exc
        )

        session.update(
            status=
                "failed",
            stage=
                "failed",
            message=
                summary,
            instruction=
                None,
            pending_action=
                None,
            error={
                "type":
                    type(
                        exc
                    ).__name__,
                "summary":
                    summary,
                "message":
                    full_message,
            },
        )
        session.record(
            "runtime_failed",
            message=
                summary,
            data={
                "error_type":
                    type(
                        exc
                    ).__name__,
                "full_message":
                    full_message,
                "latest_world":
                    _manual_world_summary(
                        session.latest_world
                    ),
            },
        )


def start_manual_runtime(
    user_text: str,
    *,
    planning_options: dict[str, Any] | None = None,
    openai_api_key: str | None = None,
    max_iterations: int = 10,
    force_device_id: str | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(
            user_text,
            str,
        )
        or not user_text.strip()
    ):
        raise ValueError(
            "user_text 必須是非空字串"
        )

    if (
        not isinstance(
            max_iterations,
            int,
        )
        or max_iterations < 1
    ):
        raise ValueError(
            "max_iterations 必須是 >= 1 的整數"
        )

    normalized = normalize_runtime_planning_options(
        planning_options
    )

    if openai_api_key is not None:
        if (
            not isinstance(
                openai_api_key,
                str,
            )
            or not openai_api_key.strip()
        ):
            raise ValueError(
                "openai_api_key 必須是非空字串或 null"
            )
        openai_api_key = openai_api_key.strip()

    if force_device_id is not None:
        force_device_id = str(
            force_device_id
        ).strip() or None

    global _MANUAL_RUNTIME_SESSION

    with _MANUAL_RUNTIME_LOCK:
        current = _MANUAL_RUNTIME_SESSION

        if (
            current is not None
            and current.status
            in {
                "starting",
                "running",
                "awaiting_confirmation",
            }
        ):
            raise RuntimeError(
                "已有 manual Runtime session 正在執行"
            )

        session = _ManualRuntimeSession(
            session_id=
                uuid4().hex,
            user_text=
                user_text.strip(),
            planning_options=
                normalized,
            max_iterations=
                max_iterations,
            openai_api_key=
                openai_api_key,
            force_device_id=
                force_device_id,
        )
        session.record(
            "session_created",
            data={
                "planning_options":
                    normalized,
                "openai_api_key_supplied":
                    bool(openai_api_key),
                "max_iterations":
                    max_iterations,
                "force_device_id":
                    force_device_id,
            },
        )

        thread = Thread(
            target=
                _manual_runtime_worker,
            args=(
                session,
            ),
            daemon=
                True,
            name=(
                "manual-runtime-"
                + session.session_id[
                    :8
                ]
            ),
        )

        session.thread = thread
        _MANUAL_RUNTIME_SESSION = session
        thread.start()

    return session.snapshot()


def get_manual_runtime_status(
    session_id: str | None = None,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if session is None:
        raise RuntimeError(
            "目前沒有 manual Runtime session"
        )

    if (
        isinstance(
            session_id,
            str,
        )
        and session_id
        and session_id
        != session.session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    return session.snapshot()


def confirm_manual_runtime_step(
    session_id: str,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if (
        session is None
        or session.session_id
        != session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    with session.lock:
        if session.status != "awaiting_confirmation":
            raise RuntimeError(
                "manual Runtime 目前不在等待確認"
            )

        step_number = session.pending_step
        session.status = "running"
        session.stage = "manual_action_confirmed"
        session.message = "已收到人工確認，Runtime 繼續。"

    session.record(
        "confirm_api_received",
        data={
            "step":
                step_number,
        },
    )
    session.continue_event.set()

    return session.snapshot()


def cancel_manual_runtime(
    session_id: str,
) -> dict[str, Any]:
    with _MANUAL_RUNTIME_LOCK:
        session = _MANUAL_RUNTIME_SESSION

    if (
        session is None
        or session.session_id
        != session_id
    ):
        raise KeyError(
            "manual Runtime session_id 不存在"
        )

    with session.lock:
        if session.status in {
            "complete",
            "stopped",
            "failed",
            "cancelled",
        }:
            return session.snapshot()

        session.cancel_requested = True
        session.message = "正在取消 manual Runtime。"

    session.record(
        "cancel_requested"
    )
    session.continue_event.set()

    return session.snapshot()


def get_health() -> dict[str, Any]:
    return response.success(
        MODULE,
        "health",
        data={
            "service":
                "plan",
            "runtime_task_interpreter":
                True,
            "providers":
                sorted(
                    _RUNTIME_ALLOWED_PROVIDERS
                ),
            "executes_robot":
                True,
            "manual_runtime":
                True,
            "physical_manipulation":
                False,
            "task_interpreter_skill":
                "agent/skills/runtime_task_interpreter.md",
        },
    )
