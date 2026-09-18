from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any

from agent.planning.one_shot_planner import build_one_shot_plan
from agent.runtime.task_interpreter import (
    TaskSpec,
    interpret_task,
)
from agent.runtime.world_state import WorldState
from utils import response


MODULE = "plan"

_ALLOWED_PROVIDERS = {
    "local",
    "openai",
}


# ============================================================
# Planning options
# ============================================================

def _normalize_stage_options(
    value: Any,
    *,
    default_provider: str,
    default_model: str | None,
    field_name: str,
) -> dict[str, Any]:
    if value is None:
        return {
            "provider":
                default_provider,

            "model":
                default_model,
        }

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{field_name} 必須是 object"
        )

    unknown = set(
        value
    ) - {
        "provider",
        "model",
    }

    if unknown:
        raise ValueError(
            f"{field_name} 包含未知欄位："
            f"{sorted(unknown)}"
        )

    provider = value.get(
        "provider",
        default_provider,
    )

    model = value.get(
        "model",
        default_model,
    )

    if provider not in _ALLOWED_PROVIDERS:
        raise ValueError(
            f"{field_name}.provider "
            "只支援 local/openai"
        )

    if model is not None:
        if (
            not isinstance(
                model,
                str,
            )
            or not model.strip()
        ):
            raise ValueError(
                f"{field_name}.model 必須是非空字串"
            )

        model = (
            model.strip()
        )

    return {
        "provider":
            provider,

        "model":
            model,
    }


def normalize_planning_options(
    options: Any,
) -> dict[str, Any]:
    """
    Accept both forms:

    Simple:
        {
            "provider": "local",
            "model": "..."
        }

    Per-stage:
        {
            "stage1": {
                "provider": "local",
                "model": "..."
            },
            "stage2": {
                "provider": "openai",
                "model": "..."
            },
            "openai_api_key": "..."
        }

    Simple provider/model are applied to both stages.
    """

    if options is None:
        return {
            "stage1": {
                "provider":
                    "local",

                "model":
                    None,
            },

            "stage2": {
                "provider":
                    "local",

                "model":
                    None,
            },
        }

    if not isinstance(
        options,
        dict,
    ):
        raise ValueError(
            "planning_options 必須是 object"
        )

    allowed = {
        "provider",
        "model",
        "stage1",
        "stage2",
        "openai_api_key",
    }

    unknown = set(
        options
    ) - allowed

    if unknown:
        raise ValueError(
            "planning_options 包含未知欄位："
            f"{sorted(unknown)}"
        )

    has_simple = (
        "provider" in options
        or "model" in options
    )

    has_stages = (
        "stage1" in options
        or "stage2" in options
    )

    if (
        has_simple
        and has_stages
    ):
        raise ValueError(
            "planning_options 不可同時使用 "
            "provider/model 與 stage1/stage2"
        )

    if has_simple:
        provider = options.get(
            "provider",
            "local",
        )

        model = options.get(
            "model"
        )

        if provider not in _ALLOWED_PROVIDERS:
            raise ValueError(
                "planning_options.provider "
                "只支援 local/openai"
            )

        if model is not None:
            if (
                not isinstance(
                    model,
                    str,
                )
                or not model.strip()
            ):
                raise ValueError(
                    "planning_options.model 必須是非空字串"
                )

            model = (
                model.strip()
            )

        stage1 = {
            "provider":
                provider,

            "model":
                model,
        }

        stage2 = deepcopy(
            stage1
        )

    else:
        stage1 = (
            _normalize_stage_options(
                options.get(
                    "stage1"
                ),
                default_provider=
                    "local",
                default_model=
                    None,
                field_name=
                    "planning_options.stage1",
            )
        )

        stage2 = (
            _normalize_stage_options(
                options.get(
                    "stage2"
                ),
                default_provider=
                    "local",
                default_model=
                    None,
                field_name=
                    "planning_options.stage2",
            )
        )

    normalized: dict[str, Any] = {
        "stage1":
            stage1,

        "stage2":
            stage2,
    }

    key = options.get(
        "openai_api_key"
    )

    if key is not None:
        if (
            not isinstance(
                key,
                str,
            )
            or not key.strip()
        ):
            raise ValueError(
                "planning_options.openai_api_key "
                "必須是非空字串"
            )

        normalized[
            "openai_api_key"
        ] = key.strip()

    return normalized


# ============================================================
# Current read-only one-shot planner
# ============================================================

def build_one_shot_value(
    user_text: str,
    *,
    context: dict[str, Any] | None = None,
    planning_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the current resolved one-shot plan.

    This is read-only:
    no physical tool is executed.
    """

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

    if context is None:
        context = {}

    if not isinstance(
        context,
        dict,
    ):
        raise ValueError(
            "context 必須是 dict"
        )

    options = (
        normalize_planning_options(
            planning_options
        )
    )

    result = (
        build_one_shot_plan(
            user_text.strip(),
            context=
                deepcopy(
                    context
                ),
            llm_options=
                options,
        )
    )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "one-shot planner 回傳格式錯誤"
        )

    return result


def plan_command(
    user_text: str,
    *,
    context: dict[str, Any] | None = None,
    planning_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Public read-only planning service response.

    Suitable for a future direct /api/plan endpoint.
    """

    action = (
        "plan_command"
    )

    started = (
        perf_counter()
    )

    try:
        result = (
            build_one_shot_value(
                user_text,
                context=
                    context,
                planning_options=
                    planning_options,
            )
        )

        timings = result.setdefault(
            "timings_ms",
            {},
        )

        timings[
            "plan_service_total"
        ] = round(
            (
                perf_counter()
                - started
            )
            * 1000.0,
            2,
        )

        return response.success(
            MODULE,
            action,
            data=
                result,
        )

    except Exception as exc:
        return response.error(
            MODULE,
            action,
            error=
                exc,
            error_type=
                type(
                    exc
                ).__name__,
            timings_ms={
                "plan_service_total":
                    round(
                        (
                            perf_counter()
                            - started
                        )
                        * 1000.0,
                        2,
                    ),
            },
        )


# ============================================================
# New runtime task interpretation
# ============================================================

def interpret_runtime_task(
    user_text: str,
    *,
    world: WorldState,
    grounded_targets: Any = None,
    planning_options: dict[str, Any] | None = None,
) -> TaskSpec:
    """
    Human task -> Goals / CoordinationSpec.

    This is the semantic entry point for the new closed-loop runtime.
    It does not generate or execute a future action sequence.
    """

    options = (
        normalize_planning_options(
            planning_options
        )
    )

    return interpret_task(
        user_text,
        world=
            world,
        grounded_targets=
            grounded_targets,
        llm_options=
            options,
    )


def task_spec_to_dict(
    spec: TaskSpec,
) -> dict[str, Any]:
    """Serialize TaskSpec for service/API debugging."""

    goals = []

    if spec.goals is not None:
        for goal in spec.goals.conditions:
            goals.append({
                "goal_id":
                    goal.goal_id,

                "subject":
                    goal.subject,

                "field":
                    goal.field,

                "operator":
                    goal.operator,

                "value":
                    deepcopy(
                        goal.value
                    ),

                "depends_on":
                    list(
                        goal.depends_on
                    ),
            })

    coordination = {
        "maintain_until": [
            {
                "subject":
                    item.subject,

                "field":
                    item.field,

                "value":
                    deepcopy(
                        item.value
                    ),

                "until_goal":
                    item.until_goal,
            }
            for item
            in spec.coordination.maintain_until
        ],

        "device_rules": [
            {
                "function_name":
                    item.function_name,

                "subject_arg":
                    item.subject_arg,

                "subject_id":
                    item.subject_id,

                "device_id":
                    item.device_id,

                "mode":
                    item.mode,
            }
            for item
            in spec.coordination.device_rules
        ],
    }

    return {
        "summary":
            spec.summary,

        "goals":
            goals,

        "coordination":
            coordination,

        "motion_request":
            deepcopy(
                spec.motion_request
            ),

        "raw":
            deepcopy(
                spec.raw
            ),
    }


def get_health() -> dict[str, Any]:
    return response.success(
        MODULE,
        "health",
        data={
            "service":
                "plan",

            "one_shot_planner":
                True,

            "runtime_task_interpreter":
                True,

            "executes_robot":
                False,

            "task_interpreter_skill":
                "agent/skills/runtime_task_interpreter.md",
        },
    )
