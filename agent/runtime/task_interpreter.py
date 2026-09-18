"""
Human task -> semantic GoalSet / CoordinationSpec interpreter.


LLM instructions:
    agent/skills/runtime_task_interpreter.md

Structured inference provider:
    agent/runtime/structured_provider.py
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .structured_provider import chat_structured
from .world_state import WorldState

from .goals import (
    GoalCondition,
    GoalSet,
)

from .coordination import (
    CoordinationSpec,
    DeviceRule,
    MaintainFactUntilGoal,
)


_SKILL_PATH = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "runtime_task_interpreter.md"
)

_ALLOWED_OPERATORS = {"eq", "ne", "in"}
_ALLOWED_DEVICE_RULE_MODES = {"required", "preferred"}
_ALLOWED_MOTION_DIRECTIONS = {"x+", "x-", "y+", "y-", "z+", "z-"}


@dataclass(frozen=True)
class TaskSpec:
    """Semantic task contract produced before the runtime loop starts."""

    goals: GoalSet | None
    coordination: CoordinationSpec
    motion_request: dict[str, Any] | None
    summary: str
    raw: dict[str, Any]


@lru_cache(maxsize=1)
def _load_skill() -> str:
    if not _SKILL_PATH.is_file():
        raise RuntimeError(
            f"Task interpreter skill 不存在：{_SKILL_PATH}"
        )

    content = _SKILL_PATH.read_text(
        encoding="utf-8",
    ).strip()

    if content.startswith("---"):
        parts = content.split("---", 2)

        if len(parts) == 3:
            content = parts[2].strip()

    if not content:
        raise RuntimeError(
            "runtime_task_interpreter.md 為空"
        )

    return content



def _semantic_value_schema() -> dict[str, Any]:
    """
    Runtime semantic values are intentionally limited to scalars
    or a flat scalar list.

    This prevents the LLM from producing shapes such as:
        {"drawer_1": "drawer_1"}

    when the expected value is simply:
        "drawer_1"
    """
    scalar_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
        ]
    }

    return {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
            {
                "type": "array",
                "items": scalar_schema,
            },
        ]
    }


def _goals_schema() -> dict[str, Any]:
    goal_schema = {
        "type": "object",
        "properties": {
            "goal_id": {"type": "string"},
            "subject": {"type": "string"},
            "field": {"type": "string"},
            "operator": {
                "type": "string",
                "enum": ["eq", "ne", "in"],
            },
            "value": _semantic_value_schema(),
            "depends_on": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "goal_id",
            "subject",
            "field",
            "operator",
            "value",
            "depends_on",
        ],
        "additionalProperties": False,
    }

    motion_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": sorted(_ALLOWED_MOTION_DIRECTIONS),
            },
            "distance_m": {
                "type": "number",
                "exclusiveMinimum": 0.0,
            },
            "device_hint": {"type": "string"},
        },
        "required": [
            "direction",
            "distance_m",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "goals": {
                "type": "array",
                "items": goal_schema,
            },
            "motion_request": motion_schema,
        },
        "required": [
            "summary",
            "goals",
        ],
        "additionalProperties": False,
    }


def _coordination_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "maintain_until": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "field": {"type": "string"},
                        "value": _semantic_value_schema(),
                        "until_goal": {"type": "string"},
                    },
                    "required": [
                        "subject",
                        "field",
                        "value",
                        "until_goal",
                    ],
                    "additionalProperties": False,
                },
            },
            "device_rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string"},
                        "subject_arg": {"type": "string"},
                        "subject_id": {"type": "string"},
                        "device_id": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": [
                                "required",
                                "preferred",
                            ],
                        },
                    },
                    "required": [
                        "function_name",
                        "mode",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "maintain_until",
            "device_rules",
        ],
        "additionalProperties": False,
    }


def _stage_options(
    llm_options: dict[str, Any] | None,
    stage_key: str,
) -> tuple[str | None, str | None]:
    """
    Read provider/model for one stage.

    If provider is omitted, structured_provider defaults to local.
    """

    options = (
        llm_options
        if isinstance(llm_options, dict)
        else {}
    )

    row = options.get(stage_key)

    if not isinstance(row, dict):
        row = {}

    provider = row.get("provider")
    model = row.get("model")

    if provider is not None:
        if (
            not isinstance(provider, str)
            or not provider.strip()
        ):
            raise ValueError(
                f"{stage_key}.provider 必須是非空字串"
            )

        provider = provider.strip()

    if model is not None:
        if (
            not isinstance(model, str)
            or not model.strip()
        ):
            raise ValueError(
                f"{stage_key}.model 必須是非空字串"
            )

        model = model.strip()

    return provider, model


def _call_stage(
    *,
    provider_stage: str,
    prompt_stage: str,
    payload: dict[str, Any],
    response_schema: dict[str, Any],
    llm_options: dict[str, Any] | None,
) -> dict[str, Any]:
    provider, model = _stage_options(
        llm_options,
        provider_stage,
    )

    user_payload = {
        "stage": prompt_stage,
        **deepcopy(payload),
    }

    message = chat_structured(
        stage=provider_stage,
        messages=[
            {
                "role": "system",
                "content": _load_skill(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                ),
            },
        ],
        response_schema=response_schema,
        provider=provider,
        model=model,
        temperature=0.0,
        max_tokens=2048,
        num_ctx=8192,
        timeout=60.0,
        keep_alive=-1,
        wait=True,
        owner=f"runtime_task_interpreter_{provider_stage}",
    )

    if not isinstance(message, dict):
        raise RuntimeError(
            f"{provider_stage} LLM response 必須是 message object"
        )

    content = message.get("content")

    if (
        not isinstance(content, str)
        or not content.strip()
    ):
        raise RuntimeError(
            f"{provider_stage} LLM response 缺少 content"
        )

    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{provider_stage} structured output 不是有效 JSON："
            f"{exc}"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(
            f"{provider_stage} structured output 必須是 JSON object"
        )

    return value


def _validate_goal_dependency_graph(
    goals: list[GoalCondition],
) -> None:
    by_id = {
        goal.goal_id: goal
        for goal in goals
    }

    if len(by_id) != len(goals):
        raise RuntimeError(
            "goal_id 不可重複"
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(goal_id: str) -> None:
        if goal_id in visited:
            return

        if goal_id in visiting:
            raise RuntimeError(
                "goal dependency 不可形成 cycle："
                f"{goal_id}"
            )

        visiting.add(goal_id)
        goal = by_id[goal_id]

        for dependency in goal.depends_on:
            if dependency == goal_id:
                raise RuntimeError(
                    f"goal {goal_id} 不可依賴自己"
                )

            if dependency not in by_id:
                raise RuntimeError(
                    f"goal {goal_id} references unknown "
                    f"dependency: {dependency!r}"
                )

            visit(dependency)

        visiting.remove(goal_id)
        visited.add(goal_id)

    for goal_id in by_id:
        visit(goal_id)


def _parse_goals(
    raw: dict[str, Any],
    *,
    world: WorldState,
) -> GoalSet | None:
    rows = raw.get("goals")

    if rows is None:
        rows = []

    if not isinstance(rows, list):
        raise RuntimeError(
            "goals 必須是 list"
        )

    parsed: list[GoalCondition] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"goals[{index}] 必須是 object"
            )

        goal_id = str(
            row.get("goal_id") or ""
        ).strip()

        subject = str(
            row.get("subject") or ""
        ).strip()

        field_name = str(
            row.get("field") or ""
        ).strip()

        operator = str(
            row.get("operator") or "eq"
        ).strip()

        depends_on_raw = row.get("depends_on")

        if not goal_id:
            raise RuntimeError(
                f"goals[{index}].goal_id 不可為空"
            )

        if not subject:
            raise RuntimeError(
                f"goals[{index}].subject 不可為空"
            )

        if not world.has_entity(subject):
            raise RuntimeError(
                "goal references unknown entity: "
                f"{subject!r}"
            )

        if not field_name:
            raise RuntimeError(
                f"goals[{index}].field 不可為空"
            )

        if operator not in _ALLOWED_OPERATORS:
            raise RuntimeError(
                f"goals[{index}].operator 無效："
                f"{operator!r}"
            )

        if not isinstance(depends_on_raw, list):
            raise RuntimeError(
                f"goals[{index}].depends_on 必須是 list"
            )

        depends_on = tuple(
            str(value).strip()
            for value in depends_on_raw
            if str(value).strip()
        )

        value = deepcopy(
            row.get("value")
        )

        if (
            operator == "in"
            and not isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            )
        ):
            raise RuntimeError(
                f"goals[{index}] 使用 operator='in' 時 "
                "value 必須是 collection"
            )

        parsed.append(
            GoalCondition(
                goal_id=goal_id,
                subject=subject,
                field=field_name,
                value=value,
                operator=operator,
                depends_on=depends_on,
            )
        )

    if not parsed:
        return None

    _validate_goal_dependency_graph(parsed)

    return GoalSet(parsed)


def _parse_motion_request(
    raw: dict[str, Any],
    *,
    world: WorldState,
) -> dict[str, Any] | None:
    value = raw.get("motion_request")

    if value is None:
        return None

    if not isinstance(value, dict):
        raise RuntimeError(
            "motion_request 必須是 object"
        )

    direction = str(
        value.get("direction") or ""
    ).strip()

    if direction not in _ALLOWED_MOTION_DIRECTIONS:
        raise RuntimeError(
            "motion_request.direction 無效："
            f"{direction!r}"
        )

    try:
        distance_m = float(
            value.get("distance_m")
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            "motion_request.distance_m 必須是 number"
        ) from exc

    if distance_m <= 0.0:
        raise RuntimeError(
            "motion_request.distance_m 必須 > 0"
        )

    result: dict[str, Any] = {
        "direction": direction,
        "distance_m": distance_m,
    }

    device_hint = str(
        value.get("device_hint") or ""
    ).strip()

    if device_hint:
        if not world.has_entity(device_hint):
            raise RuntimeError(
                "motion_request references unknown device: "
                f"{device_hint!r}"
            )

        result["device_hint"] = device_hint

    return result


def _optional_string(
    row: dict[str, Any],
    key: str,
) -> str | None:
    value = row.get(key)

    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def _parse_coordination(
    raw: dict[str, Any],
    *,
    world: WorldState,
    goals: GoalSet | None,
) -> CoordinationSpec:
    maintain_rows = raw.get("maintain_until")
    rule_rows = raw.get("device_rules")

    if not isinstance(maintain_rows, list):
        raise RuntimeError(
            "maintain_until 必須是 list"
        )

    if not isinstance(rule_rows, list):
        raise RuntimeError(
            "device_rules 必須是 list"
        )

    maintain: list[
        MaintainFactUntilGoal
    ] = []

    for index, row in enumerate(maintain_rows):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"maintain_until[{index}] 必須是 object"
            )

        subject = str(
            row.get("subject") or ""
        ).strip()

        field_name = str(
            row.get("field") or ""
        ).strip()

        until_goal = str(
            row.get("until_goal") or ""
        ).strip()

        if not subject:
            raise RuntimeError(
                f"maintain_until[{index}].subject 不可為空"
            )

        if not world.has_entity(subject):
            raise RuntimeError(
                "maintain_until references unknown entity: "
                f"{subject!r}"
            )

        if not field_name:
            raise RuntimeError(
                f"maintain_until[{index}].field 不可為空"
            )

        if goals is None:
            raise RuntimeError(
                "沒有 semantic goals 時不可建立 maintain_until"
            )

        try:
            goals.by_id(until_goal)
        except KeyError as exc:
            raise RuntimeError(
                "maintain_until references unknown goal: "
                f"{until_goal!r}"
            ) from exc

        maintain.append(
            MaintainFactUntilGoal(
                subject=subject,
                field=field_name,
                value=deepcopy(
                    row.get("value")
                ),
                until_goal=until_goal,
            )
        )

    rules: list[DeviceRule] = []

    for index, row in enumerate(rule_rows):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"device_rules[{index}] 必須是 object"
            )

        function_name = str(
            row.get("function_name") or ""
        ).strip()

        mode = str(
            row.get("mode") or "required"
        ).strip()

        if not function_name:
            raise RuntimeError(
                f"device_rules[{index}].function_name 不可為空"
            )

        if mode not in _ALLOWED_DEVICE_RULE_MODES:
            raise RuntimeError(
                f"device_rules[{index}].mode 無效："
                f"{mode!r}"
            )

        subject_arg = _optional_string(
            row,
            "subject_arg",
        )
        subject_id = _optional_string(
            row,
            "subject_id",
        )
        device_id = _optional_string(
            row,
            "device_id",
        )

        if (
            subject_id is not None
            and not world.has_entity(subject_id)
        ):
            raise RuntimeError(
                "device_rule references unknown subject entity: "
                f"{subject_id!r}"
            )

        if (
            device_id is not None
            and not world.has_entity(device_id)
        ):
            raise RuntimeError(
                "device_rule references unknown device entity: "
                f"{device_id!r}"
            )

        rules.append(
            DeviceRule(
                function_name=function_name,
                subject_arg=subject_arg,
                subject_id=subject_id,
                device_id=device_id,
                mode=mode,
            )
        )

    return CoordinationSpec(
        maintain_until=maintain,
        device_rules=rules,
    )


def interpret_task(
    user_text: str,
    *,
    world: WorldState,
    grounded_targets: Any = None,
    llm_options: dict[str, Any] | None = None,
) -> TaskSpec:
    """
    Convert one human task into semantic runtime intent.

    Stage 1:
        user task -> goals / optional relative motion

    Stage 2:
        goals + task -> explicit coordination constraints

    No action sequence is produced here.
    """

    if (
        not isinstance(user_text, str)
        or not user_text.strip()
    ):
        raise ValueError(
            "user_text 必須是非空字串"
        )

    if not isinstance(world, WorldState):
        raise TypeError(
            "world 必須是 WorldState"
        )

    user_text = user_text.strip()

    stage1 = _call_stage(
        provider_stage="stage1",
        prompt_stage="goals",
        payload={
            "user_text": user_text,
            "world_state": world.snapshot(),
            "grounded_targets": deepcopy(
                grounded_targets
            ),
        },
        response_schema=_goals_schema(),
        llm_options=llm_options,
    )

    goals = _parse_goals(
        stage1,
        world=world,
    )

    motion_request = _parse_motion_request(
        stage1,
        world=world,
    )

    if (
        goals is None
        and motion_request is None
    ):
        raise RuntimeError(
            "Task Interpreter 必須產生 "
            "semantic goals 或 motion_request"
        )

    stage2 = _call_stage(
        provider_stage="stage2",
        prompt_stage="coordination",
        payload={
            "user_text": user_text,
            "world_state": world.snapshot(),
            "goals": (
                goals.summary(world)
                if goals is not None
                else []
            ),
            "motion_request": deepcopy(
                motion_request
            ),
        },
        response_schema=_coordination_schema(),
        llm_options=llm_options,
    )

    coordination = _parse_coordination(
        stage2,
        world=world,
        goals=goals,
    )

    summary = str(
        stage1.get("summary") or ""
    ).strip()

    return TaskSpec(
        goals=goals,
        coordination=coordination,
        motion_request=motion_request,
        summary=summary,
        raw={
            "stage1": deepcopy(stage1),
            "stage2": deepcopy(stage2),
        },
    )
