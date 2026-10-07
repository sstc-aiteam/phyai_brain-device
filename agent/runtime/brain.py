"""
Closed-loop runtime decision brain.

The RuntimeBrain chooses one dispatch for the CURRENT WorldState.

LLM behavior instructions:
    agent/skills/runtime_brain.md

Structured inference provider:
    agent/runtime/structured_provider.py

This module does NOT:
- build a full future action sequence
- choose concrete device IDs
- execute robot commands
- mutate WorldState as truth
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .coordination import CoordinationSpec
from .critical_state import build_critical_state
from .dispatch import (
    ACTION_DECISION_PURPOSES,
    ActionDecisionContext,
    ActionIntent,
    DispatchDecision,
)
from .goals import GoalSet
from .structured_provider import chat_structured
from .tool_model import (
    ParameterSpec,
    ToolCatalog,
    ToolSpec,
)
from .world_state import WorldState


_SKILL_PATH = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "runtime_brain.md"
)


@lru_cache(maxsize=1)
def _load_skill() -> str:
    if not _SKILL_PATH.is_file():
        raise RuntimeError(
            f"Runtime brain skill 不存在：{_SKILL_PATH}"
        )

    content = _SKILL_PATH.read_text(
        encoding="utf-8",
    ).strip()

    # Strip optional YAML front matter.
    if content.startswith("---"):
        parts = content.split("---", 2)

        if len(parts) == 3:
            content = parts[2].strip()

    if not content:
        raise RuntimeError(
            "runtime_brain.md 為空"
        )

    return content


def _parameter_schema(
    spec: ParameterSpec,
) -> dict[str, Any]:
    """
    Convert one Tool ParameterSpec into the JSON schema exposed to the LLM.
    """

    if spec.schema:
        schema = deepcopy(
            spec.schema
        )
    else:
        schema = {
            "type": "string",
        }

    if spec.allowed_values:
        schema["enum"] = list(
            spec.allowed_values
        )

    return schema


def _action_schema(
    tool: ToolSpec,
) -> dict[str, Any]:
    """
    Build a strict action schema for one available tool.

    This makes the structured-output layer itself reject:
    - invented tool names
    - invented argument names
    - missing required arguments
    - invalid enum/scalar shapes
    """

    properties = {
        name: _parameter_schema(
            spec
        )
        for name, spec
        in tool.parameters.items()
    }

    required = [
        name
        for name, spec
        in tool.parameters.items()
        if spec.required
    ]

    return {
        "type": "object",
        "properties": {
            "function_name": {
                "type": "string",
                "enum": [
                    tool.name,
                ],
            },
            "arguments": {
                "type": "object",
                "properties":
                    properties,
                "required":
                    required,
                "additionalProperties":
                    False,
            },
        },
        "required": [
            "function_name",
            "arguments",
        ],
        "additionalProperties": False,
    }




def _action_context_schema() -> dict[str, Any]:
    target_fact_schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "field": {"type": "string"},
                    "operator": {
                        "type": "string",
                        "enum": ["eq", "ne", "in"],
                    },
                    "value": {},
                },
                "required": [
                    "subject",
                    "field",
                    "operator",
                    "value",
                ],
                "additionalProperties": False,
            },
            {
                "type": "null",
            },
        ],
    }

    return {
        "type": "object",
        "properties": {
            "purpose": {
                "type": "string",
                "enum": list(
                    ACTION_DECISION_PURPOSES
                ),
            },
            "related_goal_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "target_fact":
                target_fact_schema,
            "reason_summary": {
                "type": "string",
            },
        },
        "required": [
            "purpose",
            "related_goal_ids",
            "target_fact",
            "reason_summary",
        ],
        "additionalProperties": False,
    }


def _parse_action_context(
    raw: dict[str, Any],
    *,
    goals: GoalSet,
    world: WorldState,
) -> ActionDecisionContext:
    if not isinstance(raw, dict):
        raise RuntimeError(
            "Each action_context must be object"
        )

    purpose = str(
        raw.get("purpose") or ""
    ).strip()

    if purpose not in ACTION_DECISION_PURPOSES:
        raise RuntimeError(
            "action_context.purpose is invalid: "
            f"{purpose!r}"
        )

    related_goal_ids_raw = raw.get(
        "related_goal_ids"
    )

    if not isinstance(
        related_goal_ids_raw,
        list,
    ):
        raise RuntimeError(
            "action_context.related_goal_ids must be array"
        )

    related_goal_ids = tuple(
        str(value).strip()
        for value in related_goal_ids_raw
        if str(value).strip()
    )

    known_goal_ids = {
        goal.goal_id
        for goal in goals.conditions
    }

    unknown_goal_ids = sorted(
        set(related_goal_ids)
        - known_goal_ids
    )

    if unknown_goal_ids:
        raise RuntimeError(
            "action_context references unknown goal IDs: "
            f"{unknown_goal_ids}"
        )

    target_fact = raw.get(
        "target_fact"
    )

    if target_fact is not None:
        if not isinstance(target_fact, dict):
            raise RuntimeError(
                "action_context.target_fact must be object or null"
            )

        subject = str(
            target_fact.get("subject") or ""
        ).strip()
        field_name = str(
            target_fact.get("field") or ""
        ).strip()
        operator = str(
            target_fact.get("operator") or ""
        ).strip()

        if not subject or not field_name:
            raise RuntimeError(
                "action_context.target_fact requires subject and field"
            )

        if (
            subject != "$device"
            and not world.has_entity(subject)
        ):
            raise RuntimeError(
                "action_context.target_fact references unknown subject: "
                f"{subject!r}"
            )

        if operator not in {
            "eq",
            "ne",
            "in",
        }:
            raise RuntimeError(
                "action_context.target_fact.operator is invalid: "
                f"{operator!r}"
            )

        target_fact = {
            "subject": subject,
            "field": field_name,
            "operator": operator,
            "value": deepcopy(
                target_fact.get("value")
            ),
        }

    reason_summary = str(
        raw.get("reason_summary") or ""
    ).strip()

    if not reason_summary:
        raise RuntimeError(
            "action_context.reason_summary must be non-empty"
        )

    # Keep persisted/prompt memory compact and single-line.
    reason_summary = " ".join(
        reason_summary.split()
    )[:240]

    return ActionDecisionContext(
        purpose=purpose,
        related_goal_ids=related_goal_ids,
        target_fact=deepcopy(
            target_fact
        ),
        reason_summary=reason_summary,
    )


def _dispatch_schema(
    tools: ToolCatalog,
    *,
    max_parallel_actions: int,
    pending_intent_status: str | None = None,
) -> dict[str, Any]:
    tool_list = tools.values()

    if not tool_list:
        raise RuntimeError(
            "Runtime tool catalog 不可為空"
        )

    if (
        not isinstance(
            max_parallel_actions,
            int,
        )
        or max_parallel_actions < 1
    ):
        raise ValueError(
            "max_parallel_actions 必須是 >= 1 的整數"
        )

    action_schemas = [
        _action_schema(
            tool
        )
        for tool in tool_list
    ]

    if pending_intent_status == "READY":
        resolution_schema: dict[str, Any] = {
            "type": "string",
            "enum": [
                "continue",
                "abandon",
            ],
        }

    elif pending_intent_status == "BLOCKED":
        resolution_schema = {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "abandon",
                    ],
                },
                {
                    "type": "null",
                },
            ],
        }

    else:
        # There is no pending intent.  Strict structured output still requires
        # the field to exist, so force null instead of asking the model to
        # choose between continue / abandon / null.
        resolution_schema = {
            "type": "null",
        }

    reason_schema: dict[str, Any] = (
        {
            "type": "null",
        }
        if pending_intent_status is None
        else {
            "anyOf": [
                {
                    "type": "string",
                },
                {
                    "type": "null",
                },
            ],
        }
    )

    return {
        "type": "object",
        "properties": {
            "pending_intent_resolution":
                resolution_schema,
            "pending_intent_abandon_reason":
                reason_schema,
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems":
                    max_parallel_actions,
                "items": {
                    "anyOf":
                        action_schemas,
                },
            },
            "action_contexts": {
                "type": "array",
                "minItems": 1,
                "maxItems":
                    max_parallel_actions,
                "items":
                    _action_context_schema(),
            },
        },
        # Keep these fields explicit for strict structured output.
        # When no repair intent is involved, both are simply null.
        "required": [
            "pending_intent_resolution",
            "pending_intent_abandon_reason",
            "actions",
            "action_contexts",
        ],
        "additionalProperties": False,
    }


def _parse_action(
    raw: dict[str, Any],
    *,
    tools: ToolCatalog,
) -> ActionIntent:
    if not isinstance(
        raw,
        dict,
    ):
        raise RuntimeError(
            "Each dispatch action must be object"
        )

    function_name = raw.get(
        "function_name"
    )

    arguments = raw.get(
        "arguments"
    )

    if (
        not isinstance(
            function_name,
            str,
        )
        or not function_name.strip()
    ):
        raise RuntimeError(
            "action.function_name must be non-empty string"
        )

    function_name = (
        function_name.strip()
    )

    if not isinstance(
        arguments,
        dict,
    ):
        raise RuntimeError(
            "action.arguments must be object"
        )

    tool = tools.get(
        function_name
    )

    expected = set(
        tool.parameters
    )

    actual = set(
        arguments
    )

    missing = (
        set(
            tool.required_arguments
        )
        - actual
    )

    extra = (
        actual
        - expected
    )

    if missing:
        raise RuntimeError(
            f"{function_name} missing arguments: "
            f"{sorted(missing)}"
        )

    if extra:
        raise RuntimeError(
            f"{function_name} unsupported arguments: "
            f"{sorted(extra)}"
        )

    return ActionIntent(
        function_name=
            function_name,
        arguments=
            deepcopy(
                arguments
            ),
    )


class RuntimeBrain:
    """
    Closed-loop next-dispatch decision component.

    Default provider is resolved by structured_provider.py, which currently
    means local inference unless provider="remote" is explicitly requested.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.15,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        num_ctx: int = 8192,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = float(
            temperature
        )
        self.max_tokens = int(
            max_tokens
        )
        self.timeout = float(
            timeout
        )
        self.num_ctx = int(
            num_ctx
        )

    def decide_dispatch(
        self,
        *,
        world: WorldState,
        goals: GoalSet,
        tools: ToolCatalog,
        coordination: CoordinationSpec | None = None,
        device_context: Any = None,
        rejected_dispatches: list[
            dict[str, Any]
        ] | None = None,
        task_state: dict[
            str,
            Any,
        ] | None = None,
        task_progress_state: dict[
            str,
            Any,
        ] | None = None,
        repair_context: dict[
            str,
            Any,
        ] | None = None,
        loop_context: dict[
            str,
            Any,
        ] | None = None,
        max_parallel_actions: int = 2,
    ) -> DispatchDecision:
        if not isinstance(
            world,
            WorldState,
        ):
            raise TypeError(
                "world 必須是 WorldState"
            )

        if not isinstance(
            goals,
            GoalSet,
        ):
            raise TypeError(
                "goals 必須是 GoalSet"
            )

        if not isinstance(
            tools,
            ToolCatalog,
        ):
            raise TypeError(
                "tools 必須是 ToolCatalog"
            )

        if coordination is None:
            coordination = (
                CoordinationSpec()
            )

        if not isinstance(
            coordination,
            CoordinationSpec,
        ):
            raise TypeError(
                "coordination 必須是 CoordinationSpec"
            )

        if (
            not isinstance(
                max_parallel_actions,
                int,
            )
            or max_parallel_actions < 1
        ):
            raise ValueError(
                "max_parallel_actions 必須是 >= 1 的整數"
            )

        world_snapshot = world.snapshot()
        temporary_placement_targets = [
            entity_id
            for entity_id, entity
            in world_snapshot.get(
                "entities",
                {},
            ).items()
            if isinstance(entity, dict)
            and (
                "temporary_placement"
                in (entity.get("tags") or [])
                or "temporary_placement"
                in (
                    entity.get("affordances")
                    or []
                )
            )
        ]

        payload = {
            "critical_state":
                build_critical_state(
                    world,
                    goals,
                    coordination,
                ),

            # Internal short-term task execution memory.
            #
            # This summarizes current task progress / blockers / regressions
            # for the Brain, but CURRENT WorldState remains authoritative for
            # physical reality.
            "task_state":
                deepcopy(
                    task_state
                    or {}
                ),

            "world_state":
                world_snapshot,

            "temporary_placement_targets":
                temporary_placement_targets,

            "goal_state":
                goals.summary(
                    world
                ),

            "coordination_constraints":
                coordination.brain_view(
                    world,
                    goals,
                ),

            "device_context":
                deepcopy(
                    device_context
                ),

            "max_parallel_actions":
                max_parallel_actions,

            "available_tools":
                tools.brain_view(),

            "rejected_dispatches":
                deepcopy(
                    rejected_dispatches
                    or []
                ),

            "task_progress_state":
                deepcopy(
                    task_progress_state
                    or {}
                ),

            "repair_context":
                deepcopy(
                    repair_context
                    or {}
                ),

            "loop_context":
                deepcopy(
                    loop_context
                    or {}
                ),
        }

        message = chat_structured(
            stage="brain",
            messages=[
                {
                    "role": "system",
                    "content":
                        _load_skill(),
                },
                {
                    "role": "user",
                    "content":
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                },
            ],
            response_schema=
                _dispatch_schema(
                    tools,
                    max_parallel_actions=
                        max_parallel_actions,
                    pending_intent_status=(
                        (
                            repair_context
                            or {}
                        )
                        .get(
                            "active_frame",
                            {},
                        )
                        .get(
                            "status"
                        )
                        if isinstance(
                            (
                                repair_context
                                or {}
                            ).get(
                                "active_frame"
                            ),
                            dict,
                        )
                        else None
                    ),
                ),
            provider=
                self.provider,
            model=
                self.model,
            temperature=
                self.temperature,
            max_tokens=
                self.max_tokens,
            num_ctx=
                self.num_ctx,
            timeout=
                self.timeout,
            keep_alive=-1,
            wait=True,
            owner="runtime_brain",
            api_key=(
                self.api_key
                if str(
                    self.provider or ""
                ).strip().lower() == "openai"
                else None
            ),
        )

        if not isinstance(
            message,
            dict,
        ):
            raise RuntimeError(
                "Runtime brain LLM response "
                "必須是 message object"
            )

        content = message.get(
            "content"
        )

        if (
            not isinstance(
                content,
                str,
            )
            or not content.strip()
        ):
            raise RuntimeError(
                "Runtime brain LLM response "
                "缺少 content"
            )

        try:
            parsed = json.loads(
                content
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Runtime brain structured output "
                f"不是有效 JSON：{exc}"
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise RuntimeError(
                "Runtime brain structured output "
                "必須是 JSON object"
            )

        raw_actions = parsed.get(
            "actions"
        )

        if (
            not isinstance(
                raw_actions,
                list,
            )
            or not raw_actions
        ):
            raise RuntimeError(
                "Runtime brain missing "
                "non-empty actions"
            )

        if (
            len(
                raw_actions
            )
            > max_parallel_actions
        ):
            raise RuntimeError(
                "Runtime brain returned "
                f"{len(raw_actions)} actions, "
                f"max is {max_parallel_actions}"
            )

        actions = tuple(
            _parse_action(
                raw,
                tools=tools,
            )
            for raw in raw_actions
        )

        raw_action_contexts = parsed.get(
            "action_contexts"
        )

        if not isinstance(
            raw_action_contexts,
            list,
        ):
            raise RuntimeError(
                "Runtime brain missing action_contexts array"
            )

        if len(raw_action_contexts) != len(actions):
            raise RuntimeError(
                "action_contexts must align one-to-one with actions"
            )

        action_contexts = tuple(
            _parse_action_context(
                raw,
                goals=goals,
                world=world,
            )
            for raw in raw_action_contexts
        )

        resolution = parsed.get(
            "pending_intent_resolution"
        )

        if resolution is not None:
            if not isinstance(
                resolution,
                str,
            ):
                raise RuntimeError(
                    "pending_intent_resolution "
                    "must be string or null"
                )

            resolution = (
                resolution
                .strip()
                .lower()
            )

            if not resolution:
                resolution = None

        abandon_reason = parsed.get(
            "pending_intent_abandon_reason"
        )

        if abandon_reason is not None:
            if not isinstance(
                abandon_reason,
                str,
            ):
                raise RuntimeError(
                    "pending_intent_abandon_reason "
                    "must be string or null"
                )

            abandon_reason = (
                abandon_reason.strip()
                or None
            )

        return DispatchDecision(
            actions=actions,
            action_contexts=action_contexts,
            pending_intent_resolution=
                resolution,
            pending_intent_abandon_reason=
                abandon_reason,
        )

    def decide_next_action(
        self,
        *,
        world: WorldState,
        goals: GoalSet,
        tools: ToolCatalog,
        coordination: CoordinationSpec | None = None,
        device_context: Any = None,
        rejected_dispatches: list[
            dict[str, Any]
        ] | None = None,
    ) -> ActionIntent:
        """
        Backward-compatible single-action mode.

        New runtime code should prefer decide_dispatch().
        """

        dispatch = self.decide_dispatch(
            world=world,
            goals=goals,
            tools=tools,
            coordination=
                coordination,
            device_context=
                device_context,
            rejected_dispatches=
                rejected_dispatches,
            task_state=None,
            task_progress_state=None,
            repair_context=None,
            loop_context=None,
            max_parallel_actions=1,
        )

        return dispatch.actions[0]


__all__ = [
    "RuntimeBrain",
]
