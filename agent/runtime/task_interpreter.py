"""Two-stage Task -> Goal/Coordination interpreter. No action sequence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from agent.planning.structured_provider import chat_structured
from .coordination_schema import CoordinationSpec, DeviceRule, MaintainFactUntilGoal
from .goal_schema import GoalCondition, GoalSet
from .world_state import WorldState

_STAGE1 = (
    "Convert the user robot task into desired semantic GOALS. Do not output actions, "
    "tool calls, device allocation, or prerequisite steps. Use exact entity IDs from "
    "world_state/grounded_targets. Dependencies are task-credit dependencies, not an "
    "action schedule. For pure relative arm motion, return motion_request instead of "
    "inventing an object goal. Only include device_hint when explicitly named. JSON only."
)
_STAGE2 = (
    "Extract ONLY explicit coordination constraints. Do not output actions, schedules, "
    "candidate tools, or inferred arm allocation. maintain_until preserves a fact until "
    "a named goal completes. device_rules are only for explicit user device/side choices."
)


@dataclass
class TaskSpec:
    goals: GoalSet | None
    coordination: CoordinationSpec
    motion_request: dict[str, Any] | None
    summary: str
    raw: dict[str, Any]


def _schema1():
    goal = {
        "type":"object",
        "properties":{
            "goal_id":{"type":"string"}, "subject":{"type":"string"},
            "field":{"type":"string"}, "operator":{"type":"string"}, "value":{},
            "depends_on":{"type":"array","items":{"type":"string"}},
        },
        "required":["goal_id","subject","field","operator","value","depends_on"],
    }
    return {
        "type":"object",
        "properties":{
            "summary":{"type":"string"},
            "goals":{"type":"array","items":goal},
            "motion_request":{
                "type":"object",
                "properties":{
                    "direction":{"type":"string"}, "distance_m":{"type":"number"},
                    "device_hint":{"type":"string"},
                },
                "required":["direction","distance_m"],
            },
        },
        "required":["summary","goals"],
    }


def _schema2():
    return {
        "type":"object",
        "properties":{
            "maintain_until":{"type":"array","items":{
                "type":"object",
                "properties":{
                    "subject":{"type":"string"}, "field":{"type":"string"},
                    "value":{}, "until_goal":{"type":"string"},
                },
                "required":["subject","field","value","until_goal"],
            }},
            "device_rules":{"type":"array","items":{
                "type":"object",
                "properties":{
                    "function_name":{"type":"string"}, "subject_arg":{"type":"string"},
                    "subject_id":{"type":"string"}, "device_id":{"type":"string"},
                    "mode":{"type":"string"},
                },
                "required":["function_name","mode"],
            }},
        },
        "required":["maintain_until","device_rules"],
    }


def _call(stage, prompt, payload, schema, options):
    options = options or {}; row = options.get(stage) or {}
    msg = chat_structured(
        stage=stage,
        messages=[
            {"role":"system","content":prompt},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False)},
        ],
        response_schema=schema,
        provider=row.get("provider"), model=row.get("model"),
        api_key=options.get("openai_api_key"),
    )
    content = msg.get("content") if isinstance(msg, dict) else None
    value = json.loads(content) if isinstance(content, str) else None
    if not isinstance(value, dict):
        raise RuntimeError(f"{stage} structured output 必須是 JSON object")
    return value


def _parse_goals(raw, world):
    result = []
    for x in raw.get("goals") or []:
        subject = str(x.get("subject") or "")
        if not world.has_entity(subject):
            raise RuntimeError(f"goal references unknown entity: {subject!r}")
        result.append(GoalCondition(
            goal_id=str(x["goal_id"]), subject=subject, field=str(x["field"]),
            value=deepcopy(x.get("value")), operator=str(x.get("operator") or "eq"),
            depends_on=tuple(x.get("depends_on") or ()),
        ))
    return GoalSet(result) if result else None


def _parse_motion(raw):
    value = raw.get("motion_request")
    if value is None:
        return None
    direction = str(value.get("direction") or "").strip()
    distance = float(value.get("distance_m"))
    if not direction or distance <= 0:
        raise RuntimeError("invalid motion_request")
    result = {"direction":direction, "distance_m":distance}
    if str(value.get("device_hint") or "").strip():
        result["device_hint"] = str(value["device_hint"]).strip()
    return result


def _parse_coord(raw, world, goals):
    maintain = []
    for x in raw.get("maintain_until") or []:
        if goals is None or not world.has_entity(str(x.get("subject") or "")):
            raise RuntimeError(f"invalid maintain_until: {x!r}")
        goals.by_id(str(x["until_goal"]))
        maintain.append(MaintainFactUntilGoal(
            subject=str(x["subject"]), field=str(x["field"]),
            value=deepcopy(x.get("value")), until_goal=str(x["until_goal"]),
        ))
    rules = []
    for x in raw.get("device_rules") or []:
        opt = lambda k: str(x[k]).strip() if x.get(k) is not None else None
        rules.append(DeviceRule(
            function_name=str(x["function_name"]), subject_arg=opt("subject_arg"),
            subject_id=opt("subject_id"), device_id=opt("device_id"),
            mode=str(x.get("mode") or "required"),
        ))
    return CoordinationSpec(maintain_until=maintain, device_rules=rules)


def interpret_task(user_text: str, *, world: WorldState, grounded_targets=None,
                   llm_options: dict[str, Any] | None = None) -> TaskSpec:
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text 必須是非空字串")
    stage1 = _call("stage1", _STAGE1, {
        "user_text":user_text, "world_state":world.snapshot(),
        "grounded_targets":grounded_targets,
    }, _schema1(), llm_options)
    goals, motion = _parse_goals(stage1, world), _parse_motion(stage1)
    if goals is None and motion is None:
        raise RuntimeError("Stage1 必須產生 goals 或 motion_request")
    stage2 = _call("stage2", _STAGE2, {
        "user_text":user_text,
        "goals":goals.summary(world) if goals else [],
        "motion_request":motion, "world_state":world.snapshot(),
    }, _schema2(), llm_options)
    return TaskSpec(
        goals=goals, coordination=_parse_coord(stage2, world, goals),
        motion_request=motion, summary=str(stage1.get("summary") or "").strip(),
        raw={"stage1":stage1,"stage2":stage2},
    )
