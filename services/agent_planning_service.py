"""Thin read-only Task -> Goal/Coordination service."""
from __future__ import annotations
from copy import deepcopy
from time import perf_counter
from typing import Any

from agent.planning.target_grounding import ground_task_targets
from agent.runtime.task_interpreter import interpret_task
from services import world_state_service


def _goals(value):
    if value is None: return []
    return [{
        "goal_id":g.goal_id, "subject":g.subject, "field":g.field,
        "operator":g.operator, "value":deepcopy(g.value),
        "depends_on":list(g.depends_on),
    } for g in value.conditions]


def _coord(value):
    return {
        "maintain_until":[{
            "subject":x.subject,"field":x.field,"value":deepcopy(x.value),
            "until_goal":x.until_goal,
        } for x in value.maintain_until],
        "device_rules":[{
            "function_name":x.function_name,"subject_arg":x.subject_arg,
            "subject_id":x.subject_id,"device_id":x.device_id,"mode":x.mode,
        } for x in value.device_rules],
        "action_gates":[],
    }


def build_read_only_plan(user_text: str, *, context: dict[str, Any] | None = None,
                         device_registry=None, llm_options=None):
    del device_registry
    context = deepcopy(context or {}); started = perf_counter()
    detected = context.get("detected_objects")
    if not isinstance(detected, dict):
        detected = world_state_service.get_detected_objects()
    world = world_state_service.build_world_state(
        context=context, detected_objects=detected,
        read_live_vision=False, read_live_robot=True,
    )
    grounded = ground_task_targets(user_text, detected)
    task = interpret_task(
        user_text, world=world, grounded_targets=grounded, llm_options=llm_options,
    )
    return {
        "planning_mode":"closed_loop_goal_ir", "user_text":user_text,
        "answer":task.summary, "world_state":world.snapshot(),
        "grounded_targets":deepcopy(grounded), "goals":_goals(task.goals),
        "motion_request":deepcopy(task.motion_request),
        "coordination":_coord(task.coordination), "runtime_ready":True,
        "will_execute":False, "executable":False,
        "timings_ms":{"planning_service_total":round((perf_counter()-started)*1000,2)},
        "planner_trace":deepcopy(task.raw), "abstract_plan":None,
        "validated_plan":None, "device_statuses":[], "allocation":None,
        "resolved_plan":None, "execution_preview":[], "perception_requirements":[],
        "planning_perception_calls":[], "task_facts":deepcopy(context.get("task_facts") or []),
        "tool_calls":[],
    }
