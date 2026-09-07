"""唯讀 Agent 規劃流程的薄服務層。

本模組只把 LLM Abstract Plan 接到 Planning Pipeline，不呼叫 executor。
與舊 action 路徑分離，未來遷移或移除時不需要拆解 agent_service。
"""

from __future__ import annotations

from copy import deepcopy
import re
from time import perf_counter
from typing import Any

from agent.planning import (
    DeviceRegistry,
    build_abstract_plan,
    derive_perception_requirements,
    prepare_plan,
)
from agent.planning.structured_provider import resolve_provider
from agent.planning.target_grounding import ground_task_targets
from agent.tools import get_tool, get_tools_for_prompt
from services import llm_service


_OBJECT_PLACEMENT_TOOLS = {
    "place_object_in_trash_can",
    "place_object_in_top_cabinet",
    "place_object_in_second_drawer",
}
_PLACEMENT_CONTAINERS = {
    "place_object_in_trash_can": ("trash_can", "open_trash_can"),
    "place_object_in_top_cabinet": ("top_cabinet", "open_top_cabinet"),
    "place_object_in_second_drawer": ("second_drawer", "open_second_drawer"),
}


def _inspect_visual_state(**kwargs):
    from services.task_perception_service import inspect_visual_state

    return inspect_visual_state(**kwargs)


def _planning_tool_catalogs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Expose request-local object selection to read-only abstract planning."""
    prompt_catalog = get_tools_for_prompt()
    execution_catalog = get_tool()
    object_id_parameter = {
        "type": "string",
        "description": "必須原樣選自本次 detected_objects.objects 的 object_id。",
    }
    for tool in prompt_catalog:
        if tool.get("api_function") not in _OBJECT_PLACEMENT_TOOLS:
            continue
        tool["parameters"] = {"object_id": deepcopy(object_id_parameter)}
        tool["required"] = ["object_id"]
        tool["description"] = (
            str(tool.get("description") or "")
            + " 規劃階段只選擇 detected_objects 中既有的 object_id。"
        )
    for function_name in _OBJECT_PLACEMENT_TOOLS:
        tool = execution_catalog.get(function_name)
        if not isinstance(tool, dict):
            continue
        tool["parameters"] = {"object_id": deepcopy(object_id_parameter)}
        tool["required"] = ["object_id"]
    return prompt_catalog, execution_catalog


def _validate_planned_object_ids(plan: dict[str, Any], context: dict[str, Any]) -> None:
    detected = context.get("detected_objects")
    objects = detected.get("objects") if isinstance(detected, dict) else None
    known_ids = {
        str(row["object_id"])
        for row in (objects or [])
        if isinstance(row, dict) and row.get("object_id")
    }
    for step in plan.get("steps", []):
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict) or action.get("function_name") not in _OBJECT_PLACEMENT_TOOLS:
            continue
        arguments = action.get("arguments") or {}
        object_id = arguments.get("object_id") if isinstance(arguments, dict) else None
        if not isinstance(object_id, str) or object_id not in known_ids:
            raise ValueError(f"planning step 使用未知 object_id：{object_id!r}")


def _add_global_distinct_role_preferences(
    plan: dict[str, Any],
    parallel_pairs: list[list[str]],
    role_mapping: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Prefer distinct devices only for roles containing parallel actions."""
    updated = deepcopy(plan)
    constrained_pairs = set()
    for constraint in updated.get("constraints", []):
        roles = constraint.get("roles") if isinstance(constraint, dict) else None
        if not isinstance(roles, list) or len(roles) < 2:
            continue
        constrained_pairs.add(tuple(sorted(roles)))

    preferences = updated.setdefault("assignment_preferences", [])
    existing_pairs = set()
    for preference in preferences:
        roles = preference.get("roles") if isinstance(preference, dict) else None
        if isinstance(roles, list) and len(roles) >= 2:
            existing_pairs.add(tuple(sorted(roles)))

    added = []
    trace_by_role_pair = {}
    for action_pair in sorted(parallel_pairs):
        if not isinstance(action_pair, list) or len(action_pair) != 2:
            continue
        left_action, right_action = action_pair
        left_role = role_mapping.get(left_action)
        right_role = role_mapping.get(right_action)
        if not left_role or not right_role or left_role == right_role:
            continue
        role_pair = tuple(sorted((left_role, right_role)))
        if role_pair in constrained_pairs:
            continue
        if role_pair in trace_by_role_pair:
            trace_by_role_pair[role_pair]["parallel_pairs"].append(list(action_pair))
            continue
        if role_pair in existing_pairs:
            continue
        preference = {
            "type": "prefer_distinct_assignment",
            "roles": list(role_pair),
            "weight": 1.0,
            "reason": "可平行 actions 優先分配到不同 compatible resources",
            "source": "global_allocation_policy",
        }
        preferences.append(preference)
        existing_pairs.add(role_pair)
        trace_entry = deepcopy(preference)
        trace_entry["parallel_pairs"] = [list(action_pair)]
        added.append(trace_entry)
        trace_by_role_pair[role_pair] = trace_entry
    return updated, added


def _condition_actions_with_planning_perception(
    actions: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    conditioned = deepcopy(actions)
    facts, calls, required_edges = [], [], []
    destinations = []
    for action in conditioned:
        route = _PLACEMENT_CONTAINERS.get(action.get("function_name"))
        if route and route not in destinations:
            destinations.append(route)

    for subject, open_function in destinations:
        fact = _inspect_visual_state(
            subject=subject,
            predicate="open_state",
            context=context,
        )
        facts.append(deepcopy(fact))
        calls.append({
            "tool": "inspect_visual_state",
            "arguments": {"subject": subject, "predicate": "open_state"},
            "result": deepcopy(fact),
        })
        state = str(fact.get("value") or "").strip().lower()
        if state not in {"open", "closed"}:
            raise ValueError(f"{subject}.open_state 無法用於 planning：{state!r}")

        existing_open = [
            row for row in conditioned
            if row.get("function_name") == open_function
        ]
        conditioned = [
            row for row in conditioned
            if row.get("function_name") != open_function
        ]
        if state == "open":
            continue

        place_index = next(
            index for index, row in enumerate(conditioned)
            if _PLACEMENT_CONTAINERS.get(row.get("function_name"), (None,))[0] == subject
        )
        open_action = existing_open[0] if existing_open else {
            "key": f"planning_open_{subject}",
            "function_name": open_function,
            "arguments": {},
        }
        conditioned.insert(place_index, open_action)
        place_action = conditioned[place_index + 1]
        required_edges.append({
            "before": open_action["key"],
            "after": place_action["key"],
        })

    return {
        "actions": conditioned,
        "task_facts": facts,
        "planning_perception_calls": calls,
        "required_edges": required_edges,
    }


_CONTAINER_TERMS = {
    "trash_can": ("垃圾桶",),
    "second_drawer": ("第二層抽屜", "第二個抽屜", "抽屜"),
    "top_cabinet": ("上層櫃門", "上層櫃", "櫃門", "櫃子"),
}


def _extract_explicit_task_facts(user_text: str) -> list[dict[str, Any]]:
    """Extract only directly asserted current container states."""
    facts = []
    for subject, terms in _CONTAINER_TERMS.items():
        matched_value = None
        for term in terms:
            match = re.search(
                rf"{re.escape(term)}(?:已經|現在)?(?:是)?(開著|開的|關著|關的)",
                user_text,
            )
            if match:
                matched_value = "open" if match.group(1).startswith("開") else "closed"
                break
        if matched_value is not None:
            facts.append({
                "subject": subject,
                "predicate": "open_state",
                "value": matched_value,
                "source": "user_instruction",
            })
    return facts


def build_read_only_plan(
    user_text: str,
    *,
    context: dict[str, Any] | None = None,
    device_registry: DeviceRegistry | None = None,
    llm_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and resolve a plan while keeping execution explicitly disabled."""
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text 必須是非空字串")
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise ValueError("context 必須是 dict")

    total_started = perf_counter()
    stage_started = perf_counter()
    registry = device_registry or DeviceRegistry.from_arm_config()
    registry_ms = (perf_counter() - stage_started) * 1000
    stage_started = perf_counter()
    effective_llm_options = deepcopy(llm_options or {})
    planning_context = deepcopy(context)
    planning_context["grounded_targets"] = ground_task_targets(
        user_text, planning_context.get("detected_objects", {})
    )
    explicit_task_facts = _extract_explicit_task_facts(user_text)
    existing_task_facts = planning_context.get("task_facts")
    if isinstance(existing_task_facts, list):
        explicit_task_facts = [*deepcopy(existing_task_facts), *explicit_task_facts]
    planning_context["task_facts"] = explicit_task_facts
    prompt_tool_catalog, validation_tool_catalog = _planning_tool_catalogs()
    stage1_provider = resolve_provider(
        "stage1", (effective_llm_options.get("stage1") or {}).get("provider")
    )
    stage2_provider = resolve_provider(
        "stage2", (effective_llm_options.get("stage2") or {}).get("provider")
    )

    def run_semantic_planning() -> dict[str, Any]:
        return build_abstract_plan(
            user_text=user_text,
            tool_catalog=prompt_tool_catalog,
            context=deepcopy(planning_context),
            llm_options=effective_llm_options,
            planning_fact_resolver=_condition_actions_with_planning_perception,
        )

    if stage1_provider == stage2_provider == "openai":
        llm_result = run_semantic_planning()
    else:
        with llm_service.runtime_session(wait=True, owner="abstract_planner"):
            llm_result = run_semantic_planning()
    llm_result.setdefault("planner_trace", {})["grounded_targets"] = deepcopy(
        planning_context["grounded_targets"]
    )
    llm_result["plan"], global_preferences = _add_global_distinct_role_preferences(
        llm_result["plan"],
        llm_result.get("planner_trace", {}).get("parallel_pairs", []),
        llm_result.get("planner_trace", {}).get("role_mapping", {}),
    )
    llm_result.setdefault("planner_trace", {})[
        "global_assignment_preferences"
    ] = deepcopy(global_preferences)
    _validate_planned_object_ids(llm_result["plan"], planning_context)
    abstract_service_ms = (perf_counter() - stage_started) * 1000
    stage_started = perf_counter()
    pipeline_result = prepare_plan(
        abstract_plan=llm_result["plan"],
        tool_catalog=validation_tool_catalog,
        device_registry=registry,
    )
    pipeline_ms = (perf_counter() - stage_started) * 1000
    perception_requirements = derive_perception_requirements(
        pipeline_result["resolved_plan"],
        context=planning_context,
    )
    total_ms = (perf_counter() - total_started) * 1000
    timings = deepcopy(llm_result.get("timings_ms", {}))
    timings.update({
        "registry_build": round(registry_ms, 2),
        "abstract_service": round(abstract_service_ms, 2),
        "validation_probe_allocation_resolution": round(pipeline_ms, 2),
        "planning_service_total": round(total_ms, 2),
    })

    return {
        "planning_mode": "abstract",
        "user_text": llm_result["user_text"],
        "answer": llm_result["answer"],
        # Read-only diagnostics: exposes semantic decisions, never executor state.
        "planner_trace": deepcopy(llm_result.get("planner_trace", {})),
        "timings_ms": timings,
        "abstract_plan": pipeline_result["abstract_plan"],
        "validated_plan": pipeline_result["validated_plan"],
        "device_statuses": pipeline_result["device_statuses"],
        "allocation": pipeline_result["allocation"],
        "resolved_plan": pipeline_result["resolved_plan"],
        "execution_preview": pipeline_result["execution_preview"],
        "perception_requirements": perception_requirements,
        "task_facts": deepcopy(llm_result.get("task_facts", explicit_task_facts)),
        "planning_perception_calls": deepcopy(
            llm_result.get("planning_perception_calls", [])
        ),
        # Transitional compatibility for clients that still read this field.
        "tool_calls": [],
        "will_execute": False,
        "executable": False,
    }
