"""串接語意規劃、計畫驗證、設備分配與執行預覽。

本模組是 abstract planner 的正式入口，但不呼叫 executor 或執行硬體命令。
語言模型只抽取 actions 與 pair relations；DAG、roles 與設備分配皆由
確定性的 planning 元件完成。
測試可直接注入 device_statuses；正式唯讀檢查則可由 DeviceRegistry
透過既有 service 探測設備狀態。
"""

from __future__ import annotations

import time
import traceback
from copy import deepcopy
from typing import Any

from agent.planning.action_extractor import (
    ActionExtractionError,
    extract_atomic_actions,
)
from agent.planning.action_validation import SemanticIRError
from agent.planning.device_allocator import (
    allocate_devices,
    resolve_assignments,
    validate_assignments,
)
from agent.planning.device_registry import DeviceRegistry
from agent.planning.ir_compiler import compile_pair_relations_to_plan
from agent.planning.plan_scheduler import build_execution_preview
from agent.planning.plan_schema import (
    PlanValidationError,
    validate_plan_against_catalogs,
)
from agent.planning.relation_classifier import (
    RelationClassificationError,
    classify_action_relations,
)
from agent.tools import get_tools_for_prompt


DEFAULT_PLANNING_POLICY = {
    "allow_duplicated_subplan": False,
    "prefer_distinct_assignment": False,
}


class AbstractPlanningError(RuntimeError):
    """Raised when the semantic planning stages cannot produce a safe Plan."""


def _derive_planning_policy(user_text: str) -> dict[str, bool]:
    compact = "".join(user_text.lower().split())
    return {
        "allow_duplicated_subplan": any(
            phrase in compact
            for phrase in ("重複", "兩次", "再做一次", "再執行一次")
        ),
        "prefer_distinct_assignment": any(
            phrase in compact
            for phrase in (
                "優先讓不同手臂分工",
                "優先由不同手臂分工",
                "優先讓不同設備分工",
                "優先由不同設備分工",
                "優先分工",
                "不同手臂分工",
                "不同設備分工",
            )
        ),
    }


def _normalize_planning_policy(
    planning_policy: dict[str, Any] | None,
    user_text: str,
) -> dict[str, bool]:
    if planning_policy is None:
        planning_policy = {}
    if not isinstance(planning_policy, dict):
        raise ValueError("planning_policy 必須是 dict")
    unknown = set(planning_policy) - set(DEFAULT_PLANNING_POLICY)
    if unknown:
        raise ValueError(f"planning_policy 包含未知欄位：{sorted(unknown)}")

    normalized = DEFAULT_PLANNING_POLICY.copy()
    normalized.update(_derive_planning_policy(user_text))
    normalized.update(planning_policy)
    for name, value in normalized.items():
        if not isinstance(value, bool):
            raise ValueError(f"planning_policy.{name} 必須是 boolean")
    return normalized


def _neutral_plan_answer() -> str:
    return (
        "我已完成抽象任務規劃；實際設備將由系統依能力、"
        "可達範圍與目前可用狀態自動分配。"
    )


def build_abstract_plan(
    user_text: str,
    tool_catalog: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    planning_policy: dict[str, Any] | None = None,
    llm_options: dict[str, Any] | None = None,
    planning_fact_resolver: Any = None,
) -> dict[str, Any]:
    """Build a validated abstract Plan without allocating concrete devices."""
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text 必須是非空字串")
    if tool_catalog is None:
        tool_catalog = get_tools_for_prompt()
    if not isinstance(tool_catalog, list) or not tool_catalog:
        raise ValueError("tool_catalog 必須是非空 list")
    for index, tool in enumerate(tool_catalog):
        if not isinstance(tool, dict):
            raise ValueError(f"tool_catalog[{index}] 必須是 dict")
        function_name = tool.get("api_function")
        if not isinstance(function_name, str) or not function_name.strip():
            raise ValueError(f"tool_catalog[{index}].api_function 格式錯誤")
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise ValueError("context 必須是 dict")
    if llm_options is None:
        llm_options = {}
    if not isinstance(llm_options, dict):
        raise ValueError("llm_options 必須是 dict")

    normalized_text = user_text.strip()
    policy = _normalize_planning_policy(planning_policy, normalized_text)
    stage1_options = llm_options.get("stage1") or {}
    stage2_options = llm_options.get("stage2") or {}
    api_key = llm_options.get("openai_api_key")

    try:
        planning_started = time.perf_counter()
        stage_started = time.perf_counter()
        actions = extract_atomic_actions(
            normalized_text,
            tool_catalog,
            context=deepcopy(context),
            allow_duplicated_subplan=policy["allow_duplicated_subplan"],
            provider=stage1_options.get("provider"),
            model=stage1_options.get("model"),
            api_key=api_key,
        )
        stage1_ms = (time.perf_counter() - stage_started) * 1000

        planning_task_facts = []
        planning_perception_calls = []
        planning_required_edges = []
        if planning_fact_resolver is not None:
            resolved_facts = planning_fact_resolver(
                deepcopy(actions),
                deepcopy(context),
            )
            if not isinstance(resolved_facts, dict):
                raise TypeError("planning_fact_resolver 必須回傳 dict")
            actions = deepcopy(resolved_facts.get("actions", actions))
            planning_task_facts = deepcopy(resolved_facts.get("task_facts", []))
            planning_perception_calls = deepcopy(
                resolved_facts.get("planning_perception_calls", [])
            )
            planning_required_edges = deepcopy(
                resolved_facts.get("required_edges", [])
            )

        stage_started = time.perf_counter()
        pair_relations = classify_action_relations(
            normalized_text,
            actions,
            provider=stage2_options.get("provider"),
            model=stage2_options.get("model"),
            api_key=api_key,
        )
        for edge in planning_required_edges:
            before, after = edge.get("before"), edge.get("after")
            for relation in pair_relations:
                pair = {relation.get("action_a"), relation.get("action_b")}
                if pair != {before, after}:
                    continue
                relation["relation"] = (
                    "A_BEFORE_B"
                    if relation.get("action_a") == before
                    else "B_BEFORE_A"
                )
                relation["strategy"] = "planning_fact"
                break
        stage2_ms = (time.perf_counter() - stage_started) * 1000

        stage_started = time.perf_counter()
        compiled = compile_pair_relations_to_plan(
            actions,
            pair_relations,
            tool_catalog,
            prefer_distinct_assignment=policy["prefer_distinct_assignment"],
        )
        compile_ms = (time.perf_counter() - stage_started) * 1000
        abstract_total_ms = (time.perf_counter() - planning_started) * 1000
    except (
        ActionExtractionError,
        RelationClassificationError,
        SemanticIRError,
        PlanValidationError,
    ) as exc:
        raise AbstractPlanningError(
            f"LLM 無法產生有效 abstract plan：{exc}"
        ) from exc
    except Exception as exc:
        raise AbstractPlanningError(
            "Abstract planner 發生未預期錯誤：\n"
            "stage: two_stage_semantic_pipeline\n"
            f"error_type: {type(exc).__name__}\n"
            f"message: {exc!r}\n"
            "traceback:\n"
            f"{traceback.format_exc()}"
        ) from exc

    pair_relations = compiled["pair_relations"]
    return {
        "user_text": normalized_text,
        "answer": _neutral_plan_answer(),
        "plan": compiled["plan"],
        "task_facts": planning_task_facts,
        "planning_perception_calls": planning_perception_calls,
        "planner_trace": {
            "providers": {
                "stage1": {
                    "provider": stage1_options.get("provider") or "environment default",
                    "model": stage1_options.get("model") or "environment default",
                },
                "stage2": {
                    "provider": stage2_options.get("provider") or "environment default",
                    "model": stage2_options.get("model") or "environment default",
                },
            },
            "actions": deepcopy(compiled["actions"]),
            "pair_relations": deepcopy(pair_relations),
            "batch_pairs": [
                {
                    "id": item["pair_id"],
                    "action_a": item["action_a"],
                    "action_b": item["action_b"],
                }
                for item in pair_relations
                if item.get("strategy") == "openai_batch"
            ],
            "batch_relations": [
                {
                    "id": item["pair_id"],
                    "temporal_relation": item["relation"],
                    "resource_relation": item.get("resource_relation", "NONE"),
                }
                for item in pair_relations
                if item.get("strategy") == "openai_batch"
            ],
            "pair_decisions": deepcopy(compiled["pair_decisions"]),
            "edges": deepcopy(compiled["edges"]),
            "parallel_pairs": deepcopy(compiled["parallel_pairs"]),
            "reduced_edges": deepcopy(compiled["reduced_edges"]),
            "resource_relations": deepcopy(compiled["resource_relations"]),
            "resource_decisions": deepcopy(compiled["resource_decisions"]),
            "role_mapping": deepcopy(compiled["role_mapping"]),
            "compiled_constraints": deepcopy(compiled["compiled_constraints"]),
            "compiled_assignment_preferences": deepcopy(
                compiled["compiled_assignment_preferences"]
            ),
        },
        "timings_ms": {
            "stage1_action_extraction": round(stage1_ms, 2),
            "stage2_relation_classification": round(stage2_ms, 2),
            "dag_compile": round(compile_ms, 2),
            "abstract_planner_total": round(abstract_total_ms, 2),
        },
    }


def prepare_plan(
    abstract_plan: Any,
    tool_catalog: dict[str, dict[str, Any]],
    device_registry: DeviceRegistry | None = None,
    *,
    probe_hardware: bool = True,
    device_statuses: dict[str, dict[str, Any]] | None = None,
    status_readers: dict[str, Any] | None = None,
    resource_holding_state: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Validate and resolve an abstract plan without executing it.

    ``device_statuses`` is primarily for tests or callers that already have a
    fresh status snapshot. When it is supplied, hardware probing is skipped.
    Setting ``probe_hardware`` to false without supplying statuses performs a
    catalog-only allocation and should only be used for offline inspection.
    """
    registry = device_registry or DeviceRegistry.from_arm_config()
    if not isinstance(registry, DeviceRegistry):
        raise TypeError("device_registry 必須是 DeviceRegistry")

    validated_plan = validate_plan_against_catalogs(
        abstract_plan,
        tool_catalog,
        registry,
    )

    if device_statuses is not None:
        statuses = deepcopy(device_statuses)
    elif probe_hardware:
        statuses = registry.probe_statuses(status_readers=status_readers)
    else:
        statuses = {}

    allocation = allocate_devices(validated_plan, registry, statuses)
    assignments = validate_assignments(
        validated_plan,
        allocation["assignments"],
        registry,
        statuses,
    )
    resolved_plan = resolve_assignments(validated_plan, assignments, registry)
    execution_preview = build_execution_preview(
        resolved_plan,
        failure_policy="stop_all",
        initial_holding=resource_holding_state,
    )

    return {
        "abstract_plan": deepcopy(abstract_plan),
        "validated_plan": validated_plan,
        "device_statuses": statuses,
        "allocation": allocation,
        "resolved_plan": resolved_plan,
        "execution_preview": execution_preview,
        "executable": False,
    }
