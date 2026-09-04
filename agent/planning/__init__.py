"""AI Agent 通用任務規劃元件。

此套件涵蓋 LLM 語意解析、確定性 Plan 編譯、設備分配與執行預覽；
不呼叫 executor，也不操作機器手臂、移動平台或 IO。
"""

from agent.planning.device_allocator import (
    AllocationError,
    AssignmentValidationError,
    allocate_devices,
    resolve_assignments,
    validate_assignments,
)
from agent.planning.device_registry import DeviceRegistry
from agent.planning.action_extractor import extract_atomic_actions
from agent.planning.action_validation import (
    SemanticIRError,
    validate_semantic_action_coverage,
)
from agent.planning.ir_compiler import (
    compile_pair_relations_to_plan,
)
from agent.planning.plan_pipeline import (
    AbstractPlanningError,
    build_abstract_plan,
    prepare_plan,
)
from agent.planning.plan_scheduler import (
    SchedulerError,
    build_execution_preview,
)
from agent.planning.plan_schema import (
    PlanValidationError,
    validate_plan,
    validate_plan_against_catalogs,
)
from agent.planning.relation_classifier import classify_action_relations
from agent.planning.perception_requirements import derive_perception_requirements
from agent.planning.target_grounding import resolve_scene_mentions

__all__ = [
    "AllocationError",
    "AssignmentValidationError",
    "AbstractPlanningError",
    "DeviceRegistry",
    "PlanValidationError",
    "SchedulerError",
    "SemanticIRError",
    "allocate_devices",
    "compile_pair_relations_to_plan",
    "classify_action_relations",
    "derive_perception_requirements",
    "build_execution_preview",
    "build_abstract_plan",
    "extract_atomic_actions",
    "prepare_plan",
    "resolve_assignments",
    "resolve_scene_mentions",
    "validate_assignments",
    "validate_plan",
    "validate_plan_against_catalogs",
    "validate_semantic_action_coverage",
]
