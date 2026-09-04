"""Stage 2 LLM adapter for temporal and resource pair semantics.

Temporal ordering and logical-resource relationships are independent axes.
The classifier never creates actions, roles, devices or formal constraints;
those remain deterministic compiler responsibilities.

Local Stage 2 intentionally uses absolute action IDs for temporal ordering to
avoid A/B positional bias. Its output is immediately converted back to the
existing canonical A_BEFORE_B / B_BEFORE_A / PARALLEL / NONE representation,
so OpenAI, compiler, trace consumers and downstream planning interfaces remain
unchanged.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from textwrap import dedent
from typing import Any

from agent.planning.structured_provider import (
    StructuredProviderError,
    chat_structured,
    resolve_provider,
)


RELATIONS = {"A_BEFORE_B", "B_BEFORE_A", "PARALLEL", "NONE"}
RESOURCE_RELATIONS = {
    "SAME_RESOURCE",
    "DISTINCT_RESOURCE",
    "PREFER_SAME_RESOURCE",
    "PREFER_DISTINCT_RESOURCE",
    "NONE",
}


# LOCAL provider prompt only.
# Do not ask the local model to emit A_BEFORE_B / B_BEFORE_A directly.
ABSTRACT_RELATION_CLASSIFICATION_PROMPT = """
You classify exactly one fixed pair of actions from the original user request.

For temporal ordering, do NOT answer with A_BEFORE_B or B_BEFORE_A.
Those orientation-relative labels are intentionally forbidden.

Return exactly four fields:
- temporal_type: ORDERED, PARALLEL, or NONE
- before_action: the actual earlier action ID, or NONE
- after_action: the actual later action ID, or NONE
- resource_relation: SAME_RESOURCE, DISTINCT_RESOURCE,
  PREFER_SAME_RESOURCE, PREFER_DISTINCT_RESOURCE, or NONE

Temporal definitions:
- ORDERED: one queried action must finish before the other can start.
  Use the actual action IDs supplied in this query for before_action and
  after_action.
- PARALLEL: the user explicitly says the two queried actions may or should
  start concurrently / independently without waiting.
  before_action=NONE and after_action=NONE.
- NONE: the user specifies neither an ordering dependency nor explicit
  concurrency for this queried pair.
  before_action=NONE and after_action=NONE.

Resource definitions:
- SAME_RESOURCE: the user explicitly requires the exact same assignable
  device/subsystem/resource for both actions.
- DISTINCT_RESOURCE: the user explicitly requires different assignable
  devices/subsystems/resources.
- PREFER_SAME_RESOURCE: the user explicitly prefers the exact same assignable
  device/subsystem/resource, but it is not required.
- PREFER_DISTINCT_RESOURCE: the user explicitly prefers different assignable
  devices/subsystems/resources, but fallback is allowed.
- NONE: no explicit assignable-resource relationship is specified.

SAME_RESOURCE refers to the exact same assignable device/subsystem. It does
NOT mean:
- the same physical parent robot,
- actions involving the same object or payload,
- actions belonging to the same task,
- an arm and a mobile base mounted on the same robot,
- actions that happen sequentially.

Example:
pick_object performed by an arm and move_amr_to performed by a mobile base
must use resource_relation=NONE unless the user explicitly specifies a
resource relationship between those two actions.

Important:
- The order in which the two actions are presented is NOT execution order.
- Action IDs are identifiers only. Determine semantic order only from the
  original user request.
- Repeated actions expanded for different members of one collection are
  semantically independent unless the original user request explicitly orders
  those members. Do NOT infer ORDERED merely because collection-member actions
  are listed sequentially.
- PARALLEL requires explicit concurrency wording. Mere absence of a dependency
  means NONE, not PARALLEL.
- Resource relations are symmetric; swapping presentation order must not
  change resource_relation.
- Never infer SAME_RESOURCE or DISTINCT_RESOURCE merely from temporal order,
  PARALLEL, action type, action similarity, action IDs, or list position.
- Do not create or modify actions, dependencies, roles, devices, constraints,
  assignments, step IDs, or explanations.
"""


# OPENAI provider prompt stays on the existing canonical relation format.
OPENAI_BATCH_RELATION_CLASSIFICATION_PROMPT = """
You classify two independent semantic axes for a Python-provided list of fixed
action pairs: temporal_relation and resource_relation. Return exactly one of
each for every provided pair ID. Do not create, remove,
rename, reorder, or reinterpret actions or pair IDs.

Relation definitions:
- A_BEFORE_B: Action A must finish before Action B can start.
- B_BEFORE_A: Action B must finish before Action A can start.
- PARALLEL: The user explicitly says the two actions should or may start
  concurrently or independently without waiting for each other.
- NONE: The user does not require the two actions to wait for each other and
  does not explicitly require them to start concurrently.

Resource relation definitions:
- SAME_RESOURCE: The user explicitly requires both actions to use the exact
  same assignable device/subsystem/resource, such as "the same hand" or
  "the same arm".
- DISTINCT_RESOURCE: The user explicitly requires different assignable
  devices/subsystems/resources, such as "the other hand" or
  "different devices".
- PREFER_SAME_RESOURCE: The user explicitly prefers the exact same assignable
  device/subsystem/resource, but it is not required.
- PREFER_DISTINCT_RESOURCE: The user explicitly prefers different assignable
  devices/subsystems/resources, but fallback is allowed.
- NONE: The user specifies no explicit assignable-resource relationship.

SAME_RESOURCE means the exact same assignable device/subsystem. It does NOT
mean the same physical parent robot, the same object/payload, the same task,
or an arm and mobile base mounted on the same robot.

Example: pick_object performed by a robot arm and move_amr_to performed by a
mobile base must use resource_relation=NONE unless the original user request
explicitly requires a resource relationship.

Temporal order, PARALLEL, action similarity, shared objects, action IDs, and
array position are not evidence of a resource relationship. Do not infer SAME
or DISTINCT unless the original user request explicitly expresses it.

Action IDs, pair IDs, and positions in the actions or pairs arrays are
identifiers only. They never imply execution order. Determine every relation
solely from the meaning of the original user request.

Repeated actions expanded for different members of one collection are
semantically independent unless the original user request explicitly orders
those members. Do not infer A_BEFORE_B/B_BEFORE_A merely because one member's
actions appear earlier in the array.

PARALLEL requires explicit concurrency wording in the original request.
If two actions simply have no required dependency, use NONE rather than
PARALLEL.

You may only fill the two relation enums for existing pair IDs. Never output
action keys, dependencies, edges, roles, step IDs, assignments, devices,
constraints, requirements, explanations, or extra fields.
"""


class RelationClassificationError(RuntimeError):
    """Stage 2 failed after bounded retries."""


def _stage2_chat_structured(
    messages,
    response_schema,
    *,
    provider=None,
    model=None,
    api_key=None,
):
    """Provider switch; callers share identical messages and JSON schema."""
    try:
        return chat_structured(
            stage="stage2",
            messages=messages,
            response_schema=response_schema,
            provider=provider,
            model=model,
            api_key=api_key,
        )
    except StructuredProviderError as exc:
        raise RelationClassificationError(str(exc)) from exc


def relation_classification_schema():
    """Legacy canonical single-pair schema retained for compatibility/tests."""
    return {
        "type": "object",
        "properties": {
            "temporal_relation": {
                "type": "string",
                "enum": sorted(RELATIONS),
            },
            "resource_relation": {
                "type": "string",
                "enum": sorted(RESOURCE_RELATIONS),
            },
        },
        "required": ["temporal_relation", "resource_relation"],
        "additionalProperties": False,
    }


def local_relation_classification_schema(action_a_key, action_b_key):
    """Local-only schema using absolute action IDs for temporal ordering."""
    action_ids = sorted({action_a_key, action_b_key})
    return {
        "type": "object",
        "properties": {
            "temporal_type": {
                "type": "string",
                "enum": ["NONE", "ORDERED", "PARALLEL"],
            },
            "before_action": {
                "type": "string",
                "enum": [*action_ids, "NONE"],
            },
            "after_action": {
                "type": "string",
                "enum": [*action_ids, "NONE"],
            },
            "resource_relation": {
                "type": "string",
                "enum": sorted(RESOURCE_RELATIONS),
            },
        },
        "required": [
            "temporal_type",
            "before_action",
            "after_action",
            "resource_relation",
        ],
        "additionalProperties": False,
    }


def batch_relation_classification_schema(pair_ids):
    return {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "minItems": len(pair_ids),
                "maxItems": len(pair_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": pair_ids},
                        "temporal_relation": {
                            "type": "string",
                            "enum": sorted(RELATIONS),
                        },
                        "resource_relation": {
                            "type": "string",
                            "enum": sorted(RESOURCE_RELATIONS),
                        },
                    },
                    "required": [
                        "id",
                        "temporal_relation",
                        "resource_relation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["relations"],
        "additionalProperties": False,
    }


def _format_action(action):
    return (
        f"{action['key']} = {action['function_name']}"
        f"({json.dumps(action['arguments'], ensure_ascii=False, sort_keys=True)})"
    )


def _edges_from_relations(pair_relations):
    edges = set()
    parallel = set()
    for item in pair_relations:
        left = item["action_a"]
        right = item["action_b"]
        relation = item["relation"]

        if relation == "A_BEFORE_B":
            edges.add((left, right))
        elif relation == "B_BEFORE_A":
            edges.add((right, left))
        elif relation == "PARALLEL":
            parallel.add(tuple(sorted((left, right))))

    return edges, parallel


def _resource_relation(item):
    """Read new resource semantics while accepting old temporal-only callers."""
    return item.get("resource_relation", "NONE")


def _compact_resource_text(user_text: str) -> str:
    """Normalize only for deterministic explicit resource-cue checks."""
    return re.sub(r"\s+", "", str(user_text or "").casefold())


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _gate_resource_relation(user_text: str, relation: str) -> str:
    """Require explicit user evidence before accepting a resource relation.

    Stage2 may reason about temporal semantics, but hard/soft resource binding
    is allowed only when the original user request explicitly mentions a
    same/different assignable resource. Shared objects, sequential execution,
    or membership in the same physical parent robot are not sufficient.
    """
    if relation not in RESOURCE_RELATIONS:
        raise ValueError(f"不支援 resource relation：{relation}")

    if relation == "NONE":
        return "NONE"

    text = _compact_resource_text(user_text)

    same_cues = (
        "同一隻手", "同一只手", "同一手",
        "同一支手臂", "同一隻手臂", "同一只手臂", "同一手臂", "同一個手臂",
        "同一設備", "同一個設備", "同一台設備",
        "同一資源", "同一個資源",
        "同一subsystem", "同一個subsystem", "同一子系統", "同一個子系統",
        "samehand", "samearm", "samedevice", "sameresource", "samesubsystem",
    )
    distinct_cues = (
        "另一隻手", "另一只手", "另一手",
        "另一支手臂", "另一隻手臂", "另一只手臂", "另一手臂",
        "不同手臂", "不同設備", "不同資源", "不同subsystem", "不同子系統",
        "兩隻手分工", "两只手分工", "不同手臂分工", "不同設備分工",
        "otherhand", "otherarm", "differentarms", "differentdevices",
        "differentresources", "differentsubsystems",
    )
    preference_cues = ("優先", "优先", "prefer", "preferred")

    has_same = _contains_any(text, same_cues)
    has_distinct = _contains_any(text, distinct_cues)
    has_preference = _contains_any(text, preference_cues)

    if relation == "SAME_RESOURCE":
        return relation if has_same else "NONE"
    if relation == "DISTINCT_RESOURCE":
        return relation if has_distinct else "NONE"
    if relation == "PREFER_SAME_RESOURCE":
        return relation if has_same and has_preference else "NONE"
    if relation == "PREFER_DISTINCT_RESOURCE":
        return relation if has_distinct and has_preference else "NONE"
    return "NONE"



def _compact_temporal_text(user_text: str) -> str:
    """Normalize only for deterministic temporal-cue checks."""
    return re.sub(r"\s+", "", str(user_text or "").casefold())


def _action_by_key(
    actions: list[dict[str, Any]],
    action_key: str,
) -> dict[str, Any] | None:
    for action in actions:
        if isinstance(action, dict) and action.get("key") == action_key:
            return action
    return None


def _primary_object_id(action: dict[str, Any] | None) -> str | None:
    """Return the manipulated object identity when the action has one.

    object_id is intentionally used here rather than every *_id field.
    destination/container/payload IDs describe different semantic roles and
    must not make two independent collection members look like the same object.
    """
    if not isinstance(action, dict):
        return None

    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        return None

    object_id = arguments.get("object_id")
    if isinstance(object_id, str) and object_id.strip():
        return object_id.strip()

    return None


def _has_collection_scope(user_text: str) -> bool:
    """Detect generic all/every collection scope; no domain vocabulary."""
    text = _compact_temporal_text(user_text)
    collection_cues = (
        "所有",
        "全部",
        "每個",
        "每一个",
        "每一個",
        "all",
        "every",
        "each",
    )
    return _contains_any(text, collection_cues)


def _has_explicit_parallel_cue(user_text: str) -> bool:
    """PARALLEL is accepted only when concurrency is explicitly requested."""
    text = _compact_temporal_text(user_text)
    parallel_cues = (
        "同時",
        "同步",
        "並行",
        "并行",
        "平行執行",
        "平行执行",
        "concurrently",
        "simultaneously",
        "inparallel",
        "atthesametime",
    )
    return _contains_any(text, parallel_cues)


def _gate_temporal_relation(
    user_text: str,
    actions: list[dict[str, Any]],
    action_a_key: str,
    action_b_key: str,
    relation: str,
) -> tuple[str, str]:
    """Gate raw LLM temporal semantics before they can become DAG edges.

    This gate is intentionally domain-neutral.

    1. PARALLEL is hard semantics, so it requires an explicit concurrency cue.
       Mere independence is NONE.
    2. When Stage1 expands an all/every collection, actions operating on
       different object_id members are independent across members unless a
       future richer grounding layer supplies an explicit member-order fact.
       Their apparent list order must not become a DAG dependency.
    3. All other temporal relations remain LLM-proposed semantics and continue
       through the existing cycle/consistency validators.

    Resource occupancy is deliberately NOT encoded here. A capacity-1 gripper
    is handled later by the planning scheduler after concrete allocation.
    """
    if relation not in RELATIONS:
        raise ValueError(f"不支援 temporal relation：{relation}")

    if relation == "NONE":
        return "NONE", "raw_none"

    if relation == "PARALLEL":
        if _has_explicit_parallel_cue(user_text):
            return "PARALLEL", "explicit_parallel_cue"
        return "NONE", "parallel_without_explicit_cue"

    action_a = _action_by_key(actions, action_a_key)
    action_b = _action_by_key(actions, action_b_key)

    object_a = _primary_object_id(action_a)
    object_b = _primary_object_id(action_b)

    if (
        _has_collection_scope(user_text)
        and object_a is not None
        and object_b is not None
        and object_a != object_b
    ):
        return "NONE", "independent_collection_members"

    return relation, "accepted_raw_temporal"


def normalize_resource_semantic(action_a_key, action_b_key, relation):
    """Normalize a symmetric resource enum into an unordered semantic result."""
    if relation not in RESOURCE_RELATIONS:
        raise ValueError(f"不支援 resource relation：{relation}")

    if relation == "NONE":
        return {"type": "none"}

    return {
        "type": relation.lower(),
        "actions": sorted([action_a_key, action_b_key]),
    }


def _find_path(start, target, adjacency):
    pending = [(start, [])]
    visited = set()

    while pending:
        current, path = pending.pop()

        if current == target:
            return path

        if current in visited:
            continue

        visited.add(current)

        for child in adjacency.get(current, set()):
            pending.append((child, [*path, (current, child)]))

    return None


def validate_pair_relations(actions, pair_relations):
    """Reject cycle and transitive parallel conflicts before compilation."""
    keys = [action["key"] for action in actions]
    key_set = set(keys)
    resource_by_pair = {}

    for index, item in enumerate(pair_relations):
        if not isinstance(item, dict):
            raise RelationClassificationError(
                f"pair_relations[{index}] 必須是 object"
            )

        left = item.get("action_a")
        right = item.get("action_b")

        if left not in key_set or right not in key_set:
            raise RelationClassificationError(
                f"pair_relations[{index}] 引用不存在 action：{left}, {right}"
            )

        if left == right:
            raise RelationClassificationError(
                f"pair_relations[{index}] 不可引用同一 action 兩次：{left}"
            )

        relation = item.get("relation")
        if relation not in RELATIONS:
            raise RelationClassificationError(
                f"pair_relations[{index}] temporal relation 無效：{relation}"
            )

        resource = _resource_relation(item)
        if resource not in RESOURCE_RELATIONS:
            raise RelationClassificationError(
                f"pair_relations[{index}] resource relation 無效：{resource}"
            )

        pair = tuple(sorted((left, right)))
        previous = resource_by_pair.get(pair)

        if (
            previous is not None
            and previous != resource
            and {previous, resource}.issubset(
                {"SAME_RESOURCE", "DISTINCT_RESOURCE"}
            )
        ):
            raise RelationClassificationError(
                f"resource relations 對同一 action pair 互相矛盾：{pair}"
            )

        if resource != "NONE":
            resource_by_pair[pair] = resource

    edges, parallel_pairs = _edges_from_relations(pair_relations)
    adjacency = {key: set() for key in keys}

    for source, target in edges:
        adjacency[source].add(target)

    for key in keys:
        for child in adjacency[key]:
            if _find_path(child, key, adjacency) is not None:
                raise RelationClassificationError(
                    f"relation classifications 形成 cycle，涉及 {key} -> {child}"
                )

    for left, right in parallel_pairs:
        forward = _find_path(left, right, adjacency)
        backward = _find_path(right, left, adjacency)

        if forward is not None or backward is not None:
            path = forward if forward is not None else backward
            raise RelationClassificationError(
                f"PARALLEL conflict：{left}, {right} "
                f"同時存在 dependency path {path}"
            )


def _canonicalize_local_temporal(
    parsed,
    action_a_key,
    action_b_key,
):
    """Convert local absolute-ID output back to the canonical relation enum."""
    temporal_type = parsed["temporal_type"]
    before_action = parsed["before_action"]
    after_action = parsed["after_action"]
    valid_ids = {action_a_key, action_b_key}

    if temporal_type == "ORDERED":
        if before_action not in valid_ids or after_action not in valid_ids:
            raise ValueError(
                "ORDERED 的 before_action/after_action "
                "必須引用目前 query 的兩個 action"
            )

        if before_action == after_action:
            raise ValueError(
                "ORDERED 的 before_action/after_action 不可相同"
            )

        if {before_action, after_action} != valid_ids:
            raise ValueError(
                "ORDERED 必須完整引用目前 query 的兩個 action"
            )

        if (
            before_action == action_a_key
            and after_action == action_b_key
        ):
            return "A_BEFORE_B"

        if (
            before_action == action_b_key
            and after_action == action_a_key
        ):
            return "B_BEFORE_A"

        raise ValueError(
            "無法將 local ORDERED 結果轉成 canonical relation"
        )

    if temporal_type in {"PARALLEL", "NONE"}:
        if before_action != "NONE" or after_action != "NONE":
            raise ValueError(
                f"{temporal_type} 時 before_action/after_action "
                "必須都是 NONE"
            )
        return temporal_type

    raise ValueError(
        f"不支援 local temporal_type：{temporal_type}"
    )


def _classify_pair(
    user_text,
    actions,
    action_a,
    action_b,
    repair_context=None,
    *,
    provider=None,
    model=None,
    api_key=None,
):
    """LOCAL single-pair classification using absolute action IDs."""
    actions_text = "\n".join(
        _format_action(action)
        for action in actions
    )

    action_a_key = action_a["key"]
    action_b_key = action_b["key"]

    prompt = dedent(f"""
        Original user request:
        {user_text}

        Fixed actions (reference only; do not modify):
        {actions_text}

        Query pair:

        presented_action_1_id = {action_a_key}
        presented_action_1 = {_format_action(action_a)}

        presented_action_2_id = {action_b_key}
        presented_action_2 = {_format_action(action_b)}

        IMPORTANT:
        presented_action_1 / presented_action_2 are presentation positions only.
        They do NOT mean earlier/later.

        If one queried action must finish before the other:
        temporal_type = ORDERED
        before_action = actual earlier action ID
        after_action = actual later action ID

        If the two queried actions explicitly may/should start concurrently:
        temporal_type = PARALLEL
        before_action = NONE
        after_action = NONE

        If neither ordering nor explicit concurrency is specified:
        temporal_type = NONE
        before_action = NONE
        after_action = NONE

        Classify resource_relation independently.

        Previous graph/orientation conflict:
        {repair_context or 'none'}
    """).strip()

    messages = [
        {
            "role": "system",
            "content": ABSTRACT_RELATION_CLASSIFICATION_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    failures = []
    schema = local_relation_classification_schema(
        action_a_key,
        action_b_key,
    )

    for attempt in range(3):
        message = _stage2_chat_structured(
            messages=messages,
            response_schema=schema,
            provider=provider,
            model=model,
            api_key=api_key,
        )

        content = (
            message.get("content")
            if isinstance(message, dict)
            else None
        )

        try:
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "Local Stage 2 缺少 content"
                )

            parsed = json.loads(content)

            expected_fields = {
                "temporal_type",
                "before_action",
                "after_action",
                "resource_relation",
            }

            if (
                not isinstance(parsed, dict)
                or set(parsed) != expected_fields
            ):
                raise ValueError(
                    "Local Stage 2 只能輸出 "
                    "temporal_type/before_action/"
                    "after_action/resource_relation"
                )

            resource = parsed["resource_relation"]

            if resource not in RESOURCE_RELATIONS:
                raise ValueError(
                    f"不支援 resource relation：{resource}"
                )

            temporal = _canonicalize_local_temporal(
                parsed,
                action_a_key,
                action_b_key,
            )

            # Important:
            # downstream still receives the old canonical contract.
            return {
                "temporal_relation": temporal,
                "resource_relation": resource,
                "local_temporal_type": parsed["temporal_type"],
                "local_before_action": parsed["before_action"],
                "local_after_action": parsed["after_action"],
            }

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            failures.append(
                (
                    attempt + 1,
                    str(exc),
                    content,
                )
            )

            messages = [
                *messages[:2],
                {
                    "role": "user",
                    "content": (
                        "The previous output was invalid. "
                        "Return exactly temporal_type, before_action, "
                        "after_action, resource_relation. "
                        "For ORDERED, before_action and after_action "
                        "must be the two actual query action IDs "
                        f"({action_a_key}, {action_b_key}) in semantic "
                        "execution order. For PARALLEL/NONE both "
                        "action fields must be NONE."
                    ),
                },
            ]

    details = "\n\n".join(
        (
            f"attempt {number}: {error}\n"
            f"raw_content:\n{raw_content}"
        )
        for number, error, raw_content in failures
    )

    raise RelationClassificationError(
        f"無法分類 {action_a_key}, {action_b_key}\n{details}"
    )


def normalize_relation_semantic(
    action_a_key,
    action_b_key,
    relation,
):
    """Normalize orientation-relative enum into an absolute semantic result."""
    if relation == "A_BEFORE_B":
        return {
            "type": "edge",
            "source": action_a_key,
            "target": action_b_key,
        }

    if relation == "B_BEFORE_A":
        return {
            "type": "edge",
            "source": action_b_key,
            "target": action_a_key,
        }

    if relation == "PARALLEL":
        return {
            "type": "parallel",
            "actions": sorted(
                [action_a_key, action_b_key]
            ),
        }

    if relation == "NONE":
        return {
            "type": "none",
        }

    raise ValueError(
        f"不支援 relation：{relation}"
    )


def _build_batch_pairs(actions):
    return [
        {
            "id": f"p{index}",
            "action_a": action_a["key"],
            "action_b": action_b["key"],
        }
        for index, (action_a, action_b)
        in enumerate(
            combinations(actions, 2),
            start=1,
        )
    ]


def _validate_batch_output(parsed, batch_pairs):
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"relations"}
    ):
        raise RelationClassificationError(
            "OpenAI batch Stage 2 只能輸出 relations"
        )

    raw_relations = parsed["relations"]

    if not isinstance(raw_relations, list):
        raise RelationClassificationError(
            "OpenAI batch relations 必須是 array"
        )

    expected_ids = [
        item["id"]
        for item in batch_pairs
    ]

    received_ids = []
    relation_by_id = {}

    for index, item in enumerate(raw_relations):
        expected_fields = {
            "id",
            "temporal_relation",
            "resource_relation",
        }

        if (
            not isinstance(item, dict)
            or set(item) != expected_fields
        ):
            raise RelationClassificationError(
                "OpenAI batch relations["
                f"{index}] 只能包含 "
                "id/temporal_relation/resource_relation"
            )

        pair_id = item["id"]
        temporal = item["temporal_relation"]
        resource = item["resource_relation"]

        if pair_id not in expected_ids:
            raise RelationClassificationError(
                f"OpenAI batch 多出未知 pair：{pair_id}"
            )

        if pair_id in relation_by_id:
            raise RelationClassificationError(
                f"OpenAI batch 重複 pair：{pair_id}"
            )

        if temporal not in RELATIONS:
            raise RelationClassificationError(
                f"OpenAI batch pair {pair_id} "
                f"temporal relation 無效：{temporal}"
            )

        if resource not in RESOURCE_RELATIONS:
            raise RelationClassificationError(
                f"OpenAI batch pair {pair_id} "
                f"resource relation 無效：{resource}"
            )

        received_ids.append(pair_id)

        relation_by_id[pair_id] = {
            "temporal_relation": temporal,
            "resource_relation": resource,
        }

    missing = [
        pair_id
        for pair_id in expected_ids
        if pair_id not in relation_by_id
    ]

    if missing:
        raise RelationClassificationError(
            f"OpenAI batch 漏掉 pairs：{missing}"
        )

    if len(received_ids) != len(expected_ids):
        raise RelationClassificationError(
            "OpenAI batch relations 數量錯誤"
        )

    return relation_by_id


def _classify_openai_batch(
    user_text,
    actions,
    *,
    provider,
    model=None,
    api_key=None,
    max_attempts=2,
):
    batch_pairs = _build_batch_pairs(actions)

    # One action means zero unordered pairs.
    if not batch_pairs:
        return []

    pair_ids = [
        item["id"]
        for item in batch_pairs
    ]

    user_prompt = dedent(f"""
        Original user request:
        {user_text}

        Fixed actions:
        {json.dumps(actions, ensure_ascii=False, indent=2)}

        Python-generated unordered pairs:
        {json.dumps(batch_pairs, ensure_ascii=False, indent=2)}

        Classify every pair exactly once. Fill temporal_relation using only
        A_BEFORE_B, B_BEFORE_A, PARALLEL, or NONE, and resource_relation using
        only SAME_RESOURCE, DISTINCT_RESOURCE, PREFER_SAME_RESOURCE,
        PREFER_DISTINCT_RESOURCE, or NONE. Array order and identifier numbering
        are not temporal or resource evidence.
    """).strip()

    base_messages = [
        {
            "role": "system",
            "content": OPENAI_BATCH_RELATION_CLASSIFICATION_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    messages = list(base_messages)
    failures = []

    for attempt in range(1, max_attempts + 1):
        message = _stage2_chat_structured(
            messages,
            batch_relation_classification_schema(
                pair_ids
            ),
            provider=provider,
            model=model,
            api_key=api_key,
        )

        content = (
            message.get("content")
            if isinstance(message, dict)
            else None
        )

        try:
            if (
                not isinstance(content, str)
                or not content.strip()
            ):
                raise RelationClassificationError(
                    "OpenAI batch Stage 2 缺少 content"
                )

            parsed = json.loads(content)

            relation_by_id = _validate_batch_output(
                parsed,
                batch_pairs,
            )

            pair_relations = []

            for pair in batch_pairs:
                classified = relation_by_id[
                    pair["id"]
                ]

                raw_relation = classified[
                    "temporal_relation"
                ]

                relation, temporal_gate_reason = (
                    _gate_temporal_relation(
                        user_text,
                        actions,
                        pair["action_a"],
                        pair["action_b"],
                        raw_relation,
                    )
                )

                raw_resource_relation = classified[
                    "resource_relation"
                ]

                resource_relation = _gate_resource_relation(
                    user_text,
                    raw_resource_relation,
                )

                semantic = normalize_relation_semantic(
                    pair["action_a"],
                    pair["action_b"],
                    relation,
                )

                resource_semantic = (
                    normalize_resource_semantic(
                        pair["action_a"],
                        pair["action_b"],
                        resource_relation,
                    )
                )

                final_edge = None

                if semantic["type"] == "edge":
                    final_edge = [
                        semantic["source"],
                        semantic["target"],
                    ]

                pair_relations.append({
                    "strategy": "openai_batch",
                    "pair_id": pair["id"],
                    "original_pair": [
                        pair["action_a"],
                        pair["action_b"],
                    ],
                    "provider": "openai",
                    "action_a": pair["action_a"],
                    "action_b": pair["action_b"],
                    "query_action_a": pair["action_a"],
                    "query_action_b": pair["action_b"],
                    "raw_llm_relation": raw_relation,
                    "temporal_gate_reason": temporal_gate_reason,
                    "raw_llm_resource_relation": raw_resource_relation,
                    "normalized_semantic": semantic,
                    "normalized_resource_semantic": resource_semantic,
                    "relation": relation,
                    "resource_relation": resource_relation,
                    "final_edge": final_edge,
                    "parallel": (
                        semantic["type"] == "parallel"
                    ),
                })

            validate_pair_relations(
                actions,
                pair_relations,
            )

            return pair_relations

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            RelationClassificationError,
        ) as exc:
            failures.append({
                "attempt": attempt,
                "error": str(exc),
                "raw_content": content,
            })

            messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": (
                        "The previous complete batch was invalid: "
                        f"{exc}. Reclassify the entire fixed pairs "
                        "list. Return every existing pair ID exactly "
                        "once with both relation enums and no other "
                        "fields. Do not create a cycle or combine "
                        "PARALLEL with a dependency path."
                    ),
                },
            ]

    raise RelationClassificationError(
        "OpenAI batch Stage 2 兩次皆無效："
        f"{json.dumps(failures, ensure_ascii=False)}"
    )


def _classify_with_orientation_check(
    user_text,
    actions,
    action_a,
    action_b,
    repair_context=None,
    *,
    provider=None,
    model=None,
    api_key=None,
):
    """Require identical absolute semantics after reversing query orientation."""
    last = None
    local_repair = repair_context

    for _ in range(2):
        forward = _classify_pair(
            user_text,
            actions,
            action_a,
            action_b,
            repair_context=local_repair,
            provider=provider,
            model=model,
            api_key=api_key,
        )

        reverse = _classify_pair(
            user_text,
            actions,
            action_b,
            action_a,
            repair_context=local_repair,
            provider=provider,
            model=model,
            api_key=api_key,
        )

        forward_semantic = normalize_relation_semantic(
            action_a["key"],
            action_b["key"],
            forward["temporal_relation"],
        )

        reverse_semantic = normalize_relation_semantic(
            action_b["key"],
            action_a["key"],
            reverse["temporal_relation"],
        )

        forward = dict(forward)
        reverse = dict(reverse)

        forward_raw_resource = forward["resource_relation"]
        reverse_raw_resource = reverse["resource_relation"]

        forward["raw_resource_relation"] = forward_raw_resource
        reverse["raw_resource_relation"] = reverse_raw_resource

        forward["resource_relation"] = _gate_resource_relation(
            user_text,
            forward_raw_resource,
        )
        reverse["resource_relation"] = _gate_resource_relation(
            user_text,
            reverse_raw_resource,
        )

        forward_resource = normalize_resource_semantic(
            action_a["key"],
            action_b["key"],
            forward["resource_relation"],
        )

        reverse_resource = normalize_resource_semantic(
            action_b["key"],
            action_a["key"],
            reverse["resource_relation"],
        )

        if (
            reverse_semantic == forward_semantic
            and reverse_resource == forward_resource
        ):
            return (
                forward,
                reverse,
                forward_semantic,
                forward_resource,
            )

        last = {
            "forward_raw": forward,
            "forward_normalized": forward_semantic,
            "forward_resource_normalized": forward_resource,
            "reverse_raw": reverse,
            "reverse_normalized": reverse_semantic,
            "reverse_resource_normalized": reverse_resource,
        }

        local_repair = (
            "Orientation consistency failed. "
            "The two queried actions were presented in reverse order, "
            "but the absolute semantic result changed. "
            "Use the actual action IDs in the original user request. "
            f"First result={forward_semantic}; "
            f"reversed result={reverse_semantic}."
        )

    raise RelationClassificationError(
        "Stage 2 orientation consistency 失敗："
        f"query=({action_a['key']}, {action_b['key']}), "
        f"results={last}"
    )


def classify_action_relations(
    user_text: str,
    actions: list[dict[str, Any]],
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
):
    """Classify every unordered pair; retry Stage 2 without rerunning Stage 1."""
    selected_provider = resolve_provider(
        "stage2",
        provider,
    )

    if selected_provider == "openai":
        return _classify_openai_batch(
            user_text,
            actions,
            provider=selected_provider,
            model=model,
            api_key=api_key,
        )

    # Zero or one action means no pair to classify.
    if len(actions) < 2:
        return []

    repair_context = None
    last_relations = None

    for _ in range(2):
        pair_relations = []

        for pair_index, (
            original_a,
            original_b,
        ) in enumerate(
            combinations(actions, 2)
        ):
            # Keep the existing deterministic orientation alternation.
            # The local model now outputs absolute action IDs, so this should
            # no longer create A/B label bias.
            if pair_index % 2:
                action_a = original_b
                action_b = original_a
            else:
                action_a = original_a
                action_b = original_b

            (
                classified,
                reverse_check,
                normalized_semantic,
                normalized_resource,
            ) = _classify_with_orientation_check(
                user_text,
                actions,
                action_a,
                action_b,
                repair_context=repair_context,
                provider=provider,
                model=model,
                api_key=api_key,
            )

            raw_relation = classified[
                "temporal_relation"
            ]

            relation, temporal_gate_reason = (
                _gate_temporal_relation(
                    user_text,
                    actions,
                    action_a["key"],
                    action_b["key"],
                    raw_relation,
                )
            )

            normalized_semantic = normalize_relation_semantic(
                action_a["key"],
                action_b["key"],
                relation,
            )

            resource_relation = classified[
                "resource_relation"
            ]

            final_edge = None

            if normalized_semantic["type"] == "edge":
                final_edge = [
                    normalized_semantic["source"],
                    normalized_semantic["target"],
                ]

            pair_relations.append({
                "original_pair": [
                    original_a["key"],
                    original_b["key"],
                ],
                "provider": resolve_provider(
                    "stage2",
                    provider,
                ),
                "action_a": action_a["key"],
                "action_b": action_b["key"],
                "query_action_a": action_a["key"],
                "query_action_b": action_b["key"],
                "raw_llm_relation": raw_relation,
                "temporal_gate_reason": temporal_gate_reason,
                "raw_llm_resource_relation": classified.get(
                    "raw_resource_relation",
                    resource_relation,
                ),

                # Extra local diagnostics only.
                # Existing downstream consumers can ignore these.
                "local_temporal_type": classified.get(
                    "local_temporal_type"
                ),
                "local_before_action": classified.get(
                    "local_before_action"
                ),
                "local_after_action": classified.get(
                    "local_after_action"
                ),

                "reverse_check_relation": reverse_check[
                    "temporal_relation"
                ],
                "reverse_check_resource_relation": (
                    reverse_check[
                        "resource_relation"
                    ]
                ),
                "reverse_raw_llm_resource_relation": (
                    reverse_check.get(
                        "raw_resource_relation",
                        reverse_check["resource_relation"],
                    )
                ),
                "reverse_local_temporal_type": (
                    reverse_check.get(
                        "local_temporal_type"
                    )
                ),
                "reverse_local_before_action": (
                    reverse_check.get(
                        "local_before_action"
                    )
                ),
                "reverse_local_after_action": (
                    reverse_check.get(
                        "local_after_action"
                    )
                ),

                "normalized_semantic": (
                    normalized_semantic
                ),
                "normalized_resource_semantic": (
                    normalized_resource
                ),
                "relation": relation,
                "resource_relation": resource_relation,
                "final_edge": final_edge,
                "parallel": (
                    normalized_semantic["type"]
                    == "parallel"
                ),
            })

        try:
            validate_pair_relations(
                actions,
                pair_relations,
            )
            return pair_relations

        except RelationClassificationError as exc:
            repair_context = (
                f"{exc}. 重新判斷每一 pair；"
                "不可同時宣稱 PARALLEL 與任何 direct/"
                "transitive ordering，也不可形成 cycle。"
            )
            last_relations = pair_relations

    raise RelationClassificationError(
        "Stage 2 relations 仍有 semantic conflict："
        f"{repair_context}; "
        f"last_relations={last_relations}"
    )
