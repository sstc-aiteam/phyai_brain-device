"""Deterministically compile semantic action IR into the formal Plan schema.

No language interpretation or concrete device allocation occurs here. Formal
step/role IDs, dependencies and soft branch preferences are produced solely
from validated IR structure.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.planning.plan_schema import validate_plan
from agent.planning.relation_classifier import validate_pair_relations
from agent.planning.action_validation import SemanticIRError


def _path_exists(source, target, edges):
    adjacency = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
    pending = list(adjacency.get(source, set()))
    visited = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency.get(current, set()))
    return False


def _transitive_reduction(edges):
    reduced = set(edges)
    for edge in sorted(edges):
        candidate = set(reduced)
        candidate.remove(edge)
        if _path_exists(edge[0], edge[1], candidate):
            reduced.remove(edge)
    return reduced


def _pair_edges(pair_relations):
    edges = set()
    parallel = set()
    for item in pair_relations:
        left, right = item["action_a"], item["action_b"]
        if item["relation"] == "A_BEFORE_B":
            edges.add((left, right))
        elif item["relation"] == "B_BEFORE_A":
            edges.add((right, left))
        elif item["relation"] == "PARALLEL":
            parallel.add(tuple(sorted((left, right))))
    return edges, parallel



def _deterministic_manipulation_edges(actions):
    """Derive hard pick->place edges from object acquire/release semantics.

    The LLM is not authoritative for this invariant:
        pick_object(X) must precede the matching place_object(X).

    A place for an object that was already held at task start simply has no
    matching pick in this action list and therefore receives no synthetic edge
    here; Stage 1 validation is responsible for deciding whether such a place
    is legal.
    """
    acquired_by = {}
    edges = set()
    decisions = []

    for action in actions:
        function_name = action.get("function_name")
        arguments = action.get("arguments") or {}
        object_id = arguments.get("object_id")

        if not isinstance(object_id, str) or not object_id:
            continue

        if function_name == "pick_object":
            acquired_by[object_id] = action["key"]
            continue

        if function_name != "place_object":
            continue

        acquire_key = acquired_by.pop(object_id, None)
        if acquire_key is None:
            continue

        release_key = action["key"]
        edges.add((acquire_key, release_key))
        decisions.append({
            "object_id": object_id,
            "acquire_action": acquire_key,
            "release_action": release_key,
            "edge": [acquire_key, release_key],
            "source": "object_manipulation_invariant",
        })

    return edges, decisions


def _merge_temporal_edges(pair_edges, parallel_pairs, deterministic_edges):
    """Merge LLM semantics with hard deterministic manipulation invariants.

    For the exact same action pair, deterministic acquire->release semantics
    override a contradictory reverse edge or PARALLEL classification.

    Cross-pair LLM relations are left untouched. If those relations create a
    longer contradictory path, compilation fails rather than silently inventing
    another ordering.
    """
    edges = set(pair_edges)
    parallel = set(parallel_pairs)

    for source, target in deterministic_edges:
        edges.discard((target, source))
        parallel.discard(tuple(sorted((source, target))))
        edges.add((source, target))

    return edges, parallel


def _validate_compiled_edges(action_keys, edges):
    """Reject cycles after deterministic and LLM temporal semantics are merged."""
    key_set = set(action_keys)

    for source, target in edges:
        if source not in key_set or target not in key_set:
            raise SemanticIRError(
                "UNKNOWN_ACTION_REFERENCE",
                f"compiled edge 引用不存在 action：{source} -> {target}",
                "relations",
            )
        if source == target:
            raise SemanticIRError(
                "SELF_DEPENDENCY",
                f"compiled edge 不可自我依賴：{source}",
                "relations",
            )

    for source, target in edges:
        candidate = set(edges)
        candidate.discard((source, target))
        if _path_exists(target, source, candidate):
            raise SemanticIRError(
                "DEPENDENCY_CYCLE",
                "LLM temporal relations 與 deterministic manipulation "
                f"invariant 形成 cycle：{source} -> {target}",
                "relations",
            )


def _action_family(function_name):
    if function_name.startswith(("open_", "close_")):
        return "resource", function_name.split("_", 1)[1]
    if function_name.startswith("move_arm_"):
        return "motion", "arm"
    return "action", function_name


def _deterministic_action_groups(actions, edges, separation_pairs=None):
    """Group existing families while honoring explicit resource separation.

    ``separation_pairs`` contains hard or preferred distinct-resource action
    pairs.  It only prevents an otherwise deterministic merge; it never merges
    actions merely because the language requested the same concrete resource.
    """
    separation_pairs = set(separation_pairs or ())
    ancestors = {action["key"]: set() for action in actions}
    changed = True
    while changed:
        changed = False
        for source, target in edges:
            expanded = {source, *ancestors[source]}
            if not expanded.issubset(ancestors[target]):
                ancestors[target].update(expanded)
                changed = True

    incoming = {action["key"]: set() for action in actions}
    for source, target in edges:
        incoming[target].add(source)
    ordered_actions = []
    processed = set()
    while len(ordered_actions) < len(actions):
        ready = [
            action for action in actions
            if action["key"] not in processed
            and incoming[action["key"]].issubset(processed)
        ]
        if not ready:
            raise SemanticIRError(
                "DEPENDENCY_CYCLE",
                "無法建立 action 拓撲順序；pair relations 可能形成 cycle",
                "relations",
            )
        for action in ready:
            ordered_actions.append(action)
            processed.add(action["key"])

    role_by_action = {}
    role_templates = []
    role_members = {}
    resource_roles = {}

    def can_join(role_index, action_key):
        return not any(
            tuple(sorted((member, action_key))) in separation_pairs
            for member in role_members.get(role_index, set())
        )

    for action in ordered_actions:
        key = action["key"]
        family_type, family_key = _action_family(action["function_name"])
        role_index = None
        if family_type == "resource":
            role_index = resource_roles.get(family_key)
            if role_index is not None and not can_join(role_index, key):
                role_index = None
        elif family_type == "motion":
            for previous in ordered_actions:
                if previous["key"] not in ancestors[key]:
                    continue
                if previous["key"] not in role_by_action:
                    continue
                if _action_family(previous["function_name"])[0] == "motion":
                    candidate = role_by_action[previous["key"]]
                    if can_join(candidate, key):
                        role_index = candidate
        if role_index is None:
            role_index = len(role_templates)
            role_templates.append(action)
            role_members[role_index] = set()
            if family_type == "resource":
                resource_roles[family_key] = role_index
        role_by_action[key] = role_index
        role_members[role_index].add(key)
    return role_by_action, role_templates


def _resource_separation_pairs(pair_relations):
    return {
        tuple(sorted((item["action_a"], item["action_b"])))
        for item in pair_relations
        if item.get("resource_relation", "NONE") in {
            "DISTINCT_RESOURCE",
            "PREFER_DISTINCT_RESOURCE",
        }
    }


def _compile_resource_assignments(pair_relations, role_by_action):
    """Translate action-level resource semantics into role-level Plan data."""
    hard_same = set()
    hard_distinct = set()
    soft_same = set()
    soft_distinct = set()
    related_role_pairs = set()
    decisions = []

    for item in pair_relations:
        resource = item.get("resource_relation", "NONE")
        if resource == "NONE":
            continue
        action_pair = [item["action_a"], item["action_b"]]
        left_role = f"role_{role_by_action[item['action_a']] + 1}"
        right_role = f"role_{role_by_action[item['action_b']] + 1}"
        roles = tuple(sorted((left_role, right_role)))
        same_role = left_role == right_role
        decision = {
            "actions": action_pair,
            "resource_relation": resource,
            "roles": [left_role, right_role],
            "compiled": None,
        }

        if same_role:
            if resource == "DISTINCT_RESOURCE":
                raise SemanticIRError(
                    "RESOURCE_GROUPING_CONFLICT",
                    "explicit DISTINCT_RESOURCE actions 被分到同一 role："
                    f"{action_pair} -> {left_role}",
                    "resource_relations",
                )
            if resource == "PREFER_DISTINCT_RESOURCE":
                raise SemanticIRError(
                    "RESOURCE_GROUPING_CONFLICT",
                    "PREFER_DISTINCT_RESOURCE actions 無法安全拆分 role："
                    f"{action_pair} -> {left_role}",
                    "resource_relations",
                )
            decision["compiled"] = "already_satisfied_by_same_role"
            decisions.append(decision)
            continue

        related_role_pairs.add(roles)
        if resource == "SAME_RESOURCE":
            hard_same.add(roles)
            decision["compiled"] = "same_assignment"
        elif resource == "DISTINCT_RESOURCE":
            hard_distinct.add(roles)
            decision["compiled"] = "distinct_assignment"
        elif resource == "PREFER_SAME_RESOURCE":
            soft_same.add(roles)
            decision["compiled"] = "prefer_same_assignment"
        elif resource == "PREFER_DISTINCT_RESOURCE":
            soft_distinct.add(roles)
            decision["compiled"] = "prefer_distinct_assignment"
        decisions.append(decision)

    # Detect transitive hard contradictions before allocator search.  For
    # example same(r1,r2), same(r2,r3), distinct(r1,r3) is already impossible.
    role_ids = {
        role
        for pair in hard_same | hard_distinct
        for role in pair
    }
    parent = {role: role for role in role_ids}

    def find(role):
        while parent[role] != role:
            parent[role] = parent[parent[role]]
            role = parent[role]
        return role

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in hard_same:
        union(left, right)
    for left, right in hard_distinct:
        if find(left) == find(right):
            raise SemanticIRError(
                "RESOURCE_CONSTRAINT_CONFLICT",
                "hard SAME_RESOURCE / DISTINCT_RESOURCE constraints 互相矛盾："
                f"{left}, {right}",
                "resource_relations",
            )

    constraints = [
        {
            "type": "same_assignment",
            "roles": list(roles),
            "hard": True,
            "source": "resource_relation_compiler",
        }
        for roles in sorted(hard_same)
    ] + [
        {
            "type": "distinct_assignment",
            "roles": list(roles),
            "hard": True,
            "source": "resource_relation_compiler",
        }
        for roles in sorted(hard_distinct)
    ]
    preferences = [
        {
            "type": "prefer_same_assignment",
            "roles": list(roles),
            "weight": 1.0,
            "reason": "使用者偏好由同一 logical resource 執行",
            "source": "resource_relation_compiler",
        }
        for roles in sorted(soft_same)
    ] + [
        {
            "type": "prefer_distinct_assignment",
            "roles": list(roles),
            "weight": 1.0,
            "reason": "使用者偏好由不同 logical resources 分工",
            "source": "resource_relation_compiler",
        }
        for roles in sorted(soft_distinct)
    ]
    return constraints, preferences, decisions, related_role_pairs


def _add_object_ownership_constraints(actions, role_by_action, constraints):
    """Keep an explicitly acquired object on the same role until release."""
    acquired_by = {}
    existing = {
        (item["type"], tuple(sorted(item["roles"])))
        for item in constraints
        if item.get("hard")
    }

    for action in actions:
        function_name = action["function_name"]
        object_id = action.get("arguments", {}).get("object_id")
        if not isinstance(object_id, str) or not object_id:
            continue
        if function_name == "pick_object":
            acquired_by[object_id] = action["key"]
            continue
        if function_name != "place_object" or object_id not in acquired_by:
            continue

        acquire_key = acquired_by.pop(object_id)
        roles = tuple(sorted((
            f"role_{role_by_action[acquire_key] + 1}",
            f"role_{role_by_action[action['key']] + 1}",
        )))
        if roles[0] == roles[1]:
            continue
        if ("distinct_assignment", roles) in existing:
            raise SemanticIRError(
                "OBJECT_OWNERSHIP_CONFLICT",
                "object ownership continuity 與 distinct assignment 衝突："
                f"{object_id} ({acquire_key}, {action['key']}) -> {roles}",
                "constraints",
            )
        if ("same_assignment", roles) in existing:
            # Keep one constraint while exposing the deterministic invariant;
            # explicit Stage2 provenance remains in resource_decisions.
            for item in constraints:
                if (
                    item["type"] == "same_assignment"
                    and tuple(sorted(item["roles"])) == roles
                ):
                    item["source"] = "object_ownership_continuity"
                    break
            continue
        constraints.append({
            "type": "same_assignment",
            "roles": list(roles),
            "hard": True,
            "source": "object_ownership_continuity",
        })
        existing.add(("same_assignment", roles))

    # Ownership may connect roles transitively, so recheck all hard relations.
    role_ids = {
        role
        for item in constraints
        if item.get("hard")
        for role in item["roles"]
    }
    parent = {role: role for role in role_ids}

    def find(role):
        while parent[role] != role:
            parent[role] = parent[parent[role]]
            role = parent[role]
        return role

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for item in constraints:
        if item.get("hard") and item["type"] == "same_assignment":
            anchor = item["roles"][0]
            for role in item["roles"][1:]:
                union(anchor, role)
    for item in constraints:
        if item.get("hard") and item["type"] == "distinct_assignment":
            anchor = item["roles"][0]
            if any(find(anchor) == find(role) for role in item["roles"][1:]):
                raise SemanticIRError(
                    "OBJECT_OWNERSHIP_CONFLICT",
                    "object ownership continuity 與 hard assignment constraints 衝突："
                    f"{item['roles']}",
                    "constraints",
                )


def compile_pair_relations_to_plan(
    actions: list[dict[str, Any]],
    pair_relations: list[dict[str, str]],
    tool_catalog: list[dict[str, Any]],
    *,
    prefer_distinct_assignment: bool = False,
) -> dict[str, Any]:
    """Compile fixed actions and enum pair classifications into a formal Plan."""
    validate_pair_relations(actions, pair_relations)
    action_keys = [action["key"] for action in actions]
    action_by_key = {action["key"]: action for action in actions}
    tool_by_name = {tool["api_function"]: tool for tool in tool_catalog}
    pair_edges, parallel_pairs = _pair_edges(pair_relations)
    deterministic_edges, manipulation_decisions = (
        _deterministic_manipulation_edges(actions)
    )
    edges, parallel_pairs = _merge_temporal_edges(
        pair_edges,
        parallel_pairs,
        deterministic_edges,
    )
    _validate_compiled_edges(action_keys, edges)

    reduced_edges = _transitive_reduction(edges)
    role_by_action, role_templates = _deterministic_action_groups(
        actions,
        edges,
        _resource_separation_pairs(pair_relations),
    )

    for left, right in parallel_pairs:
        if role_by_action[left] == role_by_action[right]:
            raise SemanticIRError(
                "PARALLEL_GROUP_CONFLICT",
                f"parallel actions {left}, {right} 被 deterministic grouping 放入同一 role",
                "relations",
            )

    roles = []
    for index, template in enumerate(role_templates, start=1):
        tool = tool_by_name[template["function_name"]]
        target_types = tool.get("target_types") or ["robot_arm"]
        roles.append({
            "id": f"role_{index}",
            "target_type": sorted(target_types)[0],
            "required_capabilities": [],
            "required_zones": [],
            "binding": {"mode": "automatic"},
        })

    step_id = {key: f"step_{index}" for index, key in enumerate(action_keys, start=1)}
    order = {key: index for index, key in enumerate(action_keys)}
    steps = []
    for action in actions:
        key = action["key"]
        incoming = sorted(
            (source for source, target in reduced_edges if target == key),
            key=order.get,
        )
        steps.append({
            "id": step_id[key],
            "role": f"role_{role_by_action[key] + 1}",
            "action": {
                "function_name": action["function_name"],
                "arguments": deepcopy(action["arguments"]),
            },
            "depends_on": [
                {"step_id": step_id[source], "condition": "succeeded"}
                for source in incoming
            ],
            "satisfies": [],
        })

    constraints, preferences, resource_decisions, resource_role_pairs = (
        _compile_resource_assignments(pair_relations, role_by_action)
    )
    _add_object_ownership_constraints(actions, role_by_action, constraints)
    preference_keys = {
        (item["type"], tuple(sorted(item["roles"])))
        for item in preferences
    }
    seen_role_pairs = set()
    if prefer_distinct_assignment:
        for left, right in sorted(parallel_pairs):
            roles_pair = tuple(sorted((
                f"role_{role_by_action[left] + 1}",
                f"role_{role_by_action[right] + 1}",
            )))
            if roles_pair in seen_role_pairs:
                continue
            seen_role_pairs.add(roles_pair)
            # Keyword policy is compatibility/local fallback only. Explicit
            # resource semantics for this role pair always take precedence.
            if roles_pair in resource_role_pairs:
                continue
            preference_key = ("prefer_distinct_assignment", roles_pair)
            if preference_key in preference_keys:
                continue
            preferences.append({
                "type": "prefer_distinct_assignment",
                "roles": list(roles_pair),
                "weight": 1.0,
                "reason": "使用者語意允許 parallel branches 優先分工",
                "source": "pair_relation_compiler",
            })
            preference_keys.add(preference_key)

    plan = validate_plan({
        "schema_version": "1.0",
        "requirements": [],
        "roles": roles,
        "steps": steps,
        "constraints": constraints,
        "assignment_preferences": preferences,
    })
    pair_decisions = []
    for item in pair_relations:
        left = item["action_a"]
        right = item["action_b"]
        relation = item["relation"]
        decision = {
            "action_a": left,
            "action_b": right,
            "raw_llm_relation": relation,
            "final_edge": None,
            "parallel": tuple(sorted((left, right))) in parallel_pairs,
        }

        if (left, right) in edges:
            decision["final_edge"] = [left, right]
        elif (right, left) in edges:
            decision["final_edge"] = [right, left]

        pair_decisions.append(decision)
    return {
        "plan": plan,
        "actions": deepcopy(actions),
        "pair_relations": deepcopy(pair_relations),
        "pair_decisions": pair_decisions,
        "edges": sorted(edges),
        "parallel_pairs": [list(pair) for pair in sorted(parallel_pairs)],
        "reduced_edges": sorted(reduced_edges),
        "resource_relations": [
            {
                "action_a": item["action_a"],
                "action_b": item["action_b"],
                "resource_relation": item.get("resource_relation", "NONE"),
            }
            for item in pair_relations
        ],
        "resource_decisions": resource_decisions,
        "role_mapping": {
            action_key: f"role_{role_index + 1}"
            for action_key, role_index in role_by_action.items()
        },
        "compiled_constraints": deepcopy(constraints),
        "compiled_assignment_preferences": deepcopy(preferences),
        "manipulation_decisions": deepcopy(manipulation_decisions),
    }
