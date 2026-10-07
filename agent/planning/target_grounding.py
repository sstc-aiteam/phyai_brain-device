"""Deterministic grounding from user mentions to entities present in a scene."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


RULES_PATH = Path(__file__).resolve().parents[1] / "rules.json"
_PLACEMENT_MARKERS = ("放進", "放入", "放到", "放至", "丟進", "丟入", "收進", "移到")
_COLLECTION_MARKERS = (
    "所有", "全部", "全部的", "每個", "每一個", "各個", "這些",
    "all", "every", "each",
)
_COLLECTION_CONTEXT_MARKERS = (
    "檢查", "盤點", "辨識", "找出", "確認", "查看",
    "inspect", "check", "detect", "inventory",
)
_MISPLACED_MARKERS = ("錯放", "放錯", "misplaced")
_CORRECT_MARKERS = ("正確", "放對", "correct")


class TargetGroundingError(ValueError):
    """Raised before Stage1 when a mentioned target is unresolved or ambiguous."""


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[\s_\-]+", "", text)
    return re.sub(r"(?<!\d)0+(?=\d)", "", text)


def _legacy_rules(rules_path: Path | None) -> dict[str, dict[str, Any]]:
    if rules_path is None:
        return {}
    with rules_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle).get("objects", {})
    return value if isinstance(value, dict) else {}


def _entities(raw_entities: Any, rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw_entities, list):
        return []

    result = []
    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue

        entity_id = str(raw.get("id") or raw.get("object_id") or "").strip()
        class_name = str(raw.get("class_name") or raw.get("class") or "").strip()
        if not entity_id:
            continue

        raw_aliases = raw.get("aliases", [])
        aliases = {
            str(value).strip()
            for value in raw_aliases
            if str(value or "").strip()
        } if isinstance(raw_aliases, list) else set()

        name = str(raw.get("name") or "").strip()
        rule = rules.get(class_name, {})
        if isinstance(rule, dict):
            aliases.update(
                str(value).strip()
                for value in (
                    *(rule.get("aliases") or []),
                    rule.get("object_label"),
                    rule.get("count_label"),
                )
                if str(value or "").strip()
            )

        attributes = {
            key: value
            for key, value in raw.items()
            if key not in {
                "id", "object_id", "class_name", "class",
                "name", "aliases", "camera_source",
            }
        }

        result.append({
            "id": entity_id,
            "class_name": class_name,
            "name": name,
            "aliases": sorted(aliases),
            "camera_source": raw.get("camera_source"),
            "attributes": attributes,
        })

    return result


def _labels(entity: dict[str, Any]) -> list[tuple[int, str]]:
    labels = [(1, entity["id"])]
    labels.extend(
        (2, value)
        for value in [entity["name"], *entity["aliases"]]
        if value
    )
    if entity["class_name"]:
        labels.append((3, entity["class_name"]))

    unique = {}
    for priority, label in labels:
        key = _normalize(label)
        if key and (key not in unique or priority < unique[key][0]):
            unique[key] = (priority, label)
    return list(unique.values())


def _clause_for_span(user_text: str, start: int, end: int) -> str:
    left_breaks = [
        user_text.rfind(token, 0, start)
        for token in ("\n", "。", "；", ";", "！", "!", "？", "?")
    ]
    left = max(left_breaks) + 1

    right_candidates = []
    for token in ("\n", "。", "；", ";", "！", "!", "？", "?"):
        pos = user_text.find(token, end)
        if pos >= 0:
            right_candidates.append(pos)
    right = min(right_candidates) if right_candidates else len(user_text)
    return user_text[left:right].strip()


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(marker) in normalized for marker in markers)


def _filter_candidates_from_clause(
    candidates: dict[str, dict[str, Any]],
    clause: str,
) -> dict[str, dict[str, Any]]:
    filtered = dict(candidates)

    if _has_any(clause, _MISPLACED_MARKERS):
        rows = {
            entity_id: entity
            for entity_id, entity in filtered.items()
            if _normalize(entity.get("attributes", {}).get("placement_state")) == "misplaced"
        }
        if rows:
            filtered = rows

    elif _has_any(clause, _CORRECT_MARKERS):
        rows = {
            entity_id: entity
            for entity_id, entity in filtered.items()
            if _normalize(entity.get("attributes", {}).get("placement_state")) == "correct"
        }
        if rows:
            filtered = rows

    return filtered


def _is_collection_context(clause: str) -> bool:
    return (
        _has_any(clause, _COLLECTION_MARKERS)
        or _has_any(clause, _COLLECTION_CONTEXT_MARKERS)
    )


def resolve_scene_mentions(
    user_text: str,
    scene_entities: list[dict[str, Any]],
    *,
    rules_path: Path | None = RULES_PATH,
) -> list[dict[str, Any]]:
    """Resolve explicit mentions using entity IDs, names, aliases and classes.

    Singular ambiguity remains an error. Explicit plural/collection language,
    or an inspection/inventory clause, may resolve to multiple scene entities.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string")

    entities = _entities(scene_entities, _legacy_rules(rules_path))
    normalized_text = _normalize(user_text)
    matches = []

    for entity in entities:
        for priority, label in _labels(entity):
            exact = list(
                re.finditer(
                    re.escape(label),
                    user_text,
                    flags=re.IGNORECASE,
                )
            )
            if exact:
                matches.extend(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "mention": match.group(0),
                        "key": _normalize(label),
                        "priority": priority,
                        "entity": entity,
                    }
                    for match in exact
                )
                continue

            normalized_label = _normalize(label)
            start = normalized_text.find(normalized_label)
            if start >= 0:
                matches.append({
                    "start": start,
                    "end": start + len(normalized_label),
                    "mention": label,
                    "key": normalized_label,
                    "priority": priority + 10,
                    "entity": entity,
                })

    matches = [
        match
        for match in matches
        if not any(
            other["start"] <= match["start"]
            and other["end"] >= match["end"]
            and (other["end"] - other["start"]) > (match["end"] - match["start"])
            for other in matches
        )
    ]

    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for match in matches:
        grouped.setdefault(
            (match["start"], match["end"], match["key"]),
            [],
        ).append(match)

    resolutions = []

    for (start, end, _key), rows in sorted(grouped.items()):
        best_priority = min(row["priority"] for row in rows)
        best = [
            row
            for row in rows
            if row["priority"] == best_priority
        ]
        candidates = {
            row["entity"]["id"]: row["entity"]
            for row in best
        }
        mention = best[0]["mention"]
        clause = _clause_for_span(user_text, start, end)
        candidates = _filter_candidates_from_clause(candidates, clause)

        if len(candidates) == 1:
            entity = next(iter(candidates.values()))
            resolutions.append({
                "mention": mention,
                "status": "resolved",
                "object_id": entity["id"],
                "class_name": entity["class_name"],
                "camera_source": entity["camera_source"],
                "attributes": entity.get("attributes", {}),
                "context": clause,
            })
            continue

        if candidates and _is_collection_context(clause):
            ordered = [
                entity
                for _entity_id, entity in sorted(candidates.items())
            ]
            resolutions.append({
                "mention": mention,
                "status": "collection",
                "object_ids": [entity["id"] for entity in ordered],
                "members": [
                    {
                        "object_id": entity["id"],
                        "class_name": entity["class_name"],
                        "camera_source": entity["camera_source"],
                        "attributes": entity.get("attributes", {}),
                    }
                    for entity in ordered
                ],
                "context": clause,
            })
            continue

        resolutions.append({
            "mention": mention,
            "status": "ambiguous",
            "candidate_ids": sorted(candidates),
            "context": clause,
        })

    return resolutions


def ground_task_targets(
    user_text: str,
    detected_objects: dict[str, Any],
    *,
    scene_entities: list[dict[str, Any]] | None = None,
    rules_path: Path = RULES_PATH,
) -> list[dict[str, Any]]:
    """Return backward-compatible flat object_id rows.

    Collection results are flattened one row per member, with collection
    metadata attached so current callers that expect object_id keep working.
    """
    if scene_entities is None:
        raw_entities = (
            detected_objects.get("objects")
            if isinstance(detected_objects, dict)
            else None
        )
    else:
        raw_entities = scene_entities

    resolutions = resolve_scene_mentions(
        user_text,
        raw_entities if isinstance(raw_entities, list) else [],
        rules_path=rules_path,
    )

    ambiguous = [
        row
        for row in resolutions
        if row["status"] == "ambiguous"
    ]
    if ambiguous:
        row = ambiguous[0]
        raise TargetGroundingError(
            f"target {row['mention']!r} grounding ambiguous: "
            f"{row['candidate_ids']}"
        )

    grounded = []
    collection_counter = 0

    for row in resolutions:
        if row["status"] == "resolved":
            grounded.append({
                "mention": row["mention"],
                "object_id": row["object_id"],
                "class_name": row["class_name"],
                "camera_source": row.get("camera_source"),
                "grounding_type": "singular",
                "attributes": row.get("attributes", {}),
                "grounding_context": row.get("context"),
            })
            continue

        if row["status"] != "collection":
            continue

        collection_counter += 1
        collection_id = f"collection_{collection_counter}"
        member_ids = list(row["object_ids"])

        for member in row["members"]:
            grounded.append({
                "mention": row["mention"],
                "object_id": member["object_id"],
                "class_name": member["class_name"],
                "camera_source": member.get("camera_source"),
                "grounding_type": "collection_member",
                "collection_id": collection_id,
                "collection_member_ids": member_ids,
                "collection_context": row.get("context"),
                "attributes": member.get("attributes", {}),
            })

    if not grounded and any(
        marker in user_text
        for marker in _PLACEMENT_MARKERS
    ):
        raise TargetGroundingError(
            "找不到可 deterministic grounding 的使用者 target mention"
        )

    return grounded
