from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


_MISSING = object()


class WorldState:
    """Canonical semantic world state.

    The runtime brain never owns truth.  Perception / verification updates this
    store and every decision is made from the current snapshot.
    """

    def __init__(self, state: dict[str, Any] | None = None):
        raw = deepcopy(state or {"entities": {}})
        if "entities" not in raw:
            raw = {"entities": raw}
        if not isinstance(raw.get("entities"), dict):
            raise ValueError("WorldState.entities must be a dict")
        self._state = raw

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def clone(self) -> "WorldState":
        return WorldState(self.snapshot())

    def entity_ids(self) -> list[str]:
        return list(self._state["entities"].keys())

    def has_entity(self, entity_id: str) -> bool:
        return entity_id in self._state["entities"]

    def entity(self, entity_id: str) -> dict[str, Any]:
        if not self.has_entity(entity_id):
            raise KeyError(f"Unknown entity: {entity_id}")
        return deepcopy(self._state["entities"][entity_id])

    def get(
        self,
        entity_id: str,
        field: str,
        default: Any = None,
    ) -> Any:
        if not self.has_entity(entity_id):
            return default
        return deepcopy(
            self._state["entities"][entity_id].get(field, default)
        )

    def has_field(self, entity_id: str, field: str) -> bool:
        return (
            self.has_entity(entity_id)
            and field in self._state["entities"][entity_id]
        )

    def set(self, entity_id: str, field: str, value: Any) -> None:
        self._state["entities"].setdefault(entity_id, {})[field] = deepcopy(value)

    def update_entity(self, entity_id: str, values: dict[str, Any]) -> None:
        if not isinstance(values, dict):
            raise TypeError("values must be dict")
        entity = self._state["entities"].setdefault(entity_id, {})
        for key, value in values.items():
            entity[key] = deepcopy(value)

    def tags(self, entity_id: str) -> set[str]:
        raw = self.get(entity_id, "tags", [])
        return {str(x) for x in raw} if isinstance(raw, list) else set()

    def find(
        self,
        *,
        entity_type: str | None = None,
        required_tags: Iterable[str] = (),
    ) -> list[str]:
        tags = set(required_tags)
        found: list[str] = []
        for entity_id in self.entity_ids():
            entity = self.entity(entity_id)
            if entity_type and entity.get("type") != entity_type:
                continue
            if not tags.issubset(self.tags(entity_id)):
                continue
            found.append(entity_id)
        return found

    def fact(self, entity_id: str, field: str) -> tuple[str, str]:
        return entity_id, field

    def __repr__(self) -> str:
        return f"WorldState({self._state!r})"
