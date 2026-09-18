from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


class WorldState:
    """
    Canonical semantic truth used by the runtime.

    Expected shape:

        {
            "entities": {
                "saline_1": {
                    "type": "object",
                    "location": "table_1",
                    "tags": ["pickable"],
                },
                "drawer_1": {
                    "type": "drawer",
                    "open_state": "closed",
                },
            }
        }

    Every public read returns copied data so callers cannot mutate the
    canonical store accidentally.
    """

    def __init__(
        self,
        state: dict[str, Any] | None = None,
    ):
        raw = deepcopy(
            state
            if state is not None
            else {"entities": {}}
        )

        # Compatibility:
        # allow callers to pass either:
        #
        #   {"entities": {...}}
        #
        # or directly:
        #
        #   {"saline_1": {...}, "drawer_1": {...}}
        if "entities" not in raw:
            raw = {
                "entities": raw,
            }

        entities = raw.get(
            "entities"
        )

        if not isinstance(
            entities,
            dict,
        ):
            raise ValueError(
                "WorldState.entities must be a dict"
            )

        for entity_id, entity in entities.items():
            if not isinstance(
                entity_id,
                str,
            ) or not entity_id:
                raise ValueError(
                    "WorldState entity IDs must be non-empty strings"
                )

            if not isinstance(
                entity,
                dict,
            ):
                raise ValueError(
                    "WorldState entity must be a dict: "
                    f"{entity_id!r}"
                )

        self._state = raw

    def snapshot(
        self,
    ) -> dict[str, Any]:
        """
        Return a fully detached JSON-like snapshot.
        """

        return deepcopy(
            self._state
        )

    def clone(
        self,
    ) -> "WorldState":
        """
        Return an independent WorldState copy.
        """

        return WorldState(
            self._state
        )

    def entity_ids(
        self,
    ) -> list[str]:
        """
        Return all currently known canonical entity IDs.
        """

        return list(
            self._state[
                "entities"
            ].keys()
        )

    def has_entity(
        self,
        entity_id: str,
    ) -> bool:
        return (
            entity_id
            in self._state[
                "entities"
            ]
        )

    def entity(
        self,
        entity_id: str,
    ) -> dict[str, Any]:
        """
        Return one detached entity record.

        Raises KeyError for an unknown entity.
        """

        if not self.has_entity(
            entity_id
        ):
            raise KeyError(
                f"Unknown entity: {entity_id}"
            )

        return deepcopy(
            self._state[
                "entities"
            ][entity_id]
        )

    def get(
        self,
        entity_id: str,
        field: str,
        default: Any = None,
    ) -> Any:
        """
        Read one semantic field.

        Unknown entity / missing field returns `default`.
        """

        if not self.has_entity(
            entity_id
        ):
            return deepcopy(
                default
            )

        return deepcopy(
            self._state[
                "entities"
            ][entity_id].get(
                field,
                default,
            )
        )

    def has_field(
        self,
        entity_id: str,
        field: str,
    ) -> bool:
        return (
            self.has_entity(
                entity_id
            )
            and field
            in self._state[
                "entities"
            ][entity_id]
        )

    def set(
        self,
        entity_id: str,
        field: str,
        value: Any,
    ) -> None:
        """
        Set one semantic field.

        This method only mutates the store. It does not perform perception
        or verification.
        """

        entity = self._state[
            "entities"
        ].setdefault(
            entity_id,
            {},
        )

        entity[field] = deepcopy(
            value
        )

    def update_entity(
        self,
        entity_id: str,
        values: dict[str, Any],
    ) -> None:
        """
        Merge semantic fields into one entity.
        """

        if not isinstance(
            values,
            dict,
        ):
            raise TypeError(
                "values must be dict"
            )

        entity = self._state[
            "entities"
        ].setdefault(
            entity_id,
            {},
        )

        for key, value in values.items():
            entity[key] = deepcopy(
                value
            )

    def tags(
        self,
        entity_id: str,
    ) -> set[str]:
        """
        Return entity tags as a set of strings.
        """

        raw = self.get(
            entity_id,
            "tags",
            [],
        )

        if not isinstance(
            raw,
            (
                list,
                tuple,
                set,
            ),
        ):
            return set()

        return {
            str(value)
            for value in raw
        }

    def find(
        self,
        *,
        entity_type: str | None = None,
        required_tags: Iterable[str] = (),
    ) -> list[str]:
        """
        Find entities by semantic type and required tags.
        """

        required = {
            str(tag)
            for tag in required_tags
        }

        found: list[str] = []

        for entity_id in self.entity_ids():
            entity = self._state[
                "entities"
            ][entity_id]

            if (
                entity_type is not None
                and entity.get(
                    "type"
                )
                != entity_type
            ):
                continue

            if not required.issubset(
                self.tags(
                    entity_id
                )
            ):
                continue

            found.append(
                entity_id
            )

        return found

    def fact(
        self,
        entity_id: str,
        field: str,
    ) -> tuple[str, str]:
        """
        Return the canonical key of one semantic fact.
        """

        return (
            entity_id,
            field,
        )

    def __repr__(
        self,
    ) -> str:
        return (
            f"WorldState({self._state!r})"
        )


__all__ = [
    "WorldState",
]
