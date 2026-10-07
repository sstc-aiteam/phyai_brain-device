"""
Compact text-backed execution memory for RuntimeBrain.

Each runtime session owns one JSONL file.  The file is intentionally simple so
other teams can inspect or consume it without importing this package.

This module does not choose actions, validate actions, or decide task progress.
It only records already-executed transitions using information produced by the
existing runtime components.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .dispatch import (
    BoundDispatch,
    DispatchDecision,
    DispatchExecutionReport,
)
from .goals import GoalSet
from .progress_monitor import ProgressFeedback
from .tool_model import ToolCatalog


_SCHEMA_VERSION = 1
_DEVICE_PLACEHOLDER = "$device"


def _resolve_device_placeholders(
    value: Any,
    *,
    device_id: str,
) -> Any:
    """Resolve the Brain-facing $device placeholder before persistence."""

    if value == _DEVICE_PLACEHOLDER:
        return device_id

    if isinstance(value, list):
        return [
            _resolve_device_placeholders(
                item,
                device_id=device_id,
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _resolve_device_placeholders(
                item,
                device_id=device_id,
            )
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _resolve_device_placeholders(
                item,
                device_id=device_id,
            )
            for key, item in value.items()
        }

    return deepcopy(value)


class ExecutionMemory:
    """
    Session-scoped, text-backed execution memory.

    Only the most recent `max_entries` executed transitions are retained.  The
    file is rewritten atomically after each append because the history window is
    deliberately tiny (default: 10 rows).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: int = 10,
    ):
        if max_entries < 1:
            raise ValueError(
                "max_entries must be >= 1"
            )

        self.path = Path(path)
        self.max_entries = int(max_entries)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        rows: list[dict[str, Any]] = []

        try:
            text = self.path.read_text(
                encoding="utf-8"
            )
        except OSError:
            return []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Keep the memory reader tolerant of a partially edited line.
                continue

            if isinstance(row, dict):
                rows.append(row)

        return rows[-self.max_entries:]

    def _write_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        rows = rows[-self.max_entries:]

        content = "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row in rows
        )

        if content:
            content += "\n"

        temporary_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        temporary_path.write_text(
            content,
            encoding="utf-8",
        )
        temporary_path.replace(
            self.path
        )

    def append(
        self,
        entry: dict[str, Any],
    ) -> None:
        if not isinstance(entry, dict):
            raise TypeError(
                "execution memory entry must be object"
            )

        rows = self.read()
        rows.append(
            deepcopy(entry)
        )
        self._write_rows(rows)

    def clear(self) -> None:
        self._write_rows([])

    def brain_view(self) -> dict[str, Any]:
        """Compact short-term task memory supplied to RuntimeBrain."""

        rows = self.read()
        last_execution = (
            deepcopy(rows[-1])
            if rows
            else None
        )

        last_action: Any = None

        if isinstance(last_execution, dict):
            actions = last_execution.get(
                "actions"
            )

            if isinstance(actions, list):
                if len(actions) == 1:
                    last_action = deepcopy(
                        actions[0]
                    )
                elif actions:
                    last_action = deepcopy(
                        actions
                    )

        return {
            "recent_executions":
                deepcopy(rows),
            "last_execution":
                last_execution,
            "last_action":
                last_action,
            "history_window": {
                "count":
                    len(rows),
                "max_entries":
                    self.max_entries,
            },
        }

    def record_execution(
        self,
        *,
        iteration: int,
        decision: DispatchDecision,
        dispatch: BoundDispatch,
        report: DispatchExecutionReport,
        progress: ProgressFeedback,
        goals: GoalSet,
        tools: ToolCatalog,
    ) -> dict[str, Any]:
        """
        Persist one dispatch after execution + fresh observation.

        The memory stores only semantic changes relevant to:
        - fields written by the selected tools;
        - current task goal fields;
        - the Brain-declared immediate target facts.

        This keeps perception noise such as bbox/confidence out of the VLM
        memory without hardcoding domain-specific semantic field names.
        """

        relevant_fact_keys: set[tuple[str, str]] = {
            (
                str(goal.subject),
                str(goal.field),
            )
            for goal in goals.conditions
        }

        persisted_actions: list[dict[str, Any]] = []

        for index, bound_action in enumerate(
            dispatch.actions
        ):
            tool = tools.get(
                bound_action.action.function_name
            )

            relevant_fact_keys |= set(
                tool.write_facts(
                    bound_action
                )
            )

            context = (
                decision.action_contexts[index]
                if index < len(
                    decision.action_contexts
                )
                else None
            )

            context_dict: dict[str, Any] | None = None

            if context is not None:
                context_dict = context.as_dict()
                context_dict = _resolve_device_placeholders(
                    context_dict,
                    device_id=bound_action.device_id,
                )

                target_fact = context_dict.get(
                    "target_fact"
                )

                if isinstance(target_fact, dict):
                    subject = target_fact.get(
                        "subject"
                    )
                    field_name = target_fact.get(
                        "field"
                    )

                    if (
                        isinstance(subject, str)
                        and subject
                        and isinstance(field_name, str)
                        and field_name
                    ):
                        relevant_fact_keys.add(
                            (
                                subject,
                                field_name,
                            )
                        )

            execution_result = next(
                (
                    result
                    for result in report.results
                    if result.bound_action
                    == bound_action
                ),
                None,
            )

            action_row = bound_action.as_dict()
            action_row[
                "decision_context"
            ] = context_dict
            action_row[
                "command_success"
            ] = (
                execution_result.command_success
                if execution_result
                is not None
                else False
            )
            action_row[
                "verified_success"
            ] = (
                execution_result.verified_success
                if execution_result
                is not None
                else False
            )

            if (
                execution_result is not None
                and execution_result.error
            ):
                action_row[
                    "error"
                ] = str(
                    execution_result.error
                )

            persisted_actions.append(
                action_row
            )

        observed_changes = [
            deepcopy(row)
            for row in progress.changed_facts
            if (
                str(
                    row.get("subject")
                ),
                str(
                    row.get("field")
                ),
            )
            in relevant_fact_keys
        ]

        entry = {
            "schema_version":
                _SCHEMA_VERSION,
            "iteration":
                int(iteration),
            "actions":
                persisted_actions,
            "observed_changes":
                observed_changes,
            "progress": {
                "progress_class":
                    progress.progress_class,
                "world_changed":
                    progress.world_changed,
                "newly_credited_goals":
                    list(
                        progress.newly_credited_goals
                    ),
                "regressed_goals":
                    list(
                        progress.regressed_goals
                    ),
                "task_complete":
                    progress.task_complete,
            },
        }

        self.append(entry)
        return entry


__all__ = [
    "ExecutionMemory",
]
