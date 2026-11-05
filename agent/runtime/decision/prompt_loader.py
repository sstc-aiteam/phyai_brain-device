"""Load the PM-registry next-action Planner prompt from the editable Markdown Skill.

No robot or service imports. Read on every call so Skill edits take effect in a new
planning request without restarting an already running Python process.
"""
from __future__ import annotations

from pathlib import Path


_PROMPT_FILE = Path(__file__).resolve().parents[2] / "skills" / "plan_next_action.md"
_ALIAS_PLACEHOLDER = "{{drawer_front_alias}}"


def _read_section(source: str, section: str) -> str:
    start = f"<!-- PLAN_PROMPT:{section}:BEGIN -->"
    end = f"<!-- PLAN_PROMPT:{section}:END -->"
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(f"Plan Skill section 缺少或重複：{section}")
    middle = source.split(start, 1)[1].split(end, 1)[0]
    # Rule lines are concatenated, reproducing the original Python adjacent
    # string literals. Headings and comments outside markers do not reach LLM.
    value = "".join(line.strip() for line in middle.splitlines() if line.strip())
    if not value:
        raise RuntimeError(f"Plan Skill section 是空的：{section}")
    return value


def load_next_action_prompt(*, already_observed: bool = False,
                            drawer_front_alias: str | None = None) -> str:
    source = _PROMPT_FILE.read_text(encoding="utf-8")
    rules = _read_section(source, "base")
    if drawer_front_alias:
        alias_rule = _read_section(source, "site_alias")
        if alias_rule.count(_ALIAS_PLACEHOLDER) != 1:
            raise RuntimeError("Plan Skill site_alias 必須包含唯一的別名佔位符")
        rules += alias_rule.replace(_ALIAS_PLACEHOLDER, drawer_front_alias)
    if already_observed:
        rules += _read_section(source, "post_observe")
    return rules
