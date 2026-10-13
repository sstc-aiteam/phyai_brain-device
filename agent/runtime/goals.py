"""
Compatibility shim.

Canonical implementation:
    agent.runtime.domain.task
"""

from .domain.task import (
    GoalCondition,
    GoalSet,
)

__all__ = [
    "GoalCondition",
    "GoalSet",
]
