"""
Compatibility shim.

Canonical implementation:
    agent.runtime.decision.interpreter
"""

from .domain.task import TaskSpec
from .decision.interpreter import interpret_task

__all__ = [
    "TaskSpec",
    "interpret_task",
]
