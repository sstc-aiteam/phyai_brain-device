"""
Compatibility shim.

Canonical implementation:
    agent.runtime.domain.task
"""

from .domain.task import (
    MaintainFactUntilGoal,
    ActionGate,
    DeviceRule,
    CoordinationSpec,
)

__all__ = [
    "MaintainFactUntilGoal",
    "ActionGate",
    "DeviceRule",
    "CoordinationSpec",
]
