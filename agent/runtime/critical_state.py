"""
Compatibility shim.

Canonical implementation:
    agent.runtime.decision.context
"""

from .decision.context import build_critical_state

__all__ = [
    "build_critical_state",
]
