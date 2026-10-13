"""
Compatibility shim.

Canonical implementation:
    agent.runtime.decision.planner
"""

from .decision.planner import RuntimeBrain

__all__ = [
    "RuntimeBrain",
]
