"""
Compatibility shim.

Canonical implementation:
    agent.runtime.adapters.memory
"""

from .adapters.memory import ExecutionMemory

__all__ = [
    "ExecutionMemory",
]
