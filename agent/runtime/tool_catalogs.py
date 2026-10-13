"""
Compatibility shim.

Canonical implementation:
    agent.runtime.tools.catalog
"""

from .tools.catalog import (
    runtime_tools,
    simulation_runtime_tools,
)

__all__ = [
    "runtime_tools",
    "simulation_runtime_tools",
]
