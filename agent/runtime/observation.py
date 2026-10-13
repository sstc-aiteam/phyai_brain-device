"""
Compatibility shim.

Canonical implementation:
    agent.runtime.adapters.observation
"""

from .adapters.observation import (
    ObservationConfig,
    capture_world,
    observe_world,
    make_observer,
    default_action_target_entity,
    resolve_action_region,
    resolve_dispatch_regions,
    make_dispatch_observer,
)

__all__ = [
    "ObservationConfig",
    "capture_world",
    "observe_world",
    "make_observer",
    "default_action_target_entity",
    "resolve_action_region",
    "resolve_dispatch_regions",
    "make_dispatch_observer",
]
