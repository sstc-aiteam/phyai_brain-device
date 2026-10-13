"""
Compatibility shim.

Canonical implementation:
    agent.runtime.engine.guards.binding
"""

from .engine.guards.binding import (
    DeviceBindingError,
    DeviceBinder,
    infer_required_device,
    StaticWorldDeviceBinder,
)

__all__ = [
    "DeviceBindingError",
    "DeviceBinder",
    "infer_required_device",
    "StaticWorldDeviceBinder",
]
