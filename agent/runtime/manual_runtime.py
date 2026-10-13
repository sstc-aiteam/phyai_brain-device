"""
Compatibility shim.

Canonical implementation:
    agent.runtime.interfaces.manual_client
"""

from .interfaces.manual_client import ManualRuntime


if __name__ == "__main__":
    runtime = ManualRuntime()

    runtime.run(
        "把口罩放進垃圾桶",
        max_iterations=10,
        force_device_id="right_arm",
    )
