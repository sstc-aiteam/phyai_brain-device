"""
Semantic runtime tool catalog.

This module defines WHAT actions the closed-loop runtime may reason about.

It does NOT:
- call physical robot functions
- resolve semantic arguments into driver arguments
- hardcode specific cabinet / drawer / trash-can implementations
- execute actions

Physical execution mapping belongs to the executor / action adapter layer.
"""

from __future__ import annotations

from .tool_model import (
    ArgRef as A,
    Condition as C,
    DEVICE,
    Effect as E,
    FactRef as F,
    FactValueRef,
    InteractionLocationRef,
    ParameterSpec as P,
    ToolCatalog,
    ToolSpec,
)


def _arm(
    name,
    description,
    parameters,
    *,
    preconditions=(),
    effects=(),
    primary_arg=None,
    primary_id=None,
    control_id=None,
    fixed_device_id=None,
    exclusive=(),
    capacity=None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        preconditions=tuple(preconditions),
        effects=tuple(effects),
        device_target_type="robot_arm",
        required_device_tags=("manipulator",),
        required_capabilities=("arm_motion",),
        primary_entity_arg=primary_arg,
        primary_entity_id=primary_id,
        control_entity_id=control_id,
        fixed_device_id=fixed_device_id,
        exclusive_entity_args=tuple(exclusive),
        capacity_limited_entity_args=dict(
            capacity or {}
        ),
    )


def runtime_tools() -> ToolCatalog:
    """
    Canonical semantic tools available to RuntimeBrain.

    These actions describe semantic intent only.  Device binding and physical
    execution are performed later.
    """

    pick_object = _arm(
        "pick_object",
        "Pick one object and keep holding it.",
        {
            "object_id": P(
                "Object to pick.",
                required_tags=("pickable",),
            ),
        },
        preconditions=(
            C(
                F(A("object_id"), "held_by"),
                "eq",
                None,
            ),
            C(
                F(DEVICE, "holding"),
                "if_present_eq",
                None,
            ),
            C(
                F(DEVICE, "controlling"),
                "if_present_eq",
                None,
            ),
            C(
                F(DEVICE, "location"),
                "if_known_eq",
                InteractionLocationRef(
                    A("object_id")
                ),
            ),
        ),
        effects=(
            E(
                F(A("object_id"), "held_by"),
                DEVICE,
            ),
            E(
                F(A("object_id"), "location"),
                None,
            ),
            E(
                F(DEVICE, "holding"),
                A("object_id"),
            ),
        ),
        primary_arg="object_id",
        exclusive=("object_id",),
    )

    place_object = _arm(
        "place_object",
        "Place the object currently held by the manipulator.",
        {
            "object_id": P(
                "Held object.",
                required_tags=("pickable",),
            ),
            "destination_id": P(
                "Placement target.",
                required_tags=("placement_target",),
            ),
        },
        preconditions=(
            C(
                F(A("object_id"), "held_by"),
                "eq",
                DEVICE,
            ),
            C(
                F(DEVICE, "holding"),
                "eq",
                A("object_id"),
            ),
            C(
                F(A("destination_id"), "open_state"),
                "if_present_eq",
                "open",
            ),
            C(
                F(DEVICE, "location"),
                "if_known_eq",
                InteractionLocationRef(
                    A("destination_id"),
                    fallback_to_entity_id=True,
                ),
            ),
        ),
        effects=(
            E(
                F(A("object_id"), "held_by"),
                None,
            ),
            E(
                F(A("object_id"), "location"),
                A("destination_id"),
            ),
            E(
                F(DEVICE, "holding"),
                None,
            ),
        ),
        primary_arg="object_id",
        exclusive=("object_id",),
        capacity={
            "destination_id":
                "parallel_place_capacity",
        },
    )

    open_container = _arm(
        "open_container",
        "Open a container and keep controlling it.",
        {
            "container_id": P(
                "Openable container.",
                required_tags=("openable",),
            ),
        },
        preconditions=(
            C(
                F(A("container_id"), "open_state"),
                "eq",
                "closed",
            ),
            C(
                F(DEVICE, "holding"),
                "if_present_eq",
                None,
            ),
            C(
                F(DEVICE, "controlling"),
                "if_present_eq",
                None,
            ),
        ),
        effects=(
            E(
                F(A("container_id"), "open_state"),
                "open",
            ),
            E(
                F(DEVICE, "controlling"),
                A("container_id"),
            ),
        ),
        primary_arg="container_id",
        exclusive=("container_id",),
    )

    close_container = _arm(
        "close_container",
        "Close the controlled container and release it.",
        {
            "container_id": P(
                "Openable container.",
                required_tags=("openable",),
            ),
        },
        preconditions=(
            C(
                F(A("container_id"), "open_state"),
                "eq",
                "open",
            ),
            C(
                F(DEVICE, "controlling"),
                "eq",
                A("container_id"),
            ),
        ),
        effects=(
            E(
                F(A("container_id"), "open_state"),
                "closed",
            ),
            E(
                F(DEVICE, "controlling"),
                None,
            ),
        ),
        primary_arg="container_id",
        exclusive=("container_id",),
    )

    fill_container = _arm(
        "fill_container",
        "Fill a held container from a liquid source.",
        {
            "container_id": P(
                "Fillable container.",
                required_tags=("fillable",),
            ),
            "source_id": P(
                "Liquid source.",
                required_tags=("liquid_source",),
            ),
        },
        preconditions=(
            C(
                F(A("container_id"), "held_by"),
                "eq",
                DEVICE,
            ),
            C(
                F(DEVICE, "holding"),
                "eq",
                A("container_id"),
            ),
            C(
                F(DEVICE, "location"),
                "if_known_eq",
                InteractionLocationRef(
                    A("source_id")
                ),
            ),
        ),
        effects=(
            E(
                F(A("container_id"), "contents"),
                FactValueRef(
                    A("source_id"),
                    "liquid_type",
                ),
            ),
            E(
                F(A("container_id"), "fill_state"),
                "filled",
            ),
        ),
        primary_arg="container_id",
        exclusive=(
            "container_id",
            "source_id",
        ),
    )

    nudge_arm = _arm(
        "nudge_arm",
        "Small Cartesian semantic adjustment.",
        {
            "direction": P(
                "Direction.",
                must_exist=False,
                allowed_values=(
                    "left",
                    "right",
                    "forward",
                    "backward",
                    "up",
                    "down",
                ),
            ),
            "step": P(
                "Magnitude.",
                must_exist=False,
                allowed_values=(
                    "small",
                    "medium",
                    "large",
                ),
            ),
        },
        effects=(
            E(
                F(DEVICE, "last_nudge_direction"),
                A("direction"),
            ),
            E(
                F(DEVICE, "last_nudge_step"),
                A("step"),
            ),
        ),
    )

    navigate_to = ToolSpec(
        name="navigate_to",
        description="Navigate a compatible mobile base.",
        parameters={
            "destination_id": P(
                "Navigation target.",
                required_tags=("navigation_target",),
            ),
        },
        preconditions=(
            C(
                F(DEVICE, "location"),
                "ne",
                A("destination_id"),
            ),
        ),
        effects=(
            E(
                F(DEVICE, "location"),
                A("destination_id"),
            ),
        ),
        device_target_type="mobile_robot",
        required_device_tags=("mobile_base",),
        required_capabilities=("base_motion",),
        primary_entity_arg="destination_id",
    )

    move_payload_to = ToolSpec(
        name="move_payload_to",
        description="Move a transportable payload to a target.",
        parameters={
            "payload_id": P(
                "Payload.",
                required_tags=("transportable",),
            ),
            "destination_id": P(
                "Navigation target.",
                required_tags=("navigation_target",),
            ),
        },
        preconditions=(),
        effects=(
            E(
                F(A("payload_id"), "location"),
                A("destination_id"),
            ),
            E(
                F(DEVICE, "location"),
                A("destination_id"),
            ),
        ),
        device_target_type="mobile_robot",
        required_device_tags=("mobile_base",),
        required_capabilities=("base_motion",),
        primary_entity_arg="payload_id",
        exclusive_entity_args=("payload_id",),
    )

    inspect = ToolSpec(
        name="inspect",
        description="Inspect a machine at the current location.",
        parameters={
            "point_id": P(
                "Inspection point.",
                required_tags=("inspection_point",),
            ),
        },
        preconditions=(
            C(
                F(DEVICE, "location"),
                "eq",
                A("point_id"),
            ),
        ),
        effects=(
            E(
                F(A("point_id"), "inspected"),
                True,
            ),
        ),
        device_target_type="quadruped",
        required_device_tags=(
            "mobile_base",
            "inspection_sensor",
        ),
        required_capabilities=("rgb_inspection",),
        primary_entity_arg="point_id",
        exclusive_entity_args=("point_id",),
    )

    capture_evidence = ToolSpec(
        name="capture_evidence",
        description="Capture evidence for an anomaly.",
        parameters={
            "point_id": P(
                "Inspection point.",
                required_tags=("inspection_point",),
            ),
        },
        preconditions=(
            C(
                F(DEVICE, "location"),
                "eq",
                A("point_id"),
            ),
            C(
                F(A("point_id"), "health_state"),
                "eq",
                "anomaly",
            ),
            C(
                F(A("point_id"), "inspected"),
                "eq",
                True,
            ),
        ),
        effects=(
            E(
                F(A("point_id"), "evidence_captured"),
                True,
            ),
        ),
        device_target_type="quadruped",
        required_device_tags=(
            "mobile_base",
            "inspection_sensor",
        ),
        required_capabilities=("capture_evidence",),
        primary_entity_arg="point_id",
        exclusive_entity_args=("point_id",),
    )

    report_anomaly = ToolSpec(
        name="report_anomaly",
        description="Report an anomaly after evidence capture.",
        parameters={
            "point_id": P(
                "Inspection point.",
                required_tags=("inspection_point",),
            ),
        },
        preconditions=(
            C(
                F(DEVICE, "location"),
                "eq",
                A("point_id"),
            ),
            C(
                F(A("point_id"), "health_state"),
                "eq",
                "anomaly",
            ),
            C(
                F(A("point_id"), "evidence_captured"),
                "eq",
                True,
            ),
        ),
        effects=(
            E(
                F(A("point_id"), "anomaly_reported"),
                True,
            ),
        ),
        device_target_type="quadruped",
        required_device_tags=(
            "mobile_base",
            "inspection_sensor",
        ),
        required_capabilities=("inspection_report",),
        primary_entity_arg="point_id",
        exclusive_entity_args=("point_id",),
    )

    return ToolCatalog([
        pick_object,
        place_object,
        open_container,
        close_container,
        fill_container,
        nudge_arm,
        navigate_to,
        move_payload_to,
        inspect,
        capture_evidence,
        report_anomaly,
    ])


def simulation_runtime_tools() -> ToolCatalog:
    """
    Compatibility alias for existing benchmark code.

    New runtime code should use runtime_tools().
    """

    return runtime_tools()


__all__ = [
    "runtime_tools",
    "simulation_runtime_tools",
]
