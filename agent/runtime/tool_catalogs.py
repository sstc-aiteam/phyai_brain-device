"""Tool catalogs for simulation and real robot execution."""
from __future__ import annotations
from typing import Any

from .dispatch_schema import BoundAction
from .tool_model import (
    ArgRef as A, Condition as C, DEVICE, Effect as E, FactRef as F,
    FactValueRef, InteractionLocationRef, ParameterSpec as P,
    ToolCatalog, ToolSpec,
)
from .world_state import WorldState


def _arm(name, desc, params, *, pre=(), eff=(), primary_arg=None,
         primary_id=None, control_id=None, fixed_device_id=None,
         exclusive=(), capacity=None):
    return ToolSpec(
        name=name, description=desc, parameters=params,
        preconditions=tuple(pre), effects=tuple(eff),
        device_target_type="robot_arm", required_device_tags=("manipulator",),
        required_capabilities=("arm_motion",), primary_entity_arg=primary_arg,
        primary_entity_id=primary_id, control_entity_id=control_id,
        fixed_device_id=fixed_device_id, exclusive_entity_args=tuple(exclusive),
        capacity_limited_entity_args=dict(capacity or {}),
    )


def simulation_runtime_tools() -> ToolCatalog:
    """Generic semantic tools used by benchmark/simulation scenarios."""
    pick = _arm(
        "pick_object", "Pick one object and keep holding it.",
        {"object_id": P("Object to pick.", required_tags=("pickable",))},
        pre=(
            C(F(A("object_id"), "held_by"), "eq", None),
            C(F(DEVICE, "holding"), "if_present_eq", None),
            C(F(DEVICE, "controlling"), "if_present_eq", None),
            C(F(DEVICE, "location"), "if_known_eq", InteractionLocationRef(A("object_id"))),
        ),
        eff=(
            E(F(A("object_id"), "held_by"), DEVICE),
            E(F(A("object_id"), "location"), None),
            E(F(DEVICE, "holding"), A("object_id")),
        ),
        primary_arg="object_id", exclusive=("object_id",),
    )
    place = _arm(
        "place_object", "Place the object currently held by the manipulator.",
        {
            "object_id": P("Held object.", required_tags=("pickable",)),
            "destination_id": P("Placement target.", required_tags=("placement_target",)),
        },
        pre=(
            C(F(A("object_id"), "held_by"), "eq", DEVICE),
            C(F(DEVICE, "holding"), "eq", A("object_id")),
            C(F(A("destination_id"), "open_state"), "if_present_eq", "open"),
            C(F(DEVICE, "location"), "if_known_eq",
              InteractionLocationRef(A("destination_id"), fallback_to_entity_id=True)),
        ),
        eff=(
            E(F(A("object_id"), "held_by"), None),
            E(F(A("object_id"), "location"), A("destination_id")),
            E(F(DEVICE, "holding"), None),
        ),
        primary_arg="object_id", exclusive=("object_id",),
        capacity={"destination_id": "parallel_place_capacity"},
    )
    open_container = _arm(
        "open_container", "Open a container and keep controlling it.",
        {"container_id": P("Openable container.", required_tags=("openable",))},
        pre=(
            C(F(A("container_id"), "open_state"), "eq", "closed"),
            C(F(DEVICE, "holding"), "if_present_eq", None),
            C(F(DEVICE, "controlling"), "if_present_eq", None),
        ),
        eff=(
            E(F(A("container_id"), "open_state"), "open"),
            E(F(DEVICE, "controlling"), A("container_id")),
        ),
        primary_arg="container_id", exclusive=("container_id",),
    )
    close_container = _arm(
        "close_container", "Close the controlled container and release it.",
        {"container_id": P("Openable container.", required_tags=("openable",))},
        pre=(
            C(F(A("container_id"), "open_state"), "eq", "open"),
            C(F(DEVICE, "controlling"), "eq", A("container_id")),
        ),
        eff=(
            E(F(A("container_id"), "open_state"), "closed"),
            E(F(DEVICE, "controlling"), None),
        ),
        primary_arg="container_id", exclusive=("container_id",),
    )
    fill = _arm(
        "fill_container", "Fill a held container from a liquid source.",
        {
            "container_id": P("Fillable container.", required_tags=("fillable",)),
            "source_id": P("Liquid source.", required_tags=("liquid_source",)),
        },
        pre=(
            C(F(A("container_id"), "held_by"), "eq", DEVICE),
            C(F(DEVICE, "holding"), "eq", A("container_id")),
            C(F(DEVICE, "location"), "if_known_eq", InteractionLocationRef(A("source_id"))),
        ),
        eff=(
            E(F(A("container_id"), "contents"), FactValueRef(A("source_id"), "liquid_type")),
            E(F(A("container_id"), "fill_state"), "filled"),
        ),
        primary_arg="container_id", exclusive=("container_id", "source_id"),
    )
    nudge = _arm(
        "nudge_arm", "Small Cartesian semantic adjustment.",
        {
            "direction": P("Direction.", must_exist=False,
                           allowed_values=("left","right","forward","backward","up","down")),
            "step": P("Magnitude.", must_exist=False,
                      allowed_values=("small","medium","large")),
        },
        eff=(
            E(F(DEVICE, "last_nudge_direction"), A("direction")),
            E(F(DEVICE, "last_nudge_step"), A("step")),
        ),
    )
    navigate = ToolSpec(
        name="navigate_to", description="Navigate a compatible mobile base.",
        parameters={"destination_id": P("Navigation target.", required_tags=("navigation_target",))},
        preconditions=(C(F(DEVICE, "location"), "ne", A("destination_id")),),
        effects=(E(F(DEVICE, "location"), A("destination_id")),),
        device_target_type="mobile_robot", required_device_tags=("mobile_base",),
        required_capabilities=("base_motion",), primary_entity_arg="destination_id",
    )
    move_payload = ToolSpec(
        name="move_payload_to", description="Move a transportable payload to a target.",
        parameters={
            "payload_id": P("Payload.", required_tags=("transportable",)),
            "destination_id": P("Navigation target.", required_tags=("navigation_target",)),
        },
        preconditions=(),
        effects=(
            E(F(A("payload_id"), "location"), A("destination_id")),
            E(F(DEVICE, "location"), A("destination_id")),
        ),
        device_target_type="mobile_robot", required_device_tags=("mobile_base",),
        required_capabilities=("base_motion",), primary_entity_arg="payload_id",
        exclusive_entity_args=("payload_id",),
    )
    inspect = ToolSpec(
        name="inspect", description="Inspect a machine at the current location.",
        parameters={"point_id": P("Inspection point.", required_tags=("inspection_point",))},
        preconditions=(C(F(DEVICE, "location"), "eq", A("point_id")),),
        effects=(E(F(A("point_id"), "inspected"), True),),
        device_target_type="quadruped", required_device_tags=("mobile_base","inspection_sensor"),
        required_capabilities=("rgb_inspection",), primary_entity_arg="point_id",
        exclusive_entity_args=("point_id",),
    )
    capture = ToolSpec(
        name="capture_evidence", description="Capture evidence for an anomaly.",
        parameters={"point_id": P("Inspection point.", required_tags=("inspection_point",))},
        preconditions=(
            C(F(DEVICE, "location"), "eq", A("point_id")),
            C(F(A("point_id"), "health_state"), "eq", "anomaly"),
            C(F(A("point_id"), "inspected"), "eq", True),
        ),
        effects=(E(F(A("point_id"), "evidence_captured"), True),),
        device_target_type="quadruped", required_device_tags=("mobile_base","inspection_sensor"),
        required_capabilities=("capture_evidence",), primary_entity_arg="point_id",
        exclusive_entity_args=("point_id",),
    )
    report = ToolSpec(
        name="report_anomaly", description="Report an anomaly after evidence capture.",
        parameters={"point_id": P("Inspection point.", required_tags=("inspection_point",))},
        preconditions=(
            C(F(DEVICE, "location"), "eq", A("point_id")),
            C(F(A("point_id"), "health_state"), "eq", "anomaly"),
            C(F(A("point_id"), "evidence_captured"), "eq", True),
        ),
        effects=(E(F(A("point_id"), "anomaly_reported"), True),),
        device_target_type="quadruped", required_device_tags=("mobile_base","inspection_sensor"),
        required_capabilities=("inspection_report",), primary_entity_arg="point_id",
        exclusive_entity_args=("point_id",),
    )
    return ToolCatalog([
        pick, place, open_container, close_container, fill, nudge,
        navigate, move_payload, inspect, capture, report,
    ])


_REAL_CONTAINERS = {
    "trash_can": ("place_object_in_trash_can", "open_trash_can", "close_trash_can"),
    "top_cabinet": ("place_object_in_top_cabinet", "open_top_cabinet", "close_top_cabinet"),
    "second_drawer": ("place_object_in_second_drawer", "open_second_drawer", "close_second_drawer"),
}


def _physical_definitions(names):
    from agent.tools import get_tool
    result, missing = {}, []
    for name in names:
        value = get_tool(name)
        if not isinstance(value, dict):
            missing.append(name)
        else:
            result[name] = value
    if missing:
        raise RuntimeError(f"agent.tools missing physical tools: {sorted(missing)}")
    return result


def real_runtime_tools() -> ToolCatalog:
    """Semantic wrappers over the existing real agent/tools.py whitelist."""
    names = {"move_right_arm_initial", "move_arm_default", "move_arm_step"}
    for triple in _REAL_CONTAINERS.values():
        names.update(triple)
    defs = _physical_definitions(names)

    tools = [
        _arm("move_right_arm_initial", defs["move_right_arm_initial"]["description"], {},
             eff=(E(F(DEVICE, "pose_state"), "initial"),), fixed_device_id="right_arm"),
        _arm("move_arm_default", defs["move_arm_default"]["description"], {},
             eff=(E(F(DEVICE, "pose_state"), "default"),)),
        _arm(
            "move_arm_step", defs["move_arm_step"]["description"],
            {
                "direction": P("x+/x-/y+/y-/z+/z-", must_exist=False,
                               allowed_values=("x+","x-","y+","y-","z+","z-"),
                               schema={"type":"string"}),
                "distance": P("Meters.", must_exist=False, required=False,
                              schema={"type":"number","minimum":0.001,"maximum":0.5,"default":0.01}),
                "speed": P("m/s.", must_exist=False, required=False,
                           schema={"type":"number","minimum":0.01,"maximum":0.25,"default":0.1}),
            },
            eff=(E(F(DEVICE, "last_move_direction"), A("direction")),),
        ),
    ]

    for target, (place_name, open_name, close_name) in _REAL_CONTAINERS.items():
        tools.append(_arm(
            place_name, defs[place_name]["description"],
            {"object_id": P("Semantic object id; runtime resolves XYZ/yaw.", required_tags=("pickable",))},
            pre=(
                C(F(A("object_id"), "held_by"), "eq", None),
                C(F(DEVICE, "holding"), "if_present_eq", None),
                C(F(DEVICE, "controlling"), "if_present_eq", None),
                C(F(target, "open_state"), "if_present_eq", "open"),
            ),
            eff=(
                E(F(A("object_id"), "held_by"), None),
                E(F(A("object_id"), "location"), target),
                E(F(DEVICE, "holding"), None),
            ),
            primary_arg="object_id", exclusive=("object_id",),
        ))
        tools.append(_arm(
            open_name, defs[open_name]["description"], {},
            pre=(
                C(F(target, "open_state"), "eq", "closed"),
                C(F(DEVICE, "holding"), "if_present_eq", None),
                C(F(DEVICE, "controlling"), "if_present_eq", None),
            ),
            eff=(E(F(target, "open_state"), "open"), E(F(DEVICE, "controlling"), target)),
            primary_id=target, control_id=target,
        ))
        tools.append(_arm(
            close_name, defs[close_name]["description"], {},
            pre=(C(F(target, "open_state"), "eq", "open"), C(F(DEVICE, "controlling"), "eq", target)),
            eff=(E(F(target, "open_state"), "closed"), E(F(DEVICE, "controlling"), None)),
            primary_id=target, control_id=target,
        ))
    return ToolCatalog(tools)


def real_execution_arguments(bound: BoundAction, world: WorldState) -> dict[str, Any]:
    """Resolve semantic args into existing physical tool arguments."""
    name, args = bound.action.function_name, dict(bound.action.arguments)
    if name in {v[0] for v in _REAL_CONTAINERS.values()}:
        object_id = str(args["object_id"])
        xyz = world.get(object_id, "position") or world.get(object_id, "robot_xyz")
        if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
            raise RuntimeError(f"{object_id} missing position/robot_xyz")
        return {
            "point_xyz": [float(v) for v in xyz],
            "yaw_deg": float(world.get(object_id, "yaw_deg", 0.0) or 0.0),
        }
    if name == "move_arm_step":
        return {
            "direction": args["direction"],
            "distance": float(args.get("distance", 0.01)),
            "speed": float(args.get("speed", 0.1)),
        }
    return args
