from __future__ import annotations

from itertools import combinations

from .coordination_schema import CoordinationSpec
from .dispatch_schema import BoundDispatch
from .goal_schema import GoalSet
from .tool_model import ToolCatalog, ToolValidationError
from .world_state import WorldState


class DispatchValidationError(RuntimeError):
    pass


def _simulate_dispatch(
    dispatch: BoundDispatch,
    *,
    world: WorldState,
    tools: ToolCatalog,
) -> WorldState:
    simulated = world.clone()
    for bound in dispatch.actions:
        simulated = tools.get(
            bound.action.function_name
        ).apply_effects(simulated, bound)
    return simulated


def validate_dispatch(
    dispatch: BoundDispatch,
    *,
    world: WorldState,
    goals: GoalSet,
    tools: ToolCatalog,
    coordination: CoordinationSpec,
) -> None:
    """Hard validation only: legality, physics, resources and safety.

    This function intentionally does NOT reject a dispatch merely because it
    is the wrong task choice, wrong target, out of task order, or does not
    advance a goal. Those legal mistakes are executed and evaluated later by
    TaskProgressMonitor so the runtime can demonstrate closed-loop recovery.
    """

    # 1) Every action must be physically executable in the SAME current world.
    for bound in dispatch.actions:
        try:
            tools.get(bound.action.function_name).validate(
                bound,
                world,
            )
        except ToolValidationError as exc:
            raise DispatchValidationError(str(exc)) from exc

    # 2) One concrete exclusive device can start at most one atomic action.
    device_ids = [a.device_id for a in dispatch.actions]
    if len(device_ids) != len(set(device_ids)):
        raise DispatchValidationError(
            "parallel dispatch uses the same exclusive device twice"
        )

    # 3) Concurrent siblings may not depend on one another's effects and may
    #    not write conflicting semantic facts.
    for left, right in combinations(dispatch.actions, 2):
        left_tool = tools.get(left.action.function_name)
        right_tool = tools.get(right.action.function_name)

        left_reads = left_tool.read_facts(left)
        left_writes = left_tool.write_facts(left)
        right_reads = right_tool.read_facts(right)
        right_writes = right_tool.write_facts(right)

        if left_writes & right_reads:
            raise DispatchValidationError(
                f"{left.action.function_name} writes facts read by "
                f"{right.action.function_name}; must execute sequentially"
            )
        if right_writes & left_reads:
            raise DispatchValidationError(
                f"{right.action.function_name} writes facts read by "
                f"{left.action.function_name}; must execute sequentially"
            )
        if left_writes & right_writes:
            raise DispatchValidationError(
                "parallel actions write overlapping semantic facts"
            )

        if (
            left_tool.exclusive_entities(left.action)
            & right_tool.exclusive_entities(right.action)
        ):
            raise DispatchValidationError(
                "parallel actions manipulate the same exclusive entity"
            )

    # 4) Capacity-limited shared resources remain hard physical constraints.
    capacity_claims: dict[
        tuple[str, str],
        list[str],
    ] = {}

    for bound in dispatch.actions:
        bound_tool = tools.get(bound.action.function_name)
        for entity_id, capacity_field in (
            bound_tool.capacity_limited_entities(
                bound.action
            )
        ):
            capacity_claims.setdefault(
                (entity_id, capacity_field),
                [],
            ).append(bound.action.function_name)

    for (entity_id, capacity_field), claimers in capacity_claims.items():
        raw_capacity = world.get(entity_id, capacity_field)
        if isinstance(raw_capacity, bool):
            capacity = 1
        elif isinstance(raw_capacity, int):
            capacity = raw_capacity
        else:
            capacity = 1
        capacity = max(1, capacity)

        if len(claimers) > capacity:
            raise DispatchValidationError(
                "PARALLEL_CAPACITY_EXCEEDED: "
                f"{entity_id}.{capacity_field}={capacity}, "
                f"claims={len(claimers)} from {claimers}"
            )

    # 5) Explicit maintain-until constraints remain hard because they encode
    #    ongoing physical/safety coordination (e.g. a hand must keep holding
    #    a door while another hand places an object).
    active = [
        c for c in coordination.maintain_until
        if c.is_active(world, goals)
    ]
    if active:
        simulated = _simulate_dispatch(
            dispatch,
            world=world,
            tools=tools,
        )
        for constraint in active:
            goal_done_after = goals.by_id(
                constraint.until_goal
            ).is_satisfied(simulated)
            if goal_done_after:
                continue
            actual = simulated.get(
                constraint.subject,
                constraint.field,
            )
            if actual != constraint.value:
                raise DispatchValidationError(
                    "MAINTAIN_CONSTRAINT_VIOLATION: "
                    f"{constraint.subject}.{constraint.field} "
                    f"must stay {constraint.value!r} until goal "
                    f"{constraint.until_goal}"
                )
