"""
UI-driven Manual Runtime executor.

Human confirmation is the manual manipulation commit point.
Observation is supplied by dependency injection so this interface
module does not depend on services.plan_service.
"""

from __future__ import annotations

from ..domain.action import (
    ActionExecutionResult,
    DispatchExecutionReport,
)
from ..adapters.executor import FakeDispatchExecutor
from .manual_session import (
    ManualRuntimeCancelled,
    _ManualRuntimeSession,
    _manual_now,
    _manual_world_summary,
    _manual_short_error,
)
from ..domain.world import WorldState
from copy import deepcopy
from ..engine.guards.validation import validate_dispatch


def _manual_action_summary(
    bound_dict: dict[str, Any],
) -> str:
    """
    Human-readable proposed action shown BEFORE PRE-COMMIT validation.

    This wording is intentionally neutral: it describes what Runtime wants to
    do, but does not instruct the human to execute it yet.
    """
    actions = bound_dict.get(
        "actions",
        [],
    )

    if not actions:
        return "Runtime 想做：未知動作"

    action = actions[0]
    function_name = action.get(
        "function_name"
    )
    arguments = action.get(
        "arguments",
        {},
    )
    device_id = action.get(
        "device_id"
    )

    if function_name == "pick_object":
        text = (
            "Runtime 想做：拿起 "
            f"{arguments.get('object_id')}"
        )
    elif function_name == "place_object":
        text = (
            "Runtime 想做：把 "
            f"{arguments.get('object_id')} "
            "放到／放進 "
            f"{arguments.get('destination_id')}"
        )
    elif function_name in {
        "open_container",
        "open_trash_can",
    }:
        target = (
            arguments.get(
                "container_id"
            )
            or arguments.get(
                "object_id"
            )
        )
        text = (
            f"Runtime 想做：打開 {target}"
        )
    elif function_name in {
        "close_container",
        "close_trash_can",
    }:
        target = (
            arguments.get(
                "container_id"
            )
            or arguments.get(
                "object_id"
            )
        )
        text = (
            f"Runtime 想做：關閉 {target}"
        )
    else:
        text = (
            "Runtime 想做："
            f"{function_name} {arguments}"
        )

    if (
        isinstance(
            device_id,
            str,
        )
        and device_id
    ):
        text += (
            f"\n綁定裝置：{device_id}"
        )

    return text


def _manual_instruction(
    bound_dict: dict[str, Any],
) -> str:
    actions = bound_dict.get(
        "actions",
        [],
    )

    if not actions:
        return "請手動完成 Runtime 指定的動作。"

    action = actions[0]
    function_name = action.get(
        "function_name"
    )
    arguments = action.get(
        "arguments",
        {},
    )
    device_id = action.get(
        "device_id"
    )

    if function_name == "pick_object":
        base = (
            f"請手動拿起 {arguments.get('object_id')}，"
            "並移離原本位置，模擬機械手已經夾住它。"
        )
    elif function_name == "place_object":
        base = (
            f"請手動把 {arguments.get('object_id')} "
            f"放到／放進 {arguments.get('destination_id')}。"
        )
    elif function_name in {
        "open_container",
        "open_trash_can",
    }:
        target = (
            arguments.get(
                "container_id"
            )
            or arguments.get(
                "object_id"
            )
        )
        base = f"請手動打開 {target}。"
    elif function_name in {
        "close_container",
        "close_trash_can",
    }:
        target = (
            arguments.get(
                "container_id"
            )
            or arguments.get(
                "object_id"
            )
        )
        base = f"請手動關閉 {target}。"
    else:
        base = (
            "請手動完成："
            f"{function_name} {arguments}"
        )

    if (
        isinstance(
            device_id,
            str,
        )
        and device_id
    ):
        base += (
            f"\nRuntime 綁定裝置：{device_id}"
        )

    return base


def _manual_reconcile_holding_state(
    world: WorldState,
) -> tuple[
    WorldState,
    list[dict[str, Any]],
]:
    """
    Keep the two sides of the manual-mode holding relation consistent:

        object.held_by == arm_id
        arm.holding == object_id

    This is only a manual-executor semantic repair.  It does not infer physical
    grasp success and is not used by the production physical executor.
    """

    snapshot = world.snapshot()
    entities = snapshot.get(
        "entities",
        {},
    )

    if not isinstance(
        entities,
        dict,
    ):
        return world, []

    changes: list[
        dict[str, Any]
    ] = []

    held_objects_by_arm: dict[
        str,
        list[str],
    ] = {}

    for entity_id, entity in entities.items():
        if not isinstance(
            entity,
            dict,
        ):
            continue

        held_by = entity.get(
            "held_by"
        )

        if not isinstance(
            held_by,
            str,
        ) or not held_by:
            continue

        arm = entities.get(
            held_by
        )

        if not isinstance(
            arm,
            dict,
        ) or arm.get(
            "type"
        ) != "robot_arm":
            continue

        held_objects_by_arm.setdefault(
            held_by,
            [],
        ).append(
            entity_id
        )

    for arm_id, object_ids in held_objects_by_arm.items():
        if len(
            object_ids
        ) > 1:
            raise RuntimeError(
                "manual semantic state inconsistent: "
                f"{arm_id} has multiple objects with held_by={arm_id}: "
                f"{object_ids!r}"
            )

        object_id = object_ids[
            0
        ]
        arm = entities[
            arm_id
        ]
        current = arm.get(
            "holding"
        )

        if current in {
            None,
            "",
        }:
            arm[
                "holding"
            ] = object_id
            changes.append({
                "subject":
                    arm_id,
                "field":
                    "holding",
                "old":
                    current,
                "new":
                    object_id,
                "reason":
                    "reconcile_from_object.held_by",
            })
        elif current != object_id:
            raise RuntimeError(
                "manual semantic state inconsistent: "
                f"{arm_id}.holding={current!r} but "
                f"{object_id}.held_by={arm_id!r}"
            )

    for arm_id, arm in entities.items():
        if (
            not isinstance(
                arm,
                dict,
            )
            or arm.get(
                "type"
            ) != "robot_arm"
        ):
            continue

        object_id = arm.get(
            "holding"
        )

        if not isinstance(
            object_id,
            str,
        ) or not object_id:
            continue

        obj = entities.get(
            object_id
        )

        if not isinstance(
            obj,
            dict,
        ):
            continue

        held_by = obj.get(
            "held_by"
        )

        if held_by in {
            None,
            "",
        }:
            obj[
                "held_by"
            ] = arm_id
            changes.append({
                "subject":
                    object_id,
                "field":
                    "held_by",
                "old":
                    held_by,
                "new":
                    arm_id,
                "reason":
                    "reconcile_from_arm.holding",
            })
        elif held_by != arm_id:
            raise RuntimeError(
                "manual semantic state inconsistent: "
                f"{arm_id}.holding={object_id!r} but "
                f"{object_id}.held_by={held_by!r}"
            )

    return WorldState(
        snapshot
    ), changes


def _manual_carry_semantic_state(
    *,
    semantic_world: WorldState,
    observed_world: WorldState,
) -> WorldState:
    """
    Preserve only manual semantic facts that perception cannot yet verify.
    Fresh visual facts remain authoritative.
    """

    semantic = semantic_world.snapshot()
    observed = observed_world.snapshot()

    old_entities = semantic.get(
        "entities",
        {},
    )
    new_entities = observed.get(
        "entities",
        {},
    )

    if (
        not isinstance(
            old_entities,
            dict,
        )
        or not isinstance(
            new_entities,
            dict,
        )
    ):
        return observed_world

    semantic_fields = (
        "held_by",
        "location",
        "holding",
        "controlling",
    )

    for entity_id, current in new_entities.items():
        if not isinstance(
            current,
            dict,
        ):
            continue

        previous = old_entities.get(
            entity_id
        )

        if not isinstance(
            previous,
            dict,
        ):
            continue

        for field_name in semantic_fields:
            if field_name in previous:
                current[
                    field_name
                ] = deepcopy(
                    previous[
                        field_name
                    ]
                )

    for entity_id, previous in old_entities.items():
        if (
            entity_id in new_entities
            or not isinstance(
                previous,
                dict,
            )
        ):
            continue

        held_by = previous.get(
            "held_by"
        )
        location = previous.get(
            "location"
        )
        holding = previous.get(
            "holding"
        )
        controlling = previous.get(
            "controlling"
        )
        tags = previous.get(
            "tags"
        )
        affordances = previous.get(
            "affordances"
        )

        is_temporary_placement = (
            isinstance(
                tags,
                list,
            )
            and "temporary_placement"
            in tags
        ) or (
            isinstance(
                affordances,
                list,
            )
            and "temporary_placement"
            in affordances
        )

        keep = (
            isinstance(
                held_by,
                str,
            )
            and bool(
                held_by
            )
        ) or (
            isinstance(
                location,
                str,
            )
            and bool(
                location
            )
        ) or (
            isinstance(
                holding,
                str,
            )
            and bool(
                holding
            )
        ) or (
            isinstance(
                controlling,
                str,
            )
            and bool(
                controlling
            )
        ) or is_temporary_placement

        if not keep:
            continue

        carried = deepcopy(
            previous
        )

        for field_name in (
            "bbox",
            "position",
            "yaw_deg",
            "confidence",
            "grasp",
            "observed_at",
            "camera_source",
        ):
            carried.pop(
                field_name,
                None,
            )

        carried[
            "source"
        ] = "manual_confirmed_overlay"

        new_entities[
            entity_id
        ] = carried

    return WorldState(
        observed
    )


def _manual_blocked_report(
    bound,
    *,
    exc: Exception,
) -> DispatchExecutionReport:
    bound_actions = getattr(
        bound,
        "actions",
        None,
    )

    if not bound_actions:
        raise RuntimeError(
            "BoundDispatch has no actions"
        )

    feedback = getattr(
        exc,
        "feedback",
        None,
    )

    raw_result = {
        "status":
            "blocked",
        "reason":
            "PRECONDITION_CHANGED",
        "error_type":
            type(
                exc
            ).__name__,
        "message":
            str(
                exc
            ),
    }

    if isinstance(
        feedback,
        dict,
    ):
        raw_result[
            "feedback"
        ] = deepcopy(
            feedback
        )

    result = ActionExecutionResult(
        bound_action=
            bound_actions[
                0
            ],
        command_success=
            False,
        verified_success=
            False,
        error=
            "PRECOMMIT_VALIDATION_FAILED",
        raw_result=
            raw_result,
    )

    return DispatchExecutionReport(
        results=[
            result,
        ],
    )


class ManualUIRuntimeExecutor:
    """
    UI-driven manual manipulation executor.

    Observation-pose movement and sensing are real. Manipulation is performed
    by the human. Confirmation is the manual commit point; ToolSpec effects are
    then applied by FakeDispatchExecutor solely as a semantic overlay.
    """

    def __init__(
        self,
        *,
        session: _ManualRuntimeSession,
        observe_dispatch: Any,
    ):
        self.session = session
        self.fake = FakeDispatchExecutor(
            failure_probability=0.0
        )
        self.observe_dispatch = observe_dispatch

    def _check_cancelled(
        self,
    ) -> None:
        with self.session.lock:
            cancelled = (
                self.session
                .cancel_requested
            )

        if cancelled:
            raise ManualRuntimeCancelled(
                "manual Runtime cancelled by user"
            )

    def _wait_for_human(
        self,
        *,
        step_number: int,
        bound_dict: dict[str, Any],
    ) -> None:
        instruction = _manual_instruction(
            bound_dict
        )

        self.session.continue_event.clear()
        self.session.update_step(
            step_number,
            status=
                "awaiting_confirmation",
            gate_status=
                "awaiting_human",
            instruction=
                instruction,
        )
        self.session.update(
            status=
                "awaiting_confirmation",
            stage=
                "manual_action",
            instruction=
                instruction,
            pending_action=
                deepcopy(
                    bound_dict
                ),
            pending_step=
                step_number,
            proposed_state=
                "awaiting_human",
            proposed_reason=
                None,
            message=
                "PRE-COMMIT 已通過。請依照指示手動完成動作，完成後按確認。",
        )
        self.session.record(
            "manual_action_requested",
            message=
                instruction,
            data={
                "step":
                    step_number,
                "action":
                    bound_dict,
            },
        )

        self.session.continue_event.wait()
        self._check_cancelled()

        self.session.update_step(
            step_number,
            status=
                "human_confirmed",
            gate_status=
                "human_confirmed",
            confirmed_at=
                _manual_now(),
        )
        self.session.update(
            status=
                "running",
            stage=
                "semantic_commit",
            instruction=
                None,
            pending_action=
                None,
            proposed_state=
                "human_confirmed",
            message=
                "已收到人工確認，正在套用 manual semantic effect。",
        )
        self.session.record(
            "manual_action_confirmed",
            data={
                "step":
                    step_number,
            },
        )

    def execute(
        self,
        bound,
        *,
        world: WorldState,
        tools,
        goals=None,
        coordination=None,
    ):
        self._check_cancelled()

        bound_dict = bound.as_dict()
        step_number = self.session.begin_step(
            action=
                bound_dict,
            world_before=
                world,
        )

        proposal = _manual_action_summary(
            bound_dict
        )

        self.session.update_step(
            step_number,
            proposal=
                proposal,
            gate_status=
                "checking",
        )
        self.session.update(
            status=
                "running",
            stage=
                "precommit_observation",
            pending_step=
                step_number,
            proposed_action=
                deepcopy(
                    bound_dict
                ),
            proposed_instruction=
                proposal,
            proposed_step=
                step_number,
            proposed_state=
                "checking",
            proposed_reason=
                None,
            message=(
                f"Step {step_number}: Runtime 已選擇下一個動作；"
                "正在執行 PRE-COMMIT fresh observation / validation。"
            ),
        )
        self.session.record(
            "action_proposed",
            message=
                proposal,
            data={
                "step":
                    step_number,
                "action":
                    bound_dict,
            },
        )
        self.session.record(
            "step_started",
            data={
                "step":
                    step_number,
                "action":
                    bound_dict,
                "world_before":
                    _manual_world_summary(
                        world
                    ),
            },
        )

        raw_fresh_world = self.observe_dispatch(
            bound,
            world,
        )

        # PRE-COMMIT observation is visually fresh, but fields such as
        # object.held_by / arm.holding are manual semantic state and are not
        # produced by the physical perception stack.  Preserve those facts
        # before validating the selected action; otherwise a fresh arm_service
        # entity can erase arm.holding while object.held_by is still carried,
        # causing a false PHYSICAL_PRECONDITION_FAILED for place_object.
        fresh_world = _manual_carry_semantic_state(
            semantic_world=
                world,
            observed_world=
                raw_fresh_world,
        )
        fresh_world, precommit_reconcile_changes = (
            _manual_reconcile_holding_state(
                fresh_world
            )
        )

        self.session.update_step(
            step_number,
            status=
                "precommit_validation",
            precommit_observation_raw=
                _manual_world_summary(
                    raw_fresh_world
                ),
            precommit_world=
                _manual_world_summary(
                    fresh_world
                ),
            precommit_reconcile_changes=
                precommit_reconcile_changes,
        )
        self.session.update(
            latest_world=
                fresh_world.snapshot(),
            stage=
                "precommit_validation",
            message=(
                f"Step {step_number}: "
                "fresh observation 完成，正在重新驗證 preconditions。"
            ),
        )
        self.session.record(
            "precommit_observed",
            data={
                "step":
                    step_number,
                "raw_world":
                    _manual_world_summary(
                        raw_fresh_world
                    ),
                "world":
                    _manual_world_summary(
                        fresh_world
                    ),
                "reconcile_changes":
                    precommit_reconcile_changes,
            },
        )

        if goals is not None:
            try:
                validate_dispatch(
                    bound,
                    world=
                        fresh_world,
                    goals=
                        goals,
                    tools=
                        tools,
                    coordination=
                        coordination,
                )
            except Exception as exc:
                error_row = {
                    "type":
                        type(
                            exc
                        ).__name__,
                    "message":
                        str(
                            exc
                        ),
                    "feedback":
                        deepcopy(
                            getattr(
                                exc,
                                "feedback",
                                None,
                            )
                        ),
                }
                blocked_reason = _manual_short_error(
                    exc
                )

                self.session.update_step(
                    step_number,
                    status=
                        "blocked",
                    gate_status=
                        "blocked",
                    blocked_reason=
                        blocked_reason,
                    completed_at=
                        _manual_now(),
                    error=
                        error_row,
                    world_after=
                        _manual_world_summary(
                            fresh_world
                        ),
                )
                self.session.update(
                    status=
                        "running",
                    stage=
                        "precommit_blocked",
                    proposed_state=
                        "blocked",
                    proposed_reason=
                        blocked_reason,
                    message=(
                        f"Step {step_number}: PRE-COMMIT 驗證失敗。"
                        "上方仍會顯示 Runtime 原本想做的動作；"
                        "此動作沒有要求人類執行，Runtime 將以 fresh state 重規劃。"
                    ),
                )
                self.session.record(
                    "precommit_blocked",
                    message=
                        _manual_short_error(
                            exc
                        ),
                    data={
                        "step":
                            step_number,
                        "error":
                            error_row,
                    },
                )
                return (
                    fresh_world,
                    _manual_blocked_report(
                        bound,
                        exc=
                            exc,
                    ),
                )

        self.session.update_step(
            step_number,
            gate_status=
                "validated",
        )
        self.session.update(
            proposed_state=
                "validated",
            proposed_reason=
                None,
        )
        self.session.record(
            "precommit_validation_passed",
            data={
                "step":
                    step_number,
            },
        )

        self._wait_for_human(
            step_number=
                step_number,
            bound_dict=
                bound_dict,
        )

        self._check_cancelled()

        semantic_world_raw, report = self.fake.execute(
            bound,
            world=
                fresh_world,
            tools=
                tools,
        )

        semantic_world, reconcile_changes = (
            _manual_reconcile_holding_state(
                semantic_world_raw
            )
        )

        self.session.update_step(
            step_number,
            status=
                "postcommit_observation",
            semantic_world_raw=
                _manual_world_summary(
                    semantic_world_raw
                ),
            semantic_world=
                _manual_world_summary(
                    semantic_world
                ),
            reconcile_changes=
                reconcile_changes,
        )
        self.session.update(
            stage=
                "postcommit_observation",
            latest_world=
                semantic_world.snapshot(),
            message=(
                f"Step {step_number}: human commit 完成，"
                "正在執行 fresh POST-COMMIT observation。"
            ),
        )
        self.session.record(
            "semantic_effect_applied",
            data={
                "step":
                    step_number,
                "raw_world":
                    _manual_world_summary(
                        semantic_world_raw
                    ),
                "reconciled_world":
                    _manual_world_summary(
                        semantic_world
                    ),
                "reconcile_changes":
                    reconcile_changes,
            },
        )

        observed_world = self.observe_dispatch(
            bound,
            semantic_world,
            report,
        )

        updated_world = _manual_carry_semantic_state(
            semantic_world=
                semantic_world,
            observed_world=
                observed_world,
        )

        updated_world, post_reconcile_changes = (
            _manual_reconcile_holding_state(
                updated_world
            )
        )

        all_reconcile_changes = (
            list(
                reconcile_changes
            )
            + list(
                post_reconcile_changes
            )
        )

        self.session.update_step(
            step_number,
            status=
                "completed",
            gate_status=
                "completed",
            completed_at=
                _manual_now(),
            post_observation_raw=
                _manual_world_summary(
                    observed_world
                ),
            world_after=
                _manual_world_summary(
                    updated_world
                ),
            reconcile_changes=
                all_reconcile_changes,
        )
        self.session.update(
            latest_world=
                updated_world.snapshot(),
            status=
                "running",
            stage=
                "runtime",
            proposed_state=
                "completed",
            proposed_reason=
                None,
            message=(
                f"Step {step_number}: POST-COMMIT observation 完成，"
                "Runtime 繼續。"
            ),
        )
        self.session.record(
            "step_completed",
            data={
                "step":
                    step_number,
                "post_observation_raw":
                    _manual_world_summary(
                        observed_world
                    ),
                "world_after":
                    _manual_world_summary(
                        updated_world
                    ),
                "reconcile_changes":
                    all_reconcile_changes,
            },
        )

        return (
            updated_world,
            report,
        )


__all__ = [
    "ManualUIRuntimeExecutor",
    "_manual_action_summary",
    "_manual_instruction",
    "_manual_reconcile_holding_state",
    "_manual_carry_semantic_state",
    "_manual_blocked_report",
]
