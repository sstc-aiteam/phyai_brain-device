"""

Runtime execution layer.


The Executor receives an already validated BoundDispatch and starts the

corresponding physical callbacks.


Production rule:


    callback success != WorldState truth


After callbacks finish, the executor performs a fresh observation.  Declared

ToolSpec effects may be compared against that fresh WorldState for verification,

but they are never written into the canonical real-world state.


The semantic FakeDispatchExecutor is retained only for simulation/regression.

"""


from __future__ import annotations


from copy import deepcopy

from concurrent.futures import (

    Future,

    ThreadPoolExecutor,

    as_completed,

)

import random

from typing import Any, Callable, Protocol


from .dispatch import (

    ActionExecutionResult,

    BoundAction,

    BoundDispatch,

    DispatchExecutionReport,

)

from .tool_model import ToolCatalog

from .world_state import WorldState


class DispatchExecutor(Protocol):

    """Common executor contract."""


    def execute(

        self,

        dispatch: BoundDispatch,

        *,

        world: WorldState,

        tools: ToolCatalog,

    ) -> tuple[

        WorldState,

        DispatchExecutionReport,

    ]:

        ...


def _verify_declared_effects(

    bound: BoundAction,

    *,

    before: WorldState,

    observed: WorldState,

    tools: ToolCatalog,

) -> bool:

    """

    Compare the tool's declared semantic effects with a FRESH observation.


    `ToolSpec.apply_effects()` is used only to calculate expected values.

    The predicted WorldState is never returned as canonical truth here.


    If a tool declares no writable semantic facts, verification is vacuously

    true after a successful command because the tool made no semantic claim

    that this layer can check.

    """


    tool = tools.get(

        bound.action.function_name

    )


    write_facts = tool.write_facts(

        bound

    )


    if not write_facts:

        return True


    expected = tool.apply_effects(

        before,

        bound,

    )


    for subject, field in write_facts:

        if (

            observed.get(

                subject,

                field,

            )

            != expected.get(

                subject,

                field,

            )

        ):

            return False


    return True


class FakeDispatchExecutor:

    """

    Simulation/regression executor.


    In this executor only, ToolSpec effects define the simulated world's truth.

    This class must not be used as the production physical-world executor.

    """


    def __init__(

        self,

        *,

        failure_probability: float = 0.0,

        seed: int = 7,

    ):

        if not (

            0.0

            <= failure_probability

            <= 1.0

        ):

            raise ValueError(

                "failure_probability must be 0..1"

            )


        self.failure_probability = (

            float(

                failure_probability

            )

        )


        self.rng = random.Random(

            seed

        )


    def execute(

        self,

        dispatch: BoundDispatch,

        *,

        world: WorldState,

        tools: ToolCatalog,

    ) -> tuple[

        WorldState,

        DispatchExecutionReport,

    ]:

        updated = world.clone()


        report = DispatchExecutionReport()


        for bound in dispatch.actions:

            failed = (

                self.rng.random()

                < self.failure_probability

            )


            if failed:

                report.results.append(

                    ActionExecutionResult(

                        bound_action=bound,

                        command_success=False,

                        verified_success=False,

                        error=(

                            "simulated execution failure"

                        ),

                    )

                )

                continue


            tool = tools.get(

                bound.action.function_name

            )


            updated = tool.apply_effects(

                updated,

                bound,

            )


            report.results.append(

                ActionExecutionResult(

                    bound_action=bound,

                    command_success=True,

                    verified_success=True,

                )

            )


        return (

            updated,

            report,

        )


class CallbackExecutor:
    """
    Production-shaped concurrent callback executor.

    Normal flow:

        BoundDispatch
            ↓
        optional PRE-COMMIT fresh observation
            ↓
        optional deterministic PRE-COMMIT validation
            ↓
        invoke callbacks concurrently
            ↓
        DispatchExecutionReport(command status)
            ↓
        fresh POST-COMMIT observation
            ↓
        canonical observed WorldState
            ↓
        compare declared semantic effects
            ↓
        DispatchExecutionReport(verified status)

    The pre-commit hooks are generic and optional. Existing callers that only
    provide `observe_world` retain the previous behavior.

    The callback return value is retained as raw_result, but is not itself
    treated as semantic world truth.
    """

    def __init__(
        self,
        callbacks: dict[
            str,
            Callable[
                [
                    str,
                    dict[str, Any],
                ],
                Any,
            ],
        ],
        *,
        observe_world: Callable[
            [
                WorldState,
                DispatchExecutionReport,
            ],
            WorldState,
        ],
        max_workers: int = 4,
        argument_resolver: Callable[
            [
                BoundAction,
                WorldState,
            ],
            dict[str, Any],
        ] | None = None,
        precommit_observe: Callable[
            [
                BoundDispatch,
                WorldState,
            ],
            WorldState,
        ] | None = None,
        precommit_validate: Callable[
            [
                BoundDispatch,
                WorldState,
                ToolCatalog,
            ],
            None,
        ] | None = None,
        postcommit_observe: Callable[
            [
                BoundDispatch,
                WorldState,
                DispatchExecutionReport,
            ],
            WorldState,
        ] | None = None,
    ):
        if max_workers < 1:
            raise ValueError(
                "max_workers must be >= 1"
            )

        self.callbacks = dict(
            callbacks
        )

        self.observe_world = (
            observe_world
        )

        self.max_workers = int(
            max_workers
        )

        self.argument_resolver = (
            argument_resolver
        )

        self.precommit_observe = (
            precommit_observe
        )

        self.precommit_validate = (
            precommit_validate
        )

        self.postcommit_observe = (
            postcommit_observe
        )

    def _invoke(
        self,
        bound: BoundAction,
        world: WorldState,
    ) -> Any:
        callback = self.callbacks.get(
            bound.action.function_name
        )

        if callback is None:
            raise RuntimeError(
                "No executor callback for "
                f"{bound.action.function_name}"
            )

        if self.argument_resolver is None:
            arguments = dict(
                bound.action.arguments
            )

        else:
            arguments = (
                self.argument_resolver(
                    bound,
                    world,
                )
            )

            if not isinstance(
                arguments,
                dict,
            ):
                raise TypeError(
                    "argument_resolver must "
                    "return dict"
                )

        return callback(
            bound.device_id,
            arguments,
        )

    def _execute_callbacks(
        self,
        dispatch: BoundDispatch,
        *,
        world: WorldState,
    ) -> DispatchExecutionReport:
        """
        Execute all callbacks while preserving dispatch action order in report.
        """

        count = len(
            dispatch.actions
        )

        if count == 0:
            return (
                DispatchExecutionReport()
            )

        results: list[
            ActionExecutionResult | None
        ] = [
            None
        ] * count

        future_rows: dict[
            Future[Any],
            tuple[
                int,
                BoundAction,
            ],
        ] = {}

        with ThreadPoolExecutor(
            max_workers=min(
                self.max_workers,
                count,
            )
        ) as pool:
            for index, bound in enumerate(
                dispatch.actions
            ):
                future = pool.submit(
                    self._invoke,
                    bound,
                    world.clone(),
                )

                future_rows[
                    future
                ] = (
                    index,
                    bound,
                )

            for future in as_completed(
                future_rows
            ):
                index, bound = (
                    future_rows[
                        future
                    ]
                )

                try:
                    raw_result = (
                        future.result()
                    )

                    # Command transport/callback succeeded.
                    # Semantic verification happens only after observation.
                    results[index] = (
                        ActionExecutionResult(
                            bound_action=bound,
                            command_success=True,
                            verified_success=False,
                            raw_result=raw_result,
                        )
                    )

                except Exception as exc:
                    results[index] = (
                        ActionExecutionResult(
                            bound_action=bound,
                            command_success=False,
                            verified_success=False,
                            error=str(
                                exc
                            ),
                        )
                    )

        report = (
            DispatchExecutionReport()
        )

        report.results.extend(
            result
            for result in results
            if result is not None
        )

        return report

    def _blocked_report(
        self,
        dispatch: BoundDispatch,
        *,
        reason: str,
        error: Exception | str,
    ) -> DispatchExecutionReport:
        """
        Build a report for an action that never crossed the commit point.

        `command_success=False` is intentional: no physical callback started.
        Structured feedback is retained in raw_result for Coordinator/repair
        logic without importing Validator types into Executor.
        """

        report = (
            DispatchExecutionReport()
        )

        message = str(
            error
        )

        feedback = getattr(
            error,
            "feedback",
            None,
        )

        if not isinstance(
            feedback,
            dict,
        ):
            feedback = {}

        for bound in dispatch.actions:
            report.results.append(
                ActionExecutionResult(
                    bound_action=bound,
                    command_success=False,
                    verified_success=False,
                    error=(
                        f"PRECOMMIT_{reason}: "
                        f"{message}"
                    ),
                    raw_result={
                        "status":
                            "blocked",

                        "phase":
                            "pre_commit",

                        "reason":
                            reason,

                        "message":
                            message,

                        "feedback":
                            deepcopy(
                                feedback
                            ),
                    },
                )
            )

        return report

    def _verify_report(
        self,
        report: DispatchExecutionReport,
        *,
        before: WorldState,
        observed: WorldState,
        tools: ToolCatalog,
    ) -> DispatchExecutionReport:
        """
        Rebuild report with semantic verification based on fresh observation.
        """

        verified = (
            DispatchExecutionReport()
        )

        for result in report.results:
            if not result.command_success:
                verified.results.append(
                    result
                )
                continue

            semantic_success = (
                _verify_declared_effects(
                    result.bound_action,
                    before=before,
                    observed=observed,
                    tools=tools,
                )
            )

            verified.results.append(
                ActionExecutionResult(
                    bound_action=
                        result.bound_action,

                    command_success=True,

                    verified_success=
                        semantic_success,

                    error=(
                        None
                        if semantic_success
                        else (
                            "declared semantic effects "
                            "not confirmed by fresh "
                            "observation"
                        )
                    ),

                    raw_result=
                        result.raw_result,
                )
            )

        return verified

    def execute(
        self,
        dispatch: BoundDispatch,
        *,
        world: WorldState,
        tools: ToolCatalog,
    ) -> tuple[
        WorldState,
        DispatchExecutionReport,
    ]:
        if not isinstance(
            dispatch,
            BoundDispatch,
        ):
            raise TypeError(
                "dispatch must be BoundDispatch"
            )

        if not isinstance(
            world,
            WorldState,
        ):
            raise TypeError(
                "world must be WorldState"
            )

        if not isinstance(
            tools,
            ToolCatalog,
        ):
            raise TypeError(
                "tools must be ToolCatalog"
            )

        # The incoming world was already validated by Coordinator.
        # PRE-COMMIT observation may refresh it before any physical callback.
        commit_world = (
            world.clone()
        )

        if self.precommit_observe is not None:
            try:
                observed_before_commit = (
                    self.precommit_observe(
                        dispatch,
                        commit_world.clone(),
                    )
                )

                if not isinstance(
                    observed_before_commit,
                    WorldState,
                ):
                    raise TypeError(
                        "precommit_observe must "
                        "return WorldState"
                    )

                commit_world = (
                    observed_before_commit
                )

            except Exception as exc:
                return (
                    commit_world,
                    self._blocked_report(
                        dispatch,
                        reason=
                            "OBSERVATION_FAILED",
                        error=
                            exc,
                    ),
                )

        # Re-run deterministic legality against the fresh pre-commit world.
        # The callback is injected so RuntimeValidator remains the single
        # authority for semantic legality.
        if self.precommit_validate is not None:
            try:
                self.precommit_validate(
                    dispatch,
                    commit_world.clone(),
                    tools,
                )

            except Exception as exc:
                return (
                    commit_world,
                    self._blocked_report(
                        dispatch,
                        reason=
                            "PRECONDITION_CHANGED",
                        error=
                            exc,
                    ),
                )

        # COMMIT POINT: physical callbacks start only after fresh validation.
        before_commit = (
            commit_world.clone()
        )

        command_report = (
            self._execute_callbacks(
                dispatch,
                world=before_commit,
            )
        )

        # Re-observe regardless of individual callback failures.
        if self.postcommit_observe is not None:
            observed = (
                self.postcommit_observe(
                    dispatch,
                    before_commit.clone(),
                    command_report,
                )
            )

        else:
            observed = self.observe_world(
                before_commit.clone(),
                command_report,
            )

        if not isinstance(
            observed,
            WorldState,
        ):
            raise TypeError(
                "post-commit observer must "
                "return WorldState"
            )

        verified_report = (
            self._verify_report(
                command_report,
                before=before_commit,
                observed=observed,
                tools=tools,
            )
        )

        return (
            observed,
            verified_report,
        )


# Compatibility name for existing runtime imports.

ConcurrentCallbackExecutor = (

    CallbackExecutor

)


__all__ = [

    "DispatchExecutor",

    "FakeDispatchExecutor",

    "CallbackExecutor",

    "ConcurrentCallbackExecutor",

]
