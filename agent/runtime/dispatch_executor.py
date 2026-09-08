from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import Any, Callable

from .dispatch_schema import (
    ActionExecutionResult,
    BoundAction,
    BoundDispatch,
    DispatchExecutionReport,
)
from .tool_model import ToolCatalog
from .world_state import WorldState


class FakeDispatchExecutor:
    """Semantic simulator with independent per-action execution failures.

    Successful actions apply predicted effects. Failed actions have no effect.
    Because validate_dispatch() forbids cross-action read/write conflicts,
    applying successful action effects sequentially is equivalent to the
    validated concurrent semantic transition.
    """

    def __init__(
        self,
        *,
        failure_probability: float = 0.0,
        seed: int = 7,
    ):
        if not 0.0 <= failure_probability <= 1.0:
            raise ValueError("failure_probability must be 0..1")
        self.failure_probability = failure_probability
        self.rng = random.Random(seed)

    def execute(
        self,
        dispatch: BoundDispatch,
        *,
        world: WorldState,
        tools: ToolCatalog,
    ) -> tuple[WorldState, DispatchExecutionReport]:
        updated = world.clone()
        report = DispatchExecutionReport()

        for bound in dispatch.actions:
            failed = self.rng.random() < self.failure_probability
            if failed:
                report.results.append(ActionExecutionResult(
                    bound_action=bound,
                    command_success=False,
                    verified_success=False,
                    error="simulated execution failure",
                ))
                continue

            tool = tools.get(bound.action.function_name)
            updated = tool.apply_effects(updated, bound)
            report.results.append(ActionExecutionResult(
                bound_action=bound,
                command_success=True,
                verified_success=True,
            ))

        return updated, report


class ConcurrentCallbackExecutor:
    """Production-shaped executor adapter.

    Tool callbacks are started concurrently.  Their return values are NOT
    trusted as semantic World truth.  ``observe_world`` must perform
    verification/perception and return a fresh WorldState after execution.
    """

    def __init__(
        self,
        callbacks: dict[
            str,
            Callable[[str, dict[str, Any]], Any],
        ],
        *,
        observe_world: Callable[
            [WorldState, DispatchExecutionReport],
            WorldState,
        ],
        max_workers: int = 4,
        argument_resolver: Callable[[BoundAction, WorldState], dict[str, Any]] | None = None,
    ):
        self.callbacks = dict(callbacks)
        self.observe_world = observe_world
        self.max_workers = max_workers
        self.argument_resolver = argument_resolver

    def _invoke(self, bound: BoundAction, world: WorldState) -> Any:
        callback = self.callbacks.get(
            bound.action.function_name
        )
        if callback is None:
            raise RuntimeError(
                f"No executor callback for "
                f"{bound.action.function_name}"
            )
        args = (
            self.argument_resolver(bound, world)
            if self.argument_resolver is not None
            else dict(bound.action.arguments)
        )
        return callback(bound.device_id, args)

    def execute(
        self,
        dispatch: BoundDispatch,
        *,
        world: WorldState,
        tools: ToolCatalog,
    ) -> tuple[WorldState, DispatchExecutionReport]:
        del tools  # real verification owns semantic update

        report = DispatchExecutionReport()
        futures = {}

        with ThreadPoolExecutor(
            max_workers=min(
                self.max_workers,
                len(dispatch.actions),
            )
        ) as pool:
            for bound in dispatch.actions:
                futures[pool.submit(self._invoke, bound, world.clone())] = bound

            for future in as_completed(futures):
                bound = futures[future]
                try:
                    raw = future.result()
                    report.results.append(ActionExecutionResult(
                        bound_action=bound,
                        command_success=True,
                        # Temporary: final semantic success is determined by
                        # observe_world. Keep command status separate.
                        verified_success=True,
                        raw_result=raw,
                    ))
                except Exception as exc:
                    report.results.append(ActionExecutionResult(
                        bound_action=bound,
                        command_success=False,
                        verified_success=False,
                        error=str(exc),
                    ))

        observed = self.observe_world(world.clone(), report)
        if not isinstance(observed, WorldState):
            raise TypeError(
                "observe_world must return WorldState"
            )
        return observed, report
