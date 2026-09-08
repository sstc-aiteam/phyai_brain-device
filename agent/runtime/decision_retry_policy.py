from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionRetryPolicy:
    """Domain-agnostic dispatch liveness policy.

    Parallelism is an optimization.  Every new World iteration starts at the
    configured parallel width.  Only after repeated rejected proposals in the
    SAME iteration does the runtime temporarily lower the width.

    This policy does not choose actions and does not generate candidates.
    """

    max_attempts: int = 5
    parallel_fallback_after_rejections: int = 2
    fallback_max_parallel_actions: int = 1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.parallel_fallback_after_rejections < 0:
            raise ValueError(
                "parallel_fallback_after_rejections must be >= 0"
            )
        if self.fallback_max_parallel_actions < 1:
            raise ValueError(
                "fallback_max_parallel_actions must be >= 1"
            )

    def parallel_limit(
        self,
        *,
        configured_max_parallel_actions: int,
        rejected_count: int,
    ) -> tuple[int, bool]:
        if configured_max_parallel_actions < 1:
            raise ValueError(
                "configured_max_parallel_actions must be >= 1"
            )

        fallback_limit = min(
            configured_max_parallel_actions,
            self.fallback_max_parallel_actions,
        )
        fallback_active = (
            configured_max_parallel_actions > fallback_limit
            and rejected_count
            >= self.parallel_fallback_after_rejections
        )

        if fallback_active:
            return fallback_limit, True
        return configured_max_parallel_actions, False
