"""
Compatibility shim.

Canonical implementation:
    agent.runtime.adapters.provider
"""

from .adapters.provider import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_CTX,
    DEFAULT_TIMEOUT,
    DEFAULT_KEEP_ALIVE,
    StructuredProviderError,
    resolve_provider,
    chat_structured,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_NUM_CTX",
    "DEFAULT_TIMEOUT",
    "DEFAULT_KEEP_ALIVE",
    "StructuredProviderError",
    "resolve_provider",
    "chat_structured",
]
