"""
Backward-compatible Structured LLM Provider.

Canonical implementation:
    agent.providers.structured

Existing Runtime imports are preserved.
"""

from agent.providers.structured import (
    StructuredProviderError,
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_CTX,
    DEFAULT_TIMEOUT,
    DEFAULT_KEEP_ALIVE,
    resolve_provider,
    chat_structured,
)

__all__ = [
    "StructuredProviderError",
    "DEFAULT_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_NUM_CTX",
    "DEFAULT_TIMEOUT",
    "DEFAULT_KEEP_ALIVE",
    "resolve_provider",
    "chat_structured",
]
