from __future__ import annotations

from typing import Optional

from ..config import LLMSettings
from .base import LLMClient, LLMError


def make_client(cfg: LLMSettings) -> Optional[LLMClient]:
    """Return a client for the configured provider, or None for offline mode / when the provider is not usable (missing key)."""
    if cfg.provider in ("", "none", "off", "offline"):
        return None
    try:
        if cfg.provider == "anthropic":
            from .anthropic_client import AnthropicClient
            return AnthropicClient(cfg)
        if cfg.provider in ("openai", "openai_compatible", "ollama", "vllm"):
            from .openai_client import OpenAIClient
            return OpenAIClient(cfg)
    except LLMError:
        return None
    return None


__all__ = ["LLMClient", "LLMError", "make_client"]
