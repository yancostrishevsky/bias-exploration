"""Compatibility shim for provider client imports."""

from ai_bias_search.providers.openrouter import OpenRouterClient, OpenRouterError

__all__ = ["OpenRouterClient", "OpenRouterError"]
