"""Query rewriters: LLM providers and the rule-based baseline."""

from __future__ import annotations

from ..config import LLMSettings
from .base import LLMError, RepairItem, RewriteRequest, Rewriter
from .rule_based import RuleBasedRewriter


def create_rewriter(settings: LLMSettings) -> Rewriter:
    provider = settings.provider.lower()
    if provider in ("rule_based", "rules", "rule"):
        return RuleBasedRewriter()
    if provider in ("anthropic", "claude"):
        from .anthropic_provider import AnthropicRewriter

        return AnthropicRewriter(settings)
    if provider == "groq":
        from .groq_provider import GroqRewriter

        return GroqRewriter(settings)
    if provider == "ollama":
        from .ollama_provider import OllamaRewriter

        return OllamaRewriter(settings)
    raise ValueError(f"Unknown LLM provider {settings.provider!r} (expected rule_based | anthropic | groq | ollama)")


__all__ = ["LLMError", "RepairItem", "RewriteRequest", "Rewriter", "RuleBasedRewriter", "create_rewriter"]
