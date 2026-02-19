"""
Grok LLM Integration — xAI API Connection (Minimal)

Thin wrapper around xAI's OpenAI-compatible API.
The agent loop in agent.py calls self.grok.client directly.

Usage:
    llm = get_grok_llm()
    response = await llm.client.chat.completions.create(...)

Built for Grok 4.2 — dynamic liquidity, overnight holds OK, 0.5% risk max.
"""

import logging
import os
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# model="grok-4-1-fast-reasoning" → change to exact Grok 4.2 name when available in console
DEFAULT_MODEL = "grok-4-1-fast-reasoning"


class GrokLLM:
    """Direct Grok LLM integration for trading decisions."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not api_key:
            logger.warning("No XAI_API_KEY or GROK_API_KEY found in environment")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        logger.info(f"GrokLLM initialized (model={model}, temp={temperature})")


# Singleton
_grok_llm: Optional[GrokLLM] = None


def get_grok_llm() -> GrokLLM:
    """Get or create GrokLLM singleton."""
    global _grok_llm
    if _grok_llm is None:
        _grok_llm = GrokLLM()
    return _grok_llm


__all__ = ["GrokLLM", "get_grok_llm", "DEFAULT_MODEL"]


__all__ = ["GrokLLM", "get_grok_llm"]
