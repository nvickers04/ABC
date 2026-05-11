"""
Grok LLM Integration — xAI SDK (Native gRPC)

Uses the official xai-sdk (>= 1.8.0) for all model communication.
The agent loop in agent.py uses self.grok.client.chat.create() to build
conversations and chat.sample() to get responses.

**API model slugs live only here** (`REASONING_MODEL`, `MULTI_AGENT_MODEL`).
Elsewhere (docs, prompts, logs) refer generically to “Grok” so a version
change is a single edit plus `data/cost_tracker.py` pricing if xAI changes
rates. Slug validity: https://docs.x.ai/docs/models

Models:
    REASONING_MODEL   — single-agent ReAct trading loop (client-side tools)
    MULTI_AGENT_MODEL — multi-agent research (built-in web_search / x_search)
"""

import logging
import os
from typing import Optional

from xai_sdk import AsyncClient

logger = logging.getLogger(__name__)

# ── xAI API model slugs (edit here when xAI renames or you switch tiers) ──
REASONING_MODEL = "grok-4.3"
MULTI_AGENT_MODEL = "grok-4.20-multi-agent"


class GrokLLM:
    """Direct Grok LLM integration via xAI SDK (AsyncClient)."""

    def __init__(
        self,
        model: str = REASONING_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not api_key:
            logger.warning("No XAI_API_KEY or GROK_API_KEY found in environment")

        self.client = AsyncClient(api_key=api_key)

        logger.info(f"GrokLLM initialized (model={model}, temp={temperature})")


# Singleton
_grok_llm: Optional[GrokLLM] = None


def get_grok_llm() -> GrokLLM:
    """Get or create GrokLLM singleton."""
    global _grok_llm
    if _grok_llm is None:
        _grok_llm = GrokLLM()
    return _grok_llm


__all__ = [
    "GrokLLM",
    "get_grok_llm",
    "REASONING_MODEL",
    "MULTI_AGENT_MODEL",
]
