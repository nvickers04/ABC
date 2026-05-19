"""
Grok LLM Integration — xAI SDK (Native gRPC)

Uses the official xai-sdk (>= 1.8.0) for all model communication.
The agent loop in agent.py uses self.grok.client.chat.create() to build
conversations and chat.sample() to get responses.

**API model slugs live in** :mod:`core.prompt_config` (``PromptConfig.reasoning_model``
and ``PromptConfig.multi_agent_model``). Elsewhere refer generically to “Grok”.
Slug validity: https://docs.x.ai/docs/models

Models:
    reasoning_model   — single-agent ReAct trading loop (client-side tools)
    multi_agent_model — multi-agent research (built-in web_search / x_search)
"""

import logging
import os
from typing import Optional

from xai_sdk import AsyncClient

from core.prompt_config import get_prompt_config

logger = logging.getLogger(__name__)


class GrokLLM:
    """Direct Grok LLM integration via xAI SDK (AsyncClient)."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        pc = get_prompt_config()
        self.model = model or pc.reasoning_model
        self.temperature = pc.llm_temperature if temperature is None else temperature
        self.max_tokens = pc.llm_max_tokens if max_tokens is None else max_tokens

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not api_key:
            logger.warning("No XAI_API_KEY or GROK_API_KEY found in environment")

        self.client = AsyncClient(api_key=api_key)

        logger.info(f"GrokLLM initialized (model={self.model}, temp={self.temperature})")


# Singleton
_grok_llm: Optional[GrokLLM] = None


def get_grok_llm() -> GrokLLM:
    """Get or create GrokLLM singleton."""
    global _grok_llm
    if _grok_llm is None:
        _grok_llm = GrokLLM()
    return _grok_llm


def get_reasoning_model() -> str:
    """Return the configured reasoning model slug."""
    return get_prompt_config().reasoning_model


def get_multi_agent_model() -> str:
    """Return the configured multi-agent research model slug."""
    return get_prompt_config().multi_agent_model


__all__ = [
    "GrokLLM",
    "get_grok_llm",
    "get_reasoning_model",
    "get_multi_agent_model",
]
