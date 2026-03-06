"""
Grok LLM Integration — xAI SDK (Native gRPC)

Uses the official xai-sdk (>= 1.8.0) for all model communication.
The agent loop in agent.py uses self.grok.client.chat.create() to build
conversations and chat.sample() to get responses.

Models:
    REASONING_MODEL  — single-agent with chain-of-thought (trading loop)
    MULTI_AGENT_MODEL — 4/16 agent research swarm (built-in tools only)

Usage:
    llm = get_grok_llm()
    chat = llm.client.chat.create(model=llm.model, messages=[...])
    response = await chat.sample()
    print(response.content)
"""

import logging
import os
from typing import Optional

from xai_sdk import AsyncClient

logger = logging.getLogger(__name__)

# ── Model Slugs (beta — may change without notice) ─────────────
REASONING_MODEL = "grok-4.20-experimental-beta-0304-reasoning"
MULTI_AGENT_MODEL = "grok-4.20-multi-agent-experimental-beta-0304"


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
