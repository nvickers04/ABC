"""
Multi-Agent Research Tool — uses grok-4.20-multi-agent for web/X research.

The trading agent's reasoning model calls research(query) like any other tool.
Internally, this fires 4 multi-agent workers with web_search + x_search,
waits for the synthesized answer, and returns it as plain text.

No custom tools — multi-agent only supports built-in server-side tools.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


async def handle_research(executor, params: dict) -> Any:
    """
    Run a multi-agent research query using web_search + x_search.

    Params:
        query (str): The research question (e.g., "NVDA earnings sentiment March 2026")
        deep (bool): Use 16 agents instead of 4 (slower, more thorough). Default: false.
    """
    query = params.get("query", "").strip()
    if not query:
        return {"error": "query is required — describe what you want to research"}

    deep = params.get("deep", False)

    try:
        from xai_sdk import AsyncClient
        from xai_sdk.chat import user as sdk_user
        from xai_sdk.tools import web_search, x_search
        from core.grok_llm import MULTI_AGENT_MODEL

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        client = AsyncClient(api_key=api_key)

        # Build multi-agent chat with built-in tools
        create_kwargs = {
            "model": MULTI_AGENT_MODEL,
            "messages": [sdk_user(query)],
            "tools": [web_search(), x_search()],
        }
        if deep:
            create_kwargs["agent_count"] = 16

        chat = client.chat.create(**create_kwargs)
        response = await chat.sample()

        # Track cost
        usage = response.usage
        if hasattr(executor, 'cost_tracker') and executor.cost_tracker:
            executor.cost_tracker.log_llm_call(
                model=MULTI_AGENT_MODEL,
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                purpose="multi_agent_research",
            )

        content = response.content or ""
        agents_used = 16 if deep else 4

        logger.info(
            f"Research complete: {len(content)} chars, "
            f"{agents_used} agents, "
            f"{usage.prompt_tokens}+{usage.completion_tokens} tokens"
        )

        await client.close()

        return {
            "query": query,
            "result": content[:8000],  # Cap to avoid blowing up context
            "agents_used": agents_used,
            "tokens": usage.prompt_tokens + usage.completion_tokens,
            "truncated": len(content) > 8000,
        }

    except Exception as e:
        logger.error(f"Multi-agent research failed: {e}", exc_info=True)
        return {"error": f"Research failed: {e}", "query": query}


HANDLERS = {
    "research": handle_research,
}
