"""
Multi-Agent Research Tool — server-side multi-agent Grok for web/X research.

The trading agent's reasoning model calls research(query) like any other tool.
Internally this fires 4 (or 16 if ``deep``) multi-agent workers with web_search + x_search.
Daily spend is capped separately (``MAX_DAILY_MULTI_AGENT_RESEARCH_USD``) and can be
disabled with ``MULTI_AGENT_RESEARCH_ENABLED=0``.

No custom tools — multi-agent only supports built-in server-side tools.
"""

import logging
import os
import time
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

# ── Research cache: exact-query TTL to avoid repeated expensive calls ──
_research_cache: dict[str, tuple[dict, float]] = {}   # query -> (result_dict, timestamp)
_RESEARCH_TTL = 600  # 10 minutes

_MA_SPEND_DAY_KEY = "multi_agent_llm_day"
_MA_SPEND_USD_KEY = "multi_agent_llm_usd"


def _today_key_float() -> float:
    """Calendar day as YYYYMMDD float for research_config rollover."""
    return float(date.today().strftime("%Y%m%d"))


def _multi_agent_spend_today_usd() -> tuple[float, float]:
    """Return (spent_usd_today, today_key) rolling on local calendar date."""
    from memory import get_research_config

    tkey = _today_key_float()
    stored_day = get_research_config(_MA_SPEND_DAY_KEY, 0.0)
    stored_usd = get_research_config(_MA_SPEND_USD_KEY, 0.0)
    if abs(float(stored_day) - tkey) > 0.5:
        return 0.0, tkey
    return float(stored_usd), tkey


def _record_multi_agent_spend(cost_usd: float, today_key: float) -> None:
    from memory import get_research_config, set_research_config

    stored_day = get_research_config(_MA_SPEND_DAY_KEY, 0.0)
    base = 0.0
    if abs(float(stored_day) - today_key) < 0.5:
        base = float(get_research_config(_MA_SPEND_USD_KEY, 0.0))
    new_total = base + max(0.0, cost_usd)
    set_research_config(_MA_SPEND_DAY_KEY, today_key, "multi-agent spend day rollover")
    set_research_config(_MA_SPEND_USD_KEY, new_total, "multi-agent spend cumulative (USD est.)")


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

    # ── Check TTL cache (exact match, 10-min window) ──
    _cache_key = query.lower().strip()
    if _cache_key in _research_cache:
        cached_result, cached_ts = _research_cache[_cache_key]
        age = time.time() - cached_ts
        if age < _RESEARCH_TTL:
            logger.info(f"Research cache hit ({age:.0f}s old): {query[:60]}")
            cached_result = dict(cached_result)  # copy
            cached_result["cached"] = True
            cached_result["cache_age_s"] = round(age)
            return cached_result
        else:
            del _research_cache[_cache_key]  # expired

    from core.config import MAX_DAILY_MULTI_AGENT_RESEARCH_USD, MULTI_AGENT_RESEARCH_ENABLED

    if not MULTI_AGENT_RESEARCH_ENABLED:
        return {
            "error": "research() disabled (MULTI_AGENT_RESEARCH_ENABLED=0). Use briefing, prior_research, economic_calendar.",
            "query": query,
        }

    spent_today, spend_day_key = _multi_agent_spend_today_usd()
    if spent_today >= MAX_DAILY_MULTI_AGENT_RESEARCH_USD:
        logger.warning(
            "research() blocked: multi-agent spend $%.2f >= cap $%.2f (see MAX_DAILY_MULTI_AGENT_RESEARCH_USD)",
            spent_today,
            MAX_DAILY_MULTI_AGENT_RESEARCH_USD,
        )
        return {
            "error": (
                f"multi-agent research daily cap reached (~${spent_today:.2f} / "
                f"${MAX_DAILY_MULTI_AGENT_RESEARCH_USD:.2f} estimated). "
                "Use briefing / prior_research / cheap tools, or raise the cap tomorrow."
            ),
            "query": query,
            "multi_agent_spend_usd_today": round(spent_today, 4),
            "multi_agent_cap_usd": MAX_DAILY_MULTI_AGENT_RESEARCH_USD,
        }

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

        # Track cost (cost_tracker + persisted daily multi-agent sub-cap)
        usage = response.usage
        cost_usd = 0.0
        if hasattr(executor, "cost_tracker") and executor.cost_tracker:
            cost_usd = executor.cost_tracker.log_llm_usage(
                MULTI_AGENT_MODEL,
                usage=usage,
                purpose="multi_agent_research",
            )
        try:
            _record_multi_agent_spend(cost_usd, spend_day_key)
        except Exception as ex:
            logger.warning("multi-agent spend persist failed (non-fatal): %s", ex)
        spent_after, _ = _multi_agent_spend_today_usd()

        content = response.content or ""
        agents_used = 16 if deep else 4

        logger.info(
            "Research complete: %d chars, %d agents, %d+%d tok | $%.4f this call | "
            "multi-agent day ~$%.2f / $%.2f cap",
            len(content),
            agents_used,
            usage.prompt_tokens,
            usage.completion_tokens,
            cost_usd,
            spent_after,
            MAX_DAILY_MULTI_AGENT_RESEARCH_USD,
        )

        await client.close()

        result = {
            "query": query,
            "result": content[:8000],  # Cap to avoid blowing up context
            "agents_used": agents_used,
            "tokens": usage.prompt_tokens + usage.completion_tokens,
            "truncated": len(content) > 8000,
            "multi_agent_call_cost_usd": round(cost_usd, 4),
            "multi_agent_spend_usd_today": round(spent_after, 4),
            "multi_agent_cap_usd": MAX_DAILY_MULTI_AGENT_RESEARCH_USD,
        }

        # ── Store in cache ──
        _research_cache[_cache_key] = (result, time.time())
        # Evict oldest if > 20 cached queries
        if len(_research_cache) > 20:
            oldest = min(_research_cache, key=lambda k: _research_cache[k][1])
            del _research_cache[oldest]

        return result

    except Exception as e:
        logger.error(f"Multi-agent research failed: {e}", exc_info=True)
        return {"error": f"Research failed: {e}", "query": query}


HANDLERS = {
    "research": handle_research,
}
