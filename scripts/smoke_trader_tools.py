#!/usr/bin/env python3
"""
Paper-mode smoke: run every *safe* trader tool once (no placements / cancels).

Requires: working .env, IBKR gateway for broker-backed reads, MarketData for quotes.

  python scripts/smoke_trader_tools.py
  python scripts/smoke_trader_tools.py --symbol MSFT
  python scripts/smoke_trader_tools.py --with-research   # multi-agent research ($$$)
  python scripts/smoke_trader_tools.py --client-id 11     # unique IBKR API client id

Full registry (safe + broker-mutating, except emergency tools):

  python scripts/smoke_all_tools.py --client-id 11 --all

Listing skipped / never tools:

  python scripts/smoke_trader_tools.py --list-skipped
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("smoke_trader_tools")


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Trader tool smoke (paper-safe)")
    parser.add_argument("--symbol", default=None, help="Underlying ticker (default: first RESEARCH_UNIVERSE name)")
    parser.add_argument("--with-research", action="store_true", help="Include research() (multi-agent LLM cost)")
    parser.add_argument("--list-skipped", action="store_true", help="Print broker-mutating / never-test tools and exit")
    parser.add_argument("--client-id", type=int, default=None, help="IBKR API client id (unique vs other processes)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
    if args.client_id is not None:
        os.environ["IBKR_CLIENT_ID"] = str(args.client_id)

    from tools.smoke_manifest import NEVER_AUTOTEST, broker_mutating_names

    if args.list_skipped:
        print("NEVER_AUTOTEST:", ", ".join(sorted(NEVER_AUTOTEST)))
        print("BROKER_MUTATING (default skip):", ", ".join(sorted(broker_mutating_names())))
        return 0

    try:
        from research.config import RESEARCH_UNIVERSE
        default_sym = next((str(s).upper() for s in RESEARCH_UNIVERSE if isinstance(s, str)), "NVDA")
    except Exception:
        default_sym = "NVDA"
    symbol = (args.symbol or default_sym).upper()

    from data.broker_gateway import create_gateway
    from data.cost_tracker import get_cost_tracker
    from data.data_provider import get_data_provider
    from data.market_hours import get_market_hours_provider
    from tools.tools_executor import ToolExecutor
    from tools.trader_smoke import SmokeContext, run_safe_phase, warm_option_samples

    gateway = await create_gateway({})
    tools = ToolExecutor(
        gateway,
        get_data_provider(),
        market_hours_provider=get_market_hours_provider(),
        cost_tracker=get_cost_tracker(),
    )

    async def _minimal_state_builder() -> str:
        return "(smoke refresh_state — full agent snapshot unavailable without TradingAgent)"

    tools._state_builder = _minimal_state_builder

    ctx = SmokeContext(symbol=symbol)
    await warm_option_samples(tools, ctx)
    q = await tools.execute("quote", {"symbol": symbol})
    if q.success and isinstance(q.data, dict):
        last = q.data.get("last") or q.data.get("bid") or q.data.get("ask")
        if last is not None:
            try:
                ctx.last = float(last)
            except (TypeError, ValueError):
                pass

    result = await run_safe_phase(tools, ctx, with_research=args.with_research)

    try:
        await gateway.disconnect()
    except Exception:
        pass

    out = {"symbol": symbol, **result}
    print(json.dumps(out, indent=2))
    return 1 if result.get("failures") else 0


def main() -> None:
    try:
        code = asyncio.run(main_async())
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
