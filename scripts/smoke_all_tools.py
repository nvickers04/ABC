#!/usr/bin/env python3
"""
Exercise the full trader tool registry: safe tools + broker-mutating tools.

Does NOT run emergency tools (flatten_limits, cancel_all_orphans).

  python scripts/smoke_all_tools.py --client-id 11 --all
  python scripts/smoke_all_tools.py --client-id 11 --all --with-research   # costly
  python scripts/smoke_all_tools.py --client-id 11 --all --allow-market
  python scripts/smoke_all_tools.py --client-id 11 --all --allow-option-orders  # may place options
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


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Full trader tool smoke (paper)")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--client-id", type=int, required=True, help="Unique IBKR API client id")
    parser.add_argument("--all", action="store_true", help="Run safe + broker phases")
    parser.add_argument("--with-research", action="store_true")
    parser.add_argument("--allow-market", action="store_true", help="Include market_order (1 share BUY)")
    parser.add_argument(
        "--allow-option-orders",
        action="store_true",
        help="Include buy_option / vertical_spread and other option placements",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.all:
        parser.error("Pass --all to confirm full-registry smoke (safety gate).")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
    os.environ["IBKR_CLIENT_ID"] = str(args.client_id)

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
    from tools.trader_smoke import SmokeContext, run_broker_phase, run_safe_phase, warm_option_samples

    gateway = await create_gateway({})
    tools = ToolExecutor(
        gateway,
        get_data_provider(),
        market_hours_provider=get_market_hours_provider(),
        cost_tracker=get_cost_tracker(),
    )

    async def _minimal_state_builder() -> str:
        return "(smoke refresh_state — minimal)"

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

    safe = await run_safe_phase(tools, ctx, with_research=args.with_research)
    broker = await run_broker_phase(
        tools,
        ctx,
        allow_market=args.allow_market,
        allow_option_orders=args.allow_option_orders,
    )

    try:
        await gateway.disconnect()
    except Exception:
        pass

    out = {
        "symbol": symbol,
        "safe": safe,
        "broker": broker,
        "failures": (safe.get("failures") or []) + (broker.get("failures") or []),
    }
    print(json.dumps(out, indent=2, default=str))
    return 1 if out["failures"] else 0


def main() -> None:
    try:
        code = asyncio.run(main_async())
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
