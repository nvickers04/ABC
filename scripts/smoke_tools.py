#!/usr/bin/env python3
"""Paper-mode trader tool smoke (safe, full registry, or order path).

  python scripts/smoke_tools.py --client-id 11
  python scripts/smoke_tools.py --client-id 11 --all
  python scripts/smoke_tools.py --client-id 11 --all --allow-market
  python scripts/smoke_tools.py --client-id 11 --order-test
  python scripts/smoke_tools.py --preflight --client-id 11
  python scripts/smoke_tools.py --list-skipped
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

logger = logging.getLogger("smoke_tools")


def _default_symbol() -> str:
    try:
        from research.config import RESEARCH_UNIVERSE

        return next((str(s).upper() for s in RESEARCH_UNIVERSE if isinstance(s, str)), "NVDA")
    except Exception:
        return "NVDA"


async def _run_order_test(tools, sym: str) -> int:
    out: dict = {"symbol": sym, "steps": []}
    q = await tools.execute("quote", {"symbol": sym})
    out["steps"].append({"tool": "quote", "success": q.success})
    if not q.success:
        out["error"] = "quote failed"
        print(json.dumps(out, indent=2, default=str))
        return 1

    payload = q.data if isinstance(q.data, dict) else {}
    last = payload.get("last") or payload.get("bid") or payload.get("ask")
    if not last or float(last) <= 0:
        out["error"] = "no usable last price"
        print(json.dumps(out, indent=2, default=str))
        return 1

    limit_px = max(0.01, round(float(last) * 0.5, 2))
    lim = await tools.execute(
        "limit_order",
        {"symbol": sym, "side": "BUY", "quantity": 1, "limit_price": limit_px},
    )
    out["steps"].append({"tool": "limit_order", "success": lim.success, "limit_price": limit_px})
    lim_data = lim.data if isinstance(lim.data, dict) else {}
    oid = lim_data.get("order_id")
    if not lim.success:
        out["error"] = lim_data.get("error", str(lim))
        print(json.dumps(out, indent=2, default=str))
        return 1

    oo = await tools.execute("open_orders", {})
    out["steps"].append({"tool": "open_orders", "success": oo.success})
    if isinstance(oo.data, dict):
        out["open_orders_count"] = oo.data.get("count")

    if oid:
        cx = await tools.execute("cancel_order", {"order_id": oid})
        out["steps"].append({"tool": "cancel_order", "success": cx.success, "order_id": oid})
    else:
        out["warning"] = "no order_id; skipped cancel"

    oo2 = await tools.execute("open_orders", {})
    if isinstance(oo2.data, dict):
        out["open_orders_after_cancel"] = oo2.data.get("count")

    print(json.dumps(out, indent=2, default=str))
    return 0 if lim.success else 1


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Trader tool smoke (paper)")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--preflight", action="store_true", help="Connect + open_orders only")
    parser.add_argument("--list-skipped", action="store_true", help="Print tools not auto-tested")
    parser.add_argument("--with-research", action="store_true", help="Include research() ($$$)")
    parser.add_argument("--all", action="store_true", help="Safe + broker-mutating tools")
    parser.add_argument("--order-test", action="store_true", help="Single limit place + cancel")
    parser.add_argument("--allow-market", action="store_true")
    parser.add_argument("--allow-option-orders", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.list_skipped:
        from tools.smoke_manifest import NEVER_AUTOTEST, broker_mutating_names

        print("NEVER_AUTOTEST:", ", ".join(sorted(NEVER_AUTOTEST)))
        print("BROKER_MUTATING:", ", ".join(sorted(broker_mutating_names())))
        return 0

    if args.all and not args.client_id:
        parser.error("--all requires --client-id")
    if args.order_test and not args.client_id:
        parser.error("--order-test requires --client-id")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
    if args.client_id is not None:
        os.environ["IBKR_CLIENT_ID"] = str(args.client_id)

    symbol = (args.symbol or _default_symbol()).upper()

    from data.broker_gateway import create_gateway
    from data.cost_tracker import get_cost_tracker
    from data.data_provider import get_data_provider
    from data.market_hours import get_market_hours_provider
    from tools.tools_executor import ToolExecutor
    from tools.trader_smoke import (
        SmokeContext,
        run_broker_phase,
        run_preflight_open_orders,
        run_safe_phase,
        warm_option_samples,
    )

    gateway = await create_gateway({})
    tools = ToolExecutor(
        gateway,
        get_data_provider(),
        market_hours_provider=get_market_hours_provider(),
        cost_tracker=get_cost_tracker(),
    )

    async def _minimal_state_builder() -> str:
        return "(smoke — minimal state)"

    tools._state_builder = _minimal_state_builder

    try:
        if args.preflight:
            out = await run_preflight_open_orders(tools, symbol=symbol)
            print(json.dumps(out, indent=2, default=str))
            return 0

        if args.order_test:
            return await _run_order_test(tools, symbol)

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
        if not args.all:
            print(json.dumps({"symbol": symbol, **safe}, indent=2))
            return 1 if safe.get("failures") else 0

        broker = await run_broker_phase(
            tools,
            ctx,
            allow_market=args.allow_market,
            allow_option_orders=args.allow_option_orders,
        )
        out = {
            "symbol": symbol,
            "safe": safe,
            "broker": broker,
            "failures": (safe.get("failures") or []) + (broker.get("failures") or []),
        }
        print(json.dumps(out, indent=2, default=str))
        return 1 if out["failures"] else 0
    finally:
        try:
            await gateway.disconnect()
        except Exception:
            pass


def main() -> None:
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
