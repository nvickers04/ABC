#!/usr/bin/env python3
"""
Order-path smoke (paper): place a non-marketable BUY limit, verify open_orders, cancel.

Prerequisites
-------------
- TWS/IB Gateway; unique ``--client-id`` if another process uses IBKR.
- Default symbol is first ``RESEARCH_UNIVERSE`` name (typically NVDA).

Commands
--------
  python scripts/smoke_order_tools.py --client-id 11
  python scripts/smoke_order_tools.py --client-id 11 --preflight   # open_orders only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


async def main() -> int:
    parser = argparse.ArgumentParser(description="Order-path IBKR smoke")
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="IBKR API client id (must be unique vs other running connections)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Only connect and print open_orders summary",
    )
    parser.add_argument("--symbol", default=None, help="Underlying (default: first RESEARCH_UNIVERSE name)")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
    if args.client_id is not None:
        os.environ["IBKR_CLIENT_ID"] = str(args.client_id)

    from data.broker_gateway import create_gateway
    from data.cost_tracker import get_cost_tracker
    from data.data_provider import get_data_provider
    from data.market_hours import get_market_hours_provider
    from tools.tools_executor import ToolExecutor

    try:
        from research.config import RESEARCH_UNIVERSE
        default_sym = next((str(s).upper() for s in RESEARCH_UNIVERSE if isinstance(s, str)), "NVDA")
    except Exception:
        default_sym = "NVDA"
    sym = (args.symbol or default_sym).upper()

    gateway = await create_gateway({})
    tools = ToolExecutor(
        gateway,
        get_data_provider(),
        market_hours_provider=get_market_hours_provider(),
        cost_tracker=get_cost_tracker(),
    )

    from tools.trader_smoke import run_preflight_open_orders

    if args.preflight:
        try:
            out = await run_preflight_open_orders(tools, symbol=sym)
            print(json.dumps({"symbol": sym, **out}, indent=2, default=str))
            return 0
        finally:
            try:
                await gateway.disconnect()
            except Exception:
                pass

    out: dict = {"symbol": sym, "steps": []}

    try:
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

        # Far below market — should rest as a working order (not fill immediately).
        limit_px = max(0.01, round(float(last) * 0.5, 2))

        lim = await tools.execute(
            "limit_order",
            {"symbol": sym, "side": "BUY", "quantity": 1, "limit_price": limit_px},
        )
        out["steps"].append(
            {"tool": "limit_order", "success": lim.success, "limit_price": limit_px}
        )
        lim_data = lim.data if isinstance(lim.data, dict) else {}
        oid = lim_data.get("order_id")
        if not lim.success:
            out["error"] = lim_data.get("error", str(lim))
            print(json.dumps(out, indent=2, default=str))
            return 1

        oo = await tools.execute("open_orders", {})
        out["steps"].append({"tool": "open_orders", "success": oo.success})
        oo_data = oo.data if isinstance(oo.data, dict) else {}
        out["open_orders_count"] = oo_data.get("count")

        if oid:
            cx = await tools.execute("cancel_order", {"order_id": oid})
            out["steps"].append({"tool": "cancel_order", "success": cx.success, "order_id": oid})
        else:
            out["warning"] = "limit_order returned no order_id; skipping cancel_order"

        oo2 = await tools.execute("open_orders", {})
        out["open_orders_after_cancel"] = (
            oo2.data.get("count") if isinstance(oo2.data, dict) else None
        )

        print(json.dumps(out, indent=2, default=str))
        return 0 if lim.success else 1
    finally:
        try:
            await gateway.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
