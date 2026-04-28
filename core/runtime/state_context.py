"""StateContextBuilder — extracted state-assembly text builder.

Moved from :meth:`core.agent.TradingAgent._build_state_context`. The builder
formats the multi-section human-readable state block that is fed to the LLM
each cycle. Nothing here makes trading decisions — it is purely an
observability / context module.

Behavior parity with the original method is locked by
``tests/test_runtime_characterization.py``. Section ordering, headers, and
emoji glyphs are intentionally byte-identical so that the agent prompt does
not drift as a side effect of refactoring.
"""

from __future__ import annotations

import logging
from datetime import datetime as _dt
from typing import Optional

from core.runtime.interfaces import BrokerGatewayProtocol, MarketHoursProtocol

logger = logging.getLogger(__name__)


# ── Symbol → sector map for portfolio concentration summary ─────
# Matches the diversified RESEARCH_UNIVERSE.  Used for cheap in-prompt
# exposure reporting; no external lookup.  Unknown symbols bucket as
# "other".
_SECTOR_MAP: dict[str, str] = {
    # Mega-cap tech / AI
    "NVDA": "tech", "META": "tech", "AMD": "tech", "AVGO": "tech",
    # High-growth software
    "CRWD": "software", "NET": "software", "PLTR": "software", "APP": "software",
    # Fintech
    "SOFI": "fintech", "HOOD": "fintech",
    # Consumer discretionary
    "DKNG": "discretionary", "CAVA": "discretionary",
    # Healthcare
    "LLY": "healthcare", "UNH": "healthcare", "GILD": "healthcare",
    # Energy
    "XOM": "energy", "OXY": "energy",
    # Traditional financials
    "JPM": "financials", "GS": "financials",
    # Industrials
    "CAT": "industrials", "UPS": "industrials",
    # Staples
    "COST": "staples", "WMT": "staples",
    # Materials
    "FCX": "materials",
    # EM / China
    "BABA": "em",
    # Common hedges
    "SPY": "index_hedge", "QQQ": "index_hedge", "IWM": "index_hedge",
}


def _sector_of(symbol: str) -> str:
    return _SECTOR_MAP.get((symbol or "").upper(), "other")


class StateContextBuilder:
    """Builds the per-cycle state text block fed into the LLM prompt.

    Dependencies are injected so the builder can be unit-tested with stub
    gateways. ``market_hours_provider`` is optional at construction time —
    if omitted, the builder falls back to ``data.market_hours.get_market_hours_provider``
    on each call (matching the original lazy lookup).
    """

    def __init__(
        self,
        gateway: BrokerGatewayProtocol,
        market_hours_provider: Optional[MarketHoursProtocol] = None,
    ) -> None:
        self.gateway = gateway
        self._market_hours_provider = market_hours_provider

    def _get_market_hours(self) -> MarketHoursProtocol:
        if self._market_hours_provider is not None:
            return self._market_hours_provider
        from data.market_hours import get_market_hours_provider
        return get_market_hours_provider()

    async def build(self) -> str:
        """Build the complete dynamic state context for the agent.

        One function, one source of truth. Includes:
        - Market session + session-specific constraints
        - Account balances
        - Open positions with P&L
        - Open orders
        """
        lines: list[str] = []

        # ── Market session ──────────────────────────────────────
        session = "UNKNOWN"
        try:
            mh = self._get_market_hours()
            info = mh.get_session_info()
            session = info["session"].upper()
            time_et = info["current_time_et"]
            detail = ""
            if "minutes_to_open" in info:
                detail = f" | Open in {info['minutes_to_open']}min"
            elif "minutes_to_close" in info:
                detail = f" | Close in {info['minutes_to_close']}min"
            if info.get("early_close"):
                detail += f" | ⚠ EARLY CLOSE {info['close_time']} ET"
            lines.append(f"═══ MARKET: {session} ({time_et} ET){detail} ═══")
        except Exception as e:
            lines.append(f"═══ MARKET: UNKNOWN (error: {e}) ═══")

        # Session constraint reminder (reinforces system prompt rules)
        if session == "PREMARKET":
            lines.append("  ⚠ Limit orders ONLY. bracket_order will timeout. Options won't fill.")
        elif session == "POSTMARKET":
            lines.append("  ⚠ Limit orders ONLY. Market orders rejected.")
        elif session == "CLOSED":
            lines.append("  ⛔ Market CLOSED. Research/plan only — do NOT place orders.")

        lines.append("")
        lines.append("═══ ACCOUNT ═══")
        try:
            summary = await self.gateway.get_account_summary()
            cash = summary.get("totalcashvalue", 0)
            net_liq = summary.get("netliquidation", 0)
            daily_pnl = summary.get("dailypnl", "N/A")
            unreal = summary.get("unrealizedpnl", "N/A")
            real = summary.get("realizedpnl", "N/A")
            lines.append(f"Cash (SIZE FROM THIS): ${cash:,.2f}  |  NetLiq: ${net_liq:,.2f}")
            lines.append(f"Day P&L: {daily_pnl}  |  Unreal: {unreal}  |  Real: {real}")
        except Exception as e:
            lines.append(f"Account error: {e}")

        lines.append("")
        lines.append("═══ POSITIONS (verdict required for each before cycle ends) ═══")
        _positions_snapshot: list[dict] = []
        try:
            positions = await self.gateway.get_positions()
            _positions_snapshot = positions or []
            if not positions:
                lines.append("No open positions.")
            else:
                lines.append(
                    "  Each position needs a verdict this cycle: "
                    "HOLD (one-line reason) | TRIM | CLOSE | ROLL | TIGHTEN_STOP | HEDGE."
                )
                lines.append(
                    "  \"Brackets exist\" is not a verdict. Ask: is the original thesis intact, "
                    "and is this still the best use of this capital?"
                )
                for p in positions:
                    sym = p.get("symbol", "?")
                    qty = p.get("quantity", 0)
                    avg = p.get("avg_cost", 0)
                    mkt = p.get("market_price", 0)
                    pnl = p.get("unrealized_pnl", 0)
                    sec = p.get("sec_type", "STK")
                    pnl_pct = ((mkt - avg) / avg * 100) if avg > 0 else 0
                    if sec == "OPT":
                        strike = p.get("strike", "")
                        right = p.get("right", "")
                        exp = p.get("expiration", "")
                        dte = "?"
                        try:
                            exp_date = _dt.strptime(str(exp)[:8], "%Y%m%d").date()
                            dte = (exp_date - _dt.now().date()).days
                        except Exception:
                            pass
                        lines.append(
                            f"  {sym} {right}{strike} exp={exp} DTE={dte}: {qty} @ ${avg:.2f} "
                            f"| mkt ${mkt:.2f} | P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                        )
                    else:
                        lines.append(
                            f"  {sym}: {qty} @ ${avg:.2f} "
                            f"| mkt ${mkt:.2f} | P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                        )
        except Exception as e:
            lines.append(f"Position error: {e}")

        # ── Portfolio risk summary (cheap aggregation, no extra calls) ──
        try:
            if _positions_snapshot:
                net_liq_val = 0.0
                cash_val = 0.0
                try:
                    _s = await self.gateway.get_account_summary()
                    net_liq_val = float(_s.get("netliquidation", 0) or 0)
                    cash_val = float(_s.get("totalcashvalue", 0) or 0)
                except Exception:
                    pass

                long_stock_notional = 0.0
                short_stock_notional = 0.0
                opt_long_contracts = 0
                opt_short_contracts = 0
                total_unreal = 0.0
                # sym -> {notional, unreal}
                per_symbol: dict[str, dict[str, float]] = {}

                for p in _positions_snapshot:
                    sec = p.get("sec_type", "STK")
                    sym = p.get("symbol", "?")
                    qty = float(p.get("quantity", 0) or 0)
                    mkt = float(p.get("market_price", 0) or 0)
                    unreal = float(p.get("unrealized_pnl", 0) or 0)
                    total_unreal += unreal
                    if sec == "OPT":
                        # option notional ≈ qty * 100 * mkt (premium paid/received)
                        notional = abs(qty) * 100.0 * mkt
                        if qty > 0:
                            opt_long_contracts += int(qty)
                        else:
                            opt_short_contracts += int(-qty)
                    else:
                        notional = abs(qty) * mkt
                        if qty > 0:
                            long_stock_notional += notional
                        else:
                            short_stock_notional += notional
                    agg = per_symbol.setdefault(sym, {"notional": 0.0, "unreal": 0.0})
                    agg["notional"] += notional
                    agg["unreal"] += unreal

                lines.append("")
                lines.append("═══ PORTFOLIO RISK ═══")
                if net_liq_val > 0:
                    pct_long = long_stock_notional / net_liq_val * 100
                    pct_cash = cash_val / net_liq_val * 100
                    lines.append(
                        f"Stock long: ${long_stock_notional:,.0f} ({pct_long:.1f}% of NetLiq)  "
                        f"| Cash: ${cash_val:,.0f} ({pct_cash:.1f}%)"
                    )
                    # Idle-cash flag: if more than 30% of NetLiq is in cash,
                    # the agent has slack and should actively evaluate top
                    # candidates instead of defaulting to skip.
                    if pct_cash > 30:
                        lines.append(
                            f"  ⚠ IDLE CASH: {pct_cash:.0f}% of NetLiq is uninvested. "
                            "Required this cycle: evaluate the top composite from briefing() "
                            "that is not already held — chart_intraday + one context tool "
                            "(news OR iv_info), then verdict TAKE or PASS with reason. "
                            "PASS is fully valid (most evals should pass on marginal/warming "
                            "edge days). Do NOT enter a weak setup just because cash is idle. "
                            "What's not allowed is ending the cycle with NO evaluation."
                        )
                else:
                    lines.append(f"Stock long: ${long_stock_notional:,.0f}  | Cash: ${cash_val:,.0f}")
                if opt_long_contracts or opt_short_contracts:
                    lines.append(
                        f"Options: {opt_long_contracts} long contracts, {opt_short_contracts} short contracts "
                        f"(run position_greeks for net delta/vega/theta)"
                    )
                lines.append(f"Open unrealized P&L: ${total_unreal:+,.2f}")

                # Concentration: top 3 symbols by notional
                if per_symbol:
                    top_syms = sorted(
                        per_symbol.items(), key=lambda kv: kv[1]["notional"], reverse=True
                    )[:3]
                    if net_liq_val > 0:
                        conc_str = ", ".join(
                            f"{s}={v['notional'] / net_liq_val * 100:.1f}%" for s, v in top_syms
                        )
                    else:
                        conc_str = ", ".join(f"{s}=${v['notional']:,.0f}" for s, v in top_syms)
                    lines.append(f"Concentration (top 3 of NetLiq): {conc_str}")

                # Sector exposure — group per-symbol notional by sector
                if per_symbol:
                    sector_totals: dict[str, float] = {}
                    for sym, v in per_symbol.items():
                        sector_totals[_sector_of(sym)] = (
                            sector_totals.get(_sector_of(sym), 0.0) + v["notional"]
                        )
                    if sector_totals:
                        sorted_sectors = sorted(
                            sector_totals.items(), key=lambda kv: kv[1], reverse=True
                        )
                        if net_liq_val > 0:
                            sec_str = ", ".join(
                                f"{s}={n / net_liq_val * 100:.1f}%"
                                for s, n in sorted_sectors if n > 0
                            )
                        else:
                            sec_str = ", ".join(
                                f"{s}=${n:,.0f}" for s, n in sorted_sectors if n > 0
                            )
                        if sec_str:
                            lines.append(f"Sector exposure: {sec_str}")
        except Exception as e:
            logger.debug(f"Portfolio risk summary failed: {e}")

        # ── Research engine status (scorer + template evolution) ──
        try:
            from signals import scorer as _sc
            from signals import template_evolution as _te
            sc_state = "off"
            if _sc.is_scorer_running():
                sc_state = "paused" if _sc.is_scorer_paused() else "running"
            te_state = "off"
            if _te.is_evolution_running():
                te_state = "paused" if _te.is_evolution_paused() else "running"

            # IR snapshot freshness — tells the agent whether briefing.edge
            # is current or stale. Stale edge can still be used but the agent
            # should discount it.
            age_str = "unknown"
            try:
                from memory import get_research_config
                import time as _time
                ts = float(get_research_config("ir_snapshot_ts", 0.0))
                if ts > 0:
                    age_s = _time.time() - ts
                    if age_s < 60:
                        age_str = f"{age_s:.0f}s ago"
                    elif age_s < 3600:
                        age_str = f"{age_s/60:.0f}min ago"
                    elif age_s < 86400:
                        age_str = f"{age_s/3600:.1f}h ago (STALE)"
                    else:
                        age_str = f"{age_s/86400:.1f}d ago (VERY STALE)"
            except Exception:
                pass

            lines.append("")
            lines.append(
                f"═══ RESEARCH ENGINE ═══ scorer={sc_state}  evolution={te_state}  "
                f"edge_snapshot={age_str}"
            )
            lines.append("(control via research_engine action=start|pause|resume|stop)")
        except Exception as e:
            logger.debug(f"Engine status failed: {e}")

        lines.append("")
        lines.append("═══ OPEN ORDERS ═══")
        try:
            orders = await self.gateway.get_open_orders()
            if not orders:
                lines.append("No open orders.")
            else:
                for o in orders:
                    sym = o.get("symbol", "?")
                    action = o.get("action", "?")
                    qty = o.get("quantity", 0)
                    otype = o.get("order_type", "?")
                    lmt = o.get("lmt_price")
                    aux = o.get("aux_price")
                    oid = o.get("order_id", "?")
                    sec = o.get("sec_type", "STK")
                    trail_pct = o.get("trail_percent")
                    price_str = ""
                    if lmt:
                        price_str = f"lmt ${lmt:.2f}"
                    if aux:
                        price_str += f" aux ${aux:.2f}"
                    if trail_pct:
                        price_str += f" trail {trail_pct:.1f}%"
                    # Tag non-stock orders so agent doesn't confuse them
                    tag = ""
                    if sec == "BAG":
                        tag = " [COMBO/SPREAD]"
                    elif sec == "OPT":
                        tag = " [OPTION]"
                    lines.append(f"  #{oid} {action} {qty} {sym} {otype} {price_str}{tag}")
        except Exception as e:
            lines.append(f"Order error: {e}")

        # ── Active graduated params (execution research) ────────
        try:
            from memory import get_graduated_params
            params = get_graduated_params(active_only=True)
            if params:
                lines.append("")
                lines.append("═══ EXECUTION OPTIMIZATIONS (auto-graduated) ═══")
                for p in params[:5]:
                    lines.append(f"  {p['param_key']} = {p['param_value']} (+{p['improvement_bps']:.1f}bps)")
        except Exception:
            pass

        return "\n".join(lines)
