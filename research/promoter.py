"""
Options Promotion Engine — historical leg-level repricing for options strategies.

This module is the core of the promotion-grade evaluation tier.  Unlike the fast
search simulator (which uses Greek/delta approximations), this engine:

1. Looks up a historical option chain on the signal's entry date via MarketData.app
2. Selects the best-matching contract(s) per leg using strike + expiration proximity
3. Fetches the historical (closing) mark for each leg on both entry and exit date
4. Computes realized leg-level cash flows including bid/ask spread costs
5. Applies explicit data-quality policies (stale, missing, out-of-range quotes)
6. Returns a per-signal repriced result alongside a data-coverage metric

Supported structures: vertical_spread, iron_condor, straddle, strangle,
  calendar_spread, diagonal_spread, butterfly.

The engine is deliberately cautious: it assigns the conservative (mid-to-cross)
fill rather than the optimistic mid-to-mid fill, and it rejects positions where
data coverage is below MIN_DATA_COVERAGE.

Designed to be called by research/agent.py during the promotion evaluation step
and NOT during the fast inner search loop.

Note: MarketData.app historical data is as-traded (NOT split/dividend adjusted).
Callers should normalise strikes before calling if a split occurred between the
signal date and today.
"""

from __future__ import annotations

from collections import Counter
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Policy constants ────────────────────────────────────────────
# Minimum fraction of legs that must be successfully repriced for a signal
# to count toward promotion fitness (rather than being dropped as missing-data).
MIN_DATA_COVERAGE: float = 0.80

# Maximum age of a "current" quote before we flag it as stale (in days).
MAX_QUOTE_STALENESS_DAYS: int = 3

# Per-contract spread cost fraction added to each leg: we add half the bid/ask
# spread on entry and half on exit to simulate realistic crossing costs.
SPREAD_COST_FRACTION: float = 0.5  # multiply by bid/ask spread width

# Options per-contract multiplier (standard: 100 shares per contract)
CONTRACT_MULTIPLIER: int = 100

# Minimum open-interest + volume thresholds for a leg to be considered liquid.
MIN_OI: int = 50
MIN_VOLUME: int = 5


# ── Result dataclasses ──────────────────────────────────────────

@dataclass
class LegValuation:
    """Valuation record for a single option leg."""
    option_symbol: str
    leg_role: str           # 'long_call', 'short_put', 'long_put', etc.
    entry_date: str
    exit_date: str
    entry_mid: Optional[float] = None
    exit_mid: Optional[float] = None
    entry_bid: Optional[float] = None
    entry_ask: Optional[float] = None
    exit_bid: Optional[float] = None
    exit_ask: Optional[float] = None
    entry_iv: Optional[float] = None
    exit_iv: Optional[float] = None
    entry_delta: Optional[float] = None
    realized_pnl: Optional[float] = None  # per-contract in dollars
    fill_policy: str = "mid"   # 'mid' | 'cross' | 'missing_stale'
    data_quality: str = "ok"   # 'ok' | 'stale' | 'missing' | 'interpolated'
    is_long: bool = True


@dataclass
class SignalReprice:
    """Full repricing result for a single options signal."""
    signal_id: Optional[int]
    symbol: str
    strategy: str
    entry_date: str
    exit_date: str
    legs: list[LegValuation] = field(default_factory=list)
    total_pnl: Optional[float] = None      # sum of leg PnLs per contract in dollars
    return_pct: Optional[float] = None     # total_pnl / max_risk * 100
    max_risk: Optional[float] = None       # max capital at risk per contract
    data_coverage: float = 0.0             # fraction of legs successfully repriced
    outcome: str = "pending"               # 'repriced' | 'partial' | 'missing_data'
    rejection_code: Optional[str] = None
    rejection_reason: Optional[str] = None


# ── Main public API ─────────────────────────────────────────────

class OptionsPromoter:
    """
    Reprices one or more options signals using historical MarketData chain/quote data.

    Usage:
        promoter = OptionsPromoter(data_provider)
        results = promoter.reprice_signals(signals, trade_date="2025-03-05")
        coverage = sum(1 for r in results if r.outcome == "repriced") / len(results)
    """

    def __init__(self, data_provider):
        self._dp = data_provider
        # Symbols confirmed to have no options chain data in the API this session.
        # Keyed by (symbol, expiration, entry_date) so different dates can still be tried.
        self._no_data_cache: set[tuple[str, str, str]] = set()

    # ─────────────────────────────────────────────────────────────

    def reprice_signals(
        self,
        signals: list[dict],
        trade_date: str,
        exit_date: Optional[str] = None,
    ) -> list[SignalReprice]:
        """
        Reprice a list of options signals for a given trade date.

        Args:
            signals: list of signal dicts (must contain legs_json)
            trade_date: entry date 'YYYY-MM-DD' for historical chain lookup
            exit_date: exit date for quote lookup; defaults to trade_date + 1 business day

        Returns:
            list of SignalReprice, one per signal that has legs_json
        """
        if exit_date is None:
            exit_date = _next_trading_day(trade_date)

        results = []
        for sig in signals:
            legs = sig.get("legs_json")
            if not (legs and isinstance(legs, dict)):
                continue
            result = self._reprice_one(sig, legs, trade_date, exit_date)
            results.append(result)
        return results

    def reprice_signal(
        self,
        sig: dict,
        trade_date: str,
        exit_date: Optional[str] = None,
    ) -> SignalReprice:
        """Reprice a single signal."""
        if exit_date is None:
            exit_date = _next_trading_day(trade_date)
        legs = sig.get("legs_json") or {}
        return self._reprice_one(sig, legs, trade_date, exit_date)

    # ─────────────────────────────────────────────────────────────

    def _reprice_one(
        self,
        sig: dict,
        legs_dict: dict,
        entry_date: str,
        exit_date: str,
    ) -> SignalReprice:
        strategy = legs_dict.get("strategy", "unknown")
        symbol = sig.get("symbol", "")
        signal_id = sig.get("id")

        result = SignalReprice(
            signal_id=signal_id,
            symbol=symbol,
            strategy=strategy,
            entry_date=entry_date,
            exit_date=exit_date,
        )

        try:
            leg_specs = _extract_legs(strategy, legs_dict, symbol)
        except ValueError as exc:
            result.outcome = "missing_data"
            result.rejection_code = "invalid_legs"
            result.rejection_reason = str(exc)
            return result

        if not leg_specs:
            result.outcome = "missing_data"
            result.rejection_code = "invalid_legs"
            result.rejection_reason = "No legs extracted from legs_json"
            return result

        # Fetch historical chain on entry date for contract resolution
        requested_expiration = _primary_expiration(legs_dict) or ""
        expiration = _normalise_exp(requested_expiration) or None
        cache_key = (symbol, expiration or "", entry_date)
        if cache_key in self._no_data_cache:
            result.outcome = "missing_data"
            result.rejection_code = "no_chain_data"
            result.rejection_reason = "No chain data (cached)"
            return result

        chain = None
        chain_contracts = []
        chain_fetch_mode = "date_only"
        if expiration and _is_concrete_expiration(expiration):
            chain = self._dp.get_option_chain(symbol, expiration=expiration, date=entry_date)
            chain_contracts = chain.contracts if chain else []
            chain_fetch_mode = "exact_expiration"

        # Fallback: fetch the full historical date snapshot and resolve expiration/strike client-side.
        if not chain_contracts and expiration and _is_concrete_expiration(expiration):
            logger.debug(
                f"[Promoter] Exact historical chain miss for {symbol} on {entry_date} "
                f"exp={expiration}; retrying with date-only snapshot"
            )
            chain = self._dp.get_option_chain(symbol, date=entry_date)
            chain_contracts = chain.contracts if chain else []
            if chain_contracts:
                chain_fetch_mode = "date_only"
                logger.debug(
                    f"[Promoter] Date-only historical chain fallback returned "
                    f"{len(chain_contracts)} contracts for {symbol} on {entry_date}"
                )

        if not chain_contracts and expiration and not _is_concrete_expiration(expiration):
            chain = self._dp.get_option_chain(symbol, date=entry_date)
            chain_contracts = chain.contracts if chain else []
            chain_fetch_mode = "date_only_symbolic"
            if chain_contracts:
                logger.debug(
                    f"[Promoter] Symbolic expiration {expiration} for {symbol} on {entry_date}; "
                    f"resolving against date-only snapshot with {len(chain_contracts)} contracts"
                )

        if not chain_contracts:
            self._no_data_cache.add(cache_key)
            result.outcome = "missing_data"
            result.rejection_code = "no_chain_data"
            result.rejection_reason = (
                f"No chain data for {symbol} on {entry_date} "
                f"(exp={expiration or 'any'}, fetch=exact+date_only)"
            )
            return result

        leg_specs = _resolve_leg_expirations(leg_specs, chain_contracts, entry_date)

        valuations: list[LegValuation] = []
        for spec in leg_specs:
            val = self._price_leg(spec, chain_contracts, entry_date, exit_date)
            valuations.append(val)

        result.legs = valuations

        # Compute coverage and aggregate PnL
        priced = [v for v in valuations if v.realized_pnl is not None]
        coverage = len(priced) / len(valuations) if valuations else 0.0
        result.data_coverage = round(coverage, 4)

        if coverage < MIN_DATA_COVERAGE:
            missing_roles = [v.leg_role for v in valuations if v.realized_pnl is None]
            available_expirations = sorted({
                _normalise_exp(str(getattr(c, "expiration", "") or ""))
                for c in chain_contracts
                if getattr(c, "expiration", None)
            })[:8]
            rejection_code = _classify_missing_data(leg_specs, valuations, chain_contracts)
            result.rejection_code = rejection_code
            logger.debug(
                f"[Promoter] Coverage shortfall for {symbol} {strategy} on {entry_date}: "
                f"coverage={coverage:.0%}, fetch={chain_fetch_mode}, missing_roles={missing_roles}, "
                f"requested_exp={expiration or 'any'}, available_exps={available_expirations}, "
                f"reason={rejection_code}"
            )
            result.outcome = "missing_data"
            result.rejection_reason = (
                f"Data coverage {coverage:.0%} < minimum {MIN_DATA_COVERAGE:.0%} "
                f"(fetch={chain_fetch_mode}, reason={rejection_code}, missing_roles={missing_roles})"
            )
            return result

        total_pnl = sum(v.realized_pnl for v in priced if v.realized_pnl is not None)
        result.total_pnl = round(total_pnl, 4)

        # Compute max risk (capital at risk per contract set)
        max_risk = _compute_max_risk(strategy, legs_dict, valuations)
        result.max_risk = max_risk

        if max_risk and max_risk > 0:
            result.return_pct = round(total_pnl / max_risk * 100, 4)

        result.outcome = "repriced" if coverage >= 1.0 else "partial"
        return result

    # ─────────────────────────────────────────────────────────────

    def _price_leg(
        self,
        spec: "_LegSpec",
        chain_contracts: list,
        entry_date: str,
        exit_date: str,
    ) -> LegValuation:
        """Price a single leg using historical chain + quote data."""

        # Step 1: find the best-matching contract in the historical chain
        option_symbol = _match_contract(spec, chain_contracts)

        val = LegValuation(
            option_symbol=option_symbol or "",
            leg_role=spec.role,
            entry_date=entry_date,
            exit_date=exit_date,
            is_long=spec.is_long,
        )

        if not option_symbol:
            val.data_quality = "missing"
            val.fill_policy = "missing_stale"
            return val

        # Step 2: fetch entry-date quote
        entry_quote = self._dp.get_option_quote(option_symbol, date=entry_date)
        if not entry_quote:
            # Fallback: use the chain snapshot's embedded bid/ask/mid data.
            # The historical chain already contains end-of-day pricing that is
            # good enough for promotion scoring when per-contract quotes are
            # unavailable from the API.
            chain_contract = _find_chain_contract(option_symbol, chain_contracts)
            if chain_contract:
                entry_quote = {
                    "bid": getattr(chain_contract, "bid", None),
                    "ask": getattr(chain_contract, "ask", None),
                    "mid": getattr(chain_contract, "mid", None),
                    "iv": getattr(chain_contract, "iv", None),
                    "delta": getattr(chain_contract, "delta", None),
                    "open_interest": getattr(chain_contract, "open_interest", 0),
                    "volume": getattr(chain_contract, "volume", 0),
                    "as_of_date": getattr(chain_contract, "as_of_date", entry_date),
                }
                # Only use if there's an actual price
                if not (entry_quote.get("mid") or entry_quote.get("bid")):
                    entry_quote = None
        if entry_quote:
            val.entry_bid = entry_quote.get("bid")
            val.entry_ask = entry_quote.get("ask")
            val.entry_mid = entry_quote.get("mid") or _safe_mid(val.entry_bid, val.entry_ask)
            val.entry_iv = entry_quote.get("iv")
            val.entry_delta = entry_quote.get("delta")
            # Check staleness
            data_date = entry_quote.get("as_of_date") or entry_date
            if _days_diff(data_date, entry_date) > MAX_QUOTE_STALENESS_DAYS:
                val.data_quality = "stale"
                val.fill_policy = "missing_stale"
        else:
            val.data_quality = "missing"
            val.fill_policy = "missing_stale"
            return val

        # Step 3: fetch exit-date quote
        exit_quote = self._dp.get_option_quote(option_symbol, date=exit_date)
        if exit_quote:
            val.exit_bid = exit_quote.get("bid")
            val.exit_ask = exit_quote.get("ask")
            val.exit_mid = exit_quote.get("mid") or _safe_mid(val.exit_bid, val.exit_ask)
            val.exit_iv = exit_quote.get("iv")
            # Check staleness
            data_date = exit_quote.get("as_of_date") or exit_date
            if _days_diff(data_date, exit_date) > MAX_QUOTE_STALENESS_DAYS:
                if val.data_quality == "ok":
                    val.data_quality = "stale"
                val.fill_policy = "missing_stale"
        else:
            val.data_quality = "missing"
            val.fill_policy = "missing_stale"
            return val

        # Step 4: check minimum liquidity
        entry_oi = entry_quote.get("open_interest", 0) or 0
        entry_vol = entry_quote.get("volume", 0) or 0
        if entry_oi < MIN_OI and entry_vol < MIN_VOLUME:
            val.data_quality = "stale"  # illiquid = unreliable marks
            val.fill_policy = "missing_stale"

        # Step 5: compute realized PnL using mid-to-mid with spread-cost penalty
        if val.entry_mid is not None and val.exit_mid is not None:
            entry_spread = _safe_spread(val.entry_bid, val.entry_ask)
            exit_spread = _safe_spread(val.exit_bid, val.exit_ask)
            entry_cost = (entry_spread or 0) * SPREAD_COST_FRACTION
            exit_cost = (exit_spread or 0) * SPREAD_COST_FRACTION
            # Long leg: we pay entry + exit spread crossing costs
            # Short leg: we receive credit but also pay spread-cross costs
            raw_pnl = (val.exit_mid - val.entry_mid) * CONTRACT_MULTIPLIER
            spread_drag = (entry_cost + exit_cost) * CONTRACT_MULTIPLIER
            if spec.is_long:
                val.realized_pnl = round(raw_pnl - spread_drag, 4)
            else:
                val.realized_pnl = round(-raw_pnl - spread_drag, 4)
            val.fill_policy = "cross" if val.data_quality == "stale" else "mid"
        else:
            val.data_quality = "missing"
            val.fill_policy = "missing_stale"

        return val


# ── Internal helpers ────────────────────────────────────────────

from dataclasses import dataclass as _dc


@_dc
class _LegSpec:
    """Internal specification for one leg to be priced."""
    side: str           # 'call' or 'put'
    strike: float
    expiration: str     # YYYY-MM-DD
    is_long: bool       # True = long, False = short
    role: str           # human-readable role label


def _extract_legs(strategy: str, legs: dict, symbol: str) -> list[_LegSpec]:
    """Extract individual leg specs from a legs_json dict."""
    exp = legs.get("expiration", "")
    specs: list[_LegSpec] = []

    if strategy == "vertical_spread":
        right = str(legs.get("right", "C")).upper()
        side = "call" if right in ("C", "CALL") else "put"
        ls, ss = float(legs["long_strike"]), float(legs["short_strike"])
        specs = [
            _LegSpec(side=side, strike=ls, expiration=exp, is_long=True,
                     role=f"long_{side}"),
            _LegSpec(side=side, strike=ss, expiration=exp, is_long=False,
                     role=f"short_{side}"),
        ]

    elif strategy == "iron_condor":
        pl, ps = float(legs["put_long_strike"]), float(legs["put_short_strike"])
        cs, cl = float(legs["call_short_strike"]), float(legs["call_long_strike"])
        specs = [
            _LegSpec(side="put",  strike=pl, expiration=exp, is_long=True,  role="long_put"),
            _LegSpec(side="put",  strike=ps, expiration=exp, is_long=False, role="short_put"),
            _LegSpec(side="call", strike=cs, expiration=exp, is_long=False, role="short_call"),
            _LegSpec(side="call", strike=cl, expiration=exp, is_long=True,  role="long_call"),
        ]

    elif strategy == "straddle":
        k = float(legs["strike"])
        specs = [
            _LegSpec(side="call", strike=k, expiration=exp, is_long=True, role="long_call"),
            _LegSpec(side="put",  strike=k, expiration=exp, is_long=True, role="long_put"),
        ]

    elif strategy == "strangle":
        ps, cs = float(legs["put_strike"]), float(legs["call_strike"])
        specs = [
            _LegSpec(side="put",  strike=ps, expiration=exp, is_long=True, role="long_put"),
            _LegSpec(side="call", strike=cs, expiration=exp, is_long=True, role="long_call"),
        ]

    elif strategy == "calendar_spread":
        right = str(legs.get("right", "C")).upper()
        side = "call" if right in ("C", "CALL") else "put"
        k = float(legs["strike"])
        near_exp = legs["near_expiration"]
        far_exp = legs["far_expiration"]
        specs = [
            _LegSpec(side=side, strike=k, expiration=near_exp, is_long=False,
                     role=f"short_{side}_near"),
            _LegSpec(side=side, strike=k, expiration=far_exp,  is_long=True,
                     role=f"long_{side}_far"),
        ]

    elif strategy == "diagonal_spread":
        right = str(legs.get("right", "C")).upper()
        side = "call" if right in ("C", "CALL") else "put"
        nk, fk = float(legs["near_strike"]), float(legs["far_strike"])
        near_exp = legs["near_expiration"]
        far_exp = legs["far_expiration"]
        specs = [
            _LegSpec(side=side, strike=nk, expiration=near_exp, is_long=False,
                     role=f"short_{side}_near"),
            _LegSpec(side=side, strike=fk, expiration=far_exp,  is_long=True,
                     role=f"long_{side}_far"),
        ]

    elif strategy == "butterfly":
        right = str(legs.get("right", "C")).upper()
        side = "call" if right in ("C", "CALL") else "put"
        lo, mid, hi = (
            float(legs["low_strike"]),
            float(legs["mid_strike"]),
            float(legs["high_strike"]),
        )
        specs = [
            _LegSpec(side=side, strike=lo,  expiration=exp, is_long=True,  role=f"long_{side}_low"),
            _LegSpec(side=side, strike=mid, expiration=exp, is_long=False, role=f"short_{side}_mid"),
            _LegSpec(side=side, strike=mid, expiration=exp, is_long=False, role=f"short_{side}_mid2"),
            _LegSpec(side=side, strike=hi,  expiration=exp, is_long=True,  role=f"long_{side}_high"),
        ]

    else:
        raise ValueError(f"Unsupported strategy for promotion repricing: {strategy}")

    return specs


def _find_chain_contract(option_symbol: str, chain: list):
    """Find a contract in the chain snapshot by option_symbol."""
    for c in chain:
        if getattr(c, "option_symbol", "") == option_symbol:
            return c
    return None


def _match_contract(spec: "_LegSpec", chain: list) -> Optional[str]:
    """
    Find the best-matching OCC symbol from a historical chain for a given LegSpec.
    Returns the option_symbol string or None if no acceptable match found.
    """
    if not chain:
        return None

    # Normalise the target expiration (YYYYMMDD or YYYY-MM-DD → comparable form)
    target_exp = _normalise_exp(spec.expiration)
    target_side = spec.side.lower()
    target_strike = spec.strike

    candidates = []
    for c in chain:
        c_side = (getattr(c, "side", "") or "").lower()
        if c_side != target_side:
            continue
        c_exp = _normalise_exp(str(getattr(c, "expiration", "") or ""))
        if c_exp != target_exp:
            continue
        c_strike = getattr(c, "strike", None)
        if c_strike is None:
            continue
        diff = abs(c_strike - target_strike)
        candidates.append((diff, getattr(c, "option_symbol", "")))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    best_diff, best_sym = candidates[0]
    # Accept if within $2 of target strike (handles minor rounding in LLM-generated legs)
    if best_diff <= 2.0 and best_sym:
        return best_sym
    return None


def _resolve_leg_expirations(
    leg_specs: list[_LegSpec],
    chain_contracts: list,
    entry_date: str,
) -> list[_LegSpec]:
    """Resolve symbolic expirations like nearest_weekly against historical chain dates."""
    available_expirations = sorted({
        _normalise_exp(str(getattr(c, "expiration", "") or ""))
        for c in chain_contracts
        if getattr(c, "expiration", None)
    })
    if not available_expirations:
        return leg_specs

    resolved: list[_LegSpec] = []
    cache: dict[str, str] = {}
    for spec in leg_specs:
        normalized = _normalise_exp(spec.expiration)
        if _is_concrete_expiration(normalized):
            resolved.append(_LegSpec(
                side=spec.side,
                strike=spec.strike,
                expiration=normalized,
                is_long=spec.is_long,
                role=spec.role,
            ))
            continue

        resolved_exp = cache.get(normalized)
        if resolved_exp is None:
            resolved_exp = _resolve_symbolic_expiration(normalized, available_expirations, entry_date)
            cache[normalized] = resolved_exp
        resolved.append(_LegSpec(
            side=spec.side,
            strike=spec.strike,
            expiration=resolved_exp,
            is_long=spec.is_long,
            role=spec.role,
        ))
    return resolved


def _resolve_symbolic_expiration(exp: str, available_expirations: list[str], entry_date: str) -> str:
    """Map symbolic expirations to the nearest available historical expiration."""
    if _is_concrete_expiration(exp):
        return exp
    if not available_expirations:
        return exp

    future_exps = [candidate for candidate in available_expirations if candidate >= entry_date]
    if exp == "nearest_weekly":
        return future_exps[0] if future_exps else available_expirations[0]
    return future_exps[0] if future_exps else available_expirations[0]


def _classify_missing_data(
    leg_specs: list[_LegSpec],
    valuations: list[LegValuation],
    chain_contracts: list,
) -> str:
    """Classify why repricing failed after a historical chain was found."""
    available_expirations = {
        _normalise_exp(str(getattr(c, "expiration", "") or ""))
        for c in chain_contracts
        if getattr(c, "expiration", None)
    }

    missing_specs = [spec for spec, valuation in zip(leg_specs, valuations) if valuation.realized_pnl is None]
    if not missing_specs:
        return "partial_quote_gap"

    if any(_normalise_exp(spec.expiration) not in available_expirations for spec in missing_specs):
        return "expiration_unavailable"

    if any(not valuation.option_symbol for valuation in valuations if valuation.realized_pnl is None):
        return "contract_unmatched"

    return "quote_missing_or_illiquid"


def _is_concrete_expiration(exp: str) -> bool:
    """Return True when expiration is already a concrete calendar date."""
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(exp).strip()))


def _primary_expiration(legs: dict) -> Optional[str]:
    """Extract the primary expiration date from a legs_json dict."""
    for key in ("expiration", "far_expiration", "near_expiration"):
        val = legs.get(key)
        if val:
            return str(val)
    return None


def _compute_max_risk(strategy: str, legs: dict, valuations: list[LegValuation]) -> Optional[float]:
    """
    Estimate max capital at risk per contract set based on strategy structure.
    Returns dollars (already multiplied by CONTRACT_MULTIPLIER).
    """
    try:
        if strategy == "vertical_spread":
            ls = float(legs.get("long_strike", 0))
            ss = float(legs.get("short_strike", 0))
            width = abs(ls - ss) * CONTRACT_MULTIPLIER
            # Debit spread: max risk = debit paid
            # Credit spread: max risk = width - credit received
            entry_pnl = sum(
                (v.entry_mid or 0) * CONTRACT_MULTIPLIER * (1 if v.is_long else -1)
                for v in valuations if v.entry_mid is not None
            )
            debit = abs(entry_pnl) if entry_pnl > 0 else width - abs(entry_pnl)
            return max(debit, 0.01)

        elif strategy == "iron_condor":
            pl = float(legs.get("put_long_strike", 0))
            ps = float(legs.get("put_short_strike", 0))
            cs = float(legs.get("call_short_strike", 0))
            cl = float(legs.get("call_long_strike", 0))
            put_width = abs(ps - pl) * CONTRACT_MULTIPLIER
            call_width = abs(cl - cs) * CONTRACT_MULTIPLIER
            # Net credit = sum of (premium received for shorts - premium paid for longs)
            net_credit = sum(
                (v.entry_mid or 0) * CONTRACT_MULTIPLIER * (-1 if v.is_long else 1)
                for v in valuations if v.entry_mid is not None
            )
            # Max loss = wider wing width - net credit received
            max_wing = max(put_width, call_width)
            return max(max_wing - max(net_credit, 0), 0.01)

        elif strategy in ("straddle", "strangle"):
            total_debit = sum(
                (v.entry_mid or 0) * CONTRACT_MULTIPLIER
                for v in valuations if v.is_long and v.entry_mid is not None
            )
            return max(total_debit, 0.01)

        elif strategy in ("calendar_spread", "diagonal_spread"):
            debit = sum(
                (v.entry_mid or 0) * CONTRACT_MULTIPLIER * (1 if v.is_long else -1)
                for v in valuations if v.entry_mid is not None
            )
            return max(abs(debit), 0.01)

        elif strategy == "butterfly":
            debit = sum(
                (v.entry_mid or 0) * CONTRACT_MULTIPLIER * (1 if v.is_long else -1)
                for v in valuations if v.entry_mid is not None
            )
            return max(abs(debit), 0.01)

    except Exception as exc:
        logger.debug(f"_compute_max_risk failed for {strategy}: {exc}")

    return None


def _normalise_exp(exp: str) -> str:
    """Normalise expiration to YYYY-MM-DD for consistent comparison."""
    exp = str(exp).strip()
    if re.match(r'^\d{8}$', exp):
        return f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"
    # Remove time component if present
    if "T" in exp:
        exp = exp.split("T")[0]
    return exp


def _safe_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return bid or ask


def _safe_spread(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None:
        return ask - bid
    return None


def _days_diff(date_a: str, date_b: str) -> int:
    """Return |date_a - date_b| in calendar days. Returns 0 on parse failure."""
    try:
        a = datetime.fromisoformat(date_a[:10]).date()
        b = datetime.fromisoformat(date_b[:10]).date()
        return abs((a - b).days)
    except Exception:
        return 0


def _next_trading_day(dt_str: str) -> str:
    """
    Return the next calendar day after dt_str (skipping weekends).
    This is a rough approximation; it does not account for market holidays.
    """
    try:
        d = datetime.fromisoformat(dt_str[:10]).date()
    except Exception:
        return dt_str
    d += timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d += timedelta(days=1)
    return d.isoformat()


# ── Aggregate helpers for use by research/agent.py ─────────────

def score_repriced_signals(repriced: list[SignalReprice]) -> dict:
    """
    Compute aggregate promotion fitness from a list of SignalReprice results.

    Returns a dict compatible with compute_expectancy for downstream comparison.
    """
    valid = [r for r in repriced if r.return_pct is not None]
    total = len(repriced)

    if not valid:
        missing_counts = Counter(r.rejection_code or "unknown" for r in repriced)
        return {
            "promotion_fitness": 0.0,
            "options_coverage_pct": 0.0,
            "repriced_count": 0,
            "total_count": total,
            "missing_breakdown": dict(missing_counts),
            "missing_breakdown_text": ", ".join(
                f"{count} {reason}" for reason, count in sorted(missing_counts.items())
            ) or "no repriced contracts",
        }

    returns = [r.return_pct for r in valid]
    winners = [x for x in returns if x > 0]
    losers  = [x for x in returns if x < 0]

    hit_rate = len(winners) / len(returns) * 100 if returns else 0.0
    avg_win  = sum(winners) / len(winners) if winners else 0.0
    avg_loss = sum(losers) / len(losers) if losers else 0.0
    expectancy = (hit_rate / 100 * avg_win) + ((1 - hit_rate / 100) * avg_loss)
    gross_wins = sum(winners) if winners else 0.0
    gross_losses = abs(sum(losers)) if losers else 0.0
    profit_factor = (
        gross_wins / gross_losses if gross_losses > 0
        else float("inf") if gross_wins > 0 else 0.0
    )
    coverage_pct = sum(r.data_coverage for r in repriced) / len(repriced) * 100

    # Promotion fitness: harsher than search fitness — requires real coverage
    sample_pen = min(1.0, len(valid) / 5.0)
    coverage_pen = min(1.0, coverage_pct / 80.0)  # needs ≥80% coverage
    promotion_fitness = max(0.0, expectancy * profit_factor * sample_pen * coverage_pen)
    missing_counts = Counter(r.rejection_code or "unknown" for r in repriced if r.return_pct is None)

    return {
        "promotion_fitness": round(promotion_fitness, 6),
        "options_coverage_pct": round(coverage_pct, 2),
        "repriced_count": len(valid),
        "total_count": total,
        "missing_breakdown": dict(missing_counts),
        "missing_breakdown_text": ", ".join(
            f"{count} {reason}" for reason, count in sorted(missing_counts.items())
        ) or "fully repriced",
        "hit_rate": round(hit_rate, 2),
        "expectancy": round(expectancy, 4),
        "profit_factor": round(profit_factor, 2),
    }
