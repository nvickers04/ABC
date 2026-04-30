"""Signal: Auction imbalance (NYSE/NASDAQ open & close cross).

NYSE and NASDAQ publish order-imbalance feeds in the minutes leading into
the opening cross (09:25→09:30 ET, using the wider NASDAQ window) and the
closing cross (15:50→16:00 ET).  The number is **net unmatched shares** at
the indicative cross price — positive = buy-side imbalance (price expected
to print up), negative = sell-side.

Score construction (paired-volume + dislocation blend):

    imb_component      = tanh(5  * imbalance / (paired + |imbalance|))
    dislocation_comp   = tanh(100 * (auction_price - mid) / mid)
    score              = 0.6 * imb_component + 0.4 * dislocation_comp     ∈ [-1, 1]

  Why paired-volume normalization (not ADV)?
    Desks score auction imbalance against PAIRED VOLUME at the indicative
    price.  10% of paired is real signal; 1% of paired is noise.  ADV
    conflates "big auction" with "big imbalance".  ADV is kept as a
    fallback denominator only when paired is missing/zero.

  Why include auction_price vs. mid?
    The deviation between the indicative cross price and the current quote
    is the cleanest single piece of information in the feed: it says the
    auction is about to print at a price meaningfully different from where
    the stock is trading right now.  Scaled tanh(100x) so 50bp ≈ 0.46.

Confidence:
    base = 0.5 * proximity_to_cross + 0.5 * |score|
    × 0.5 if regulatory_imbalance disagrees in sign with imbalance
        (cancellable orders dominate → less reliable)
    × 0.5 if paired_volume < 1% of ADV when ADV is known
        (auction is too thin to trust the print)

Known limitation: NYSE doesn't publish until 09:28 ET, NASDAQ from 09:25.
We use the wider 09:25 window, so NYSE-listed names will return
``no_imbalance_data`` for those first 3 minutes.  Exchange-aware windowing
requires plumbing ``exchange`` through ``Quote`` and is left for later.
"""

from __future__ import annotations

import math
from datetime import datetime, time
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from signals.base import Signal, SignalResult

_ET = ZoneInfo("America/New_York")

_OPEN_WINDOW_START = time(9, 25)
_OPEN_WINDOW_END = time(9, 30)
_OPEN_CROSS_TIME = time(9, 30)
_CLOSE_WINDOW_START = time(15, 50)
_CLOSE_WINDOW_END = time(16, 0)
_CLOSE_CROSS_TIME = time(16, 0)

# Tanh sharpness:
#   imb_pct_paired = 10%  → tanh(0.5)  ≈ 0.46
#   imb_pct_paired = 50%  → tanh(2.5)  ≈ 0.99
#   dislocation    = 50bp → tanh(0.5)  ≈ 0.46
#   dislocation    = 200bp→ tanh(2.0)  ≈ 0.96
_IMB_SCALE = 5.0
_DISLOCATION_SCALE = 100.0
_IMB_WEIGHT = 0.6
_DISLOC_WEIGHT = 0.4

_REG_DISAGREE_MULT = 0.5
_THIN_AUCTION_MULT = 0.5
_THIN_AUCTION_FRACTION_OF_ADV = 0.01


def _now_et(dt: Optional[datetime] = None) -> datetime:
    if dt is None:
        return datetime.now(_ET)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)


def _classify_window(now: datetime) -> Tuple[Optional[str], Optional[time]]:
    """Return (window_name, cross_time) or (None, None) if outside any window."""
    if now.weekday() >= 5:
        return None, None
    t = now.time()
    if _OPEN_WINDOW_START <= t < _OPEN_WINDOW_END:
        return "open", _OPEN_CROSS_TIME
    if _CLOSE_WINDOW_START <= t < _CLOSE_WINDOW_END:
        return "close", _CLOSE_CROSS_TIME
    return None, None


def _minutes_to_cross(now: datetime, cross: time) -> float:
    cross_dt = datetime.combine(now.date(), cross, tzinfo=_ET)
    return max(0.0, (cross_dt - now).total_seconds() / 60.0)


def _adv_from_daily_candles(candles_daily) -> float:
    """20-day average volume from daily candles, or 0.0 if unavailable."""
    if candles_daily is None:
        return 0.0
    try:
        if len(candles_daily) < 20:
            return 0.0
        vols = list(candles_daily.volume)[-20:]
    except (AttributeError, TypeError):
        return 0.0
    try:
        clean = [float(v) for v in vols if v is not None and float(v) > 0]
    except (TypeError, ValueError):
        return 0.0
    if len(clean) < 20:
        return 0.0
    return sum(clean) / len(clean)


def _imbalance_pct_paired(
    imbalance: float, paired: Optional[int], adv: float
) -> Tuple[float, str]:
    """Return (signed_ratio, denominator_used in {"paired","adv","none"}).

    Prefer paired-volume normalization: imb / (paired + |imb|).  If paired
    is missing or zero, fall back to imb / adv.  If both are unusable,
    return (0.0, "none").
    """
    if paired is not None and paired > 0:
        denom = float(paired) + abs(imbalance)
        if denom > 0:
            return imbalance / denom, "paired"
    if adv > 0:
        return imbalance / adv, "adv"
    return 0.0, "none"


def _dislocation(auction_price: Optional[float], mid: Optional[float]) -> float:
    """Signed (auction_price - mid) / mid.  Returns 0.0 if either is missing."""
    if auction_price is None or mid is None or mid <= 0:
        return 0.0
    return (auction_price - mid) / mid


def compute_auction_score(
    *,
    auction_imbalance: Optional[float],
    auction_volume: Optional[int],
    auction_price: Optional[float],
    regulatory_imbalance: Optional[float],
    adv: float,
    now: datetime,
    mid: Optional[float] = None,
) -> SignalResult:
    """Pure function — score auction imbalance given fully resolved inputs."""
    window, cross = _classify_window(now)
    if window is None:
        return SignalResult(0.0, 0.0, {"abstain": "outside_auction_window"})

    if auction_imbalance is None:
        return SignalResult(0.0, 0.0, {
            "window": window,
            "abstain": "no_imbalance_data",
        })

    ratio, denom_used = _imbalance_pct_paired(auction_imbalance, auction_volume, adv)
    if denom_used == "none":
        return SignalResult(0.0, 0.0, {
            "window": window,
            "abstain": "no_paired_or_adv",
        })

    dislocation = _dislocation(auction_price, mid)

    imb_component = math.tanh(ratio * _IMB_SCALE)
    disloc_component = math.tanh(dislocation * _DISLOCATION_SCALE)
    score = _IMB_WEIGHT * imb_component + _DISLOC_WEIGHT * disloc_component
    score = max(-1.0, min(1.0, score))  # numerical guard

    if window == "open":
        window_minutes = (
            datetime.combine(now.date(), _OPEN_WINDOW_END, tzinfo=_ET)
            - datetime.combine(now.date(), _OPEN_WINDOW_START, tzinfo=_ET)
        ).total_seconds() / 60.0
    else:
        window_minutes = (
            datetime.combine(now.date(), _CLOSE_WINDOW_END, tzinfo=_ET)
            - datetime.combine(now.date(), _CLOSE_WINDOW_START, tzinfo=_ET)
        ).total_seconds() / 60.0
    mins_to_cross = _minutes_to_cross(now, cross)
    proximity = 1.0 - min(mins_to_cross / window_minutes, 1.0) if window_minutes > 0 else 1.0
    confidence = 0.5 * proximity + 0.5 * abs(score)

    reg_disagrees = (
        regulatory_imbalance is not None
        and regulatory_imbalance != 0
        and (regulatory_imbalance > 0) != (auction_imbalance > 0)
    )
    if reg_disagrees:
        confidence *= _REG_DISAGREE_MULT

    thin_auction = (
        auction_volume is not None
        and adv > 0
        and float(auction_volume) < _THIN_AUCTION_FRACTION_OF_ADV * adv
    )
    if thin_auction:
        confidence *= _THIN_AUCTION_MULT

    components = {
        "mode": "primary_paired" if denom_used == "paired" else "proxy_adv" if denom_used == "adv" else "none",
        "window": window,
        "imbalance_shares": int(auction_imbalance),
        "paired_shares": int(auction_volume) if auction_volume is not None else None,
        "imbalance_pct_paired": float(round(ratio * 100.0, 4)),
        "denominator": denom_used,
        "auction_price": float(auction_price) if auction_price is not None else None,
        "mid": float(mid) if mid is not None else None,
        "dislocation_bps": float(round(dislocation * 10_000.0, 2)),
        "regulatory_imbalance": (
            int(regulatory_imbalance) if regulatory_imbalance is not None else None
        ),
        "regulatory_disagrees": bool(reg_disagrees),
        "thin_auction": bool(thin_auction),
        "minutes_to_cross": float(round(mins_to_cross, 2)),
        "imb_component": float(round(imb_component, 4)),
        "dislocation_component": float(round(disloc_component, 4)),
    }
    return SignalResult(score, confidence, components)


class AuctionImbalanceSignal(Signal):
    name = "auction_imbalance"
    category = "microstructure"
    data_source = "mda_quotes"
    refresh_rate = "every_round"
    tier = 1
    # NYSE auction imbalance is only available via IBKR generic ticks.
    # The research daemon has IBKR disabled (it shares MDA), so this
    # signal would emit zeros there; only the trader runs it.
    requires_ibkr = True
    # Auction print resolves within minutes of the cross; override the
    # microstructure default (5min × 6 bars) with 1-min × 30.
    return_resolution = "1min"
    return_horizon = 30
    return_lookback_days = 5

    def compute(self, symbol: str, data: dict) -> SignalResult:
        quote = data.get("quote")
        if quote is None:
            return SignalResult(0.0, 0.0, {"abstain": "no_quote"})

        adv = _adv_from_daily_candles(data.get("candles_daily"))
        now = _now_et(data.get("now"))

        mid = getattr(quote, "mid", None)
        if mid is None:
            mid = getattr(quote, "last", None)

        return compute_auction_score(
            auction_imbalance=getattr(quote, "auction_imbalance", None),
            auction_volume=getattr(quote, "auction_volume", None),
            auction_price=getattr(quote, "auction_price", None),
            regulatory_imbalance=getattr(quote, "regulatory_imbalance", None),
            adv=adv,
            now=now,
            mid=mid,
        )
