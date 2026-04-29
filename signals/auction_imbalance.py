"""Signal: Auction imbalance (NYSE/NASDAQ open & close cross).

NYSE and NASDAQ publish order-imbalance feeds in the minutes leading into
the opening cross (09:25→09:30 ET, using the wider NASDAQ window) and the
closing cross (15:50→16:00 ET).  The number is **net unmatched shares** at
the indicative cross price — positive = buy-side imbalance (price expected
to print up), negative = sell-side.

Inputs (from `data` dict):
  - ``quote``:         data_provider.Quote with ``auction_imbalance``,
                        ``auction_volume``, ``auction_price``,
                        ``regulatory_imbalance`` populated when in window.
  - ``candles_daily``: list of daily candles; we use the last 20 ``volume``
                        bars as ADV.  20 is the convention elsewhere.

Outside the active windows (or on weekends, or when the imbalance feed is
silent): score=0.0, confidence=0.0.  We never trade off auction data
outside the publication window.
"""

from __future__ import annotations

import math
from datetime import datetime, time
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from signals.base import Signal, SignalResult

_ET = ZoneInfo("America/New_York")

# Window definitions (ET, weekdays only).
# We use the WIDER NASDAQ window for the open (09:25) so we never miss
# imbalance data on NASDAQ-listed names; the cost is ~3 minutes of
# low-confidence scores for NYSE-listed names that publish at 09:28.
_OPEN_WINDOW_START = time(9, 25)
_OPEN_WINDOW_END = time(9, 30)
_OPEN_CROSS_TIME = time(9, 30)
_CLOSE_WINDOW_START = time(15, 50)
_CLOSE_WINDOW_END = time(16, 0)
_CLOSE_CROSS_TIME = time(16, 0)

# Score scaling: tanh(ratio * SCALE).  ratio = imbalance / ADV.
# At SCALE=50, |imbalance|=2% of ADV → tanh(1.0) ≈ 0.76.
# At |imbalance|=4% of ADV → tanh(2.0) ≈ 0.96.
_SCORE_SCALE = 50.0


def _now_et(dt: Optional[datetime] = None) -> datetime:
    if dt is None:
        return datetime.now(_ET)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)


def _classify_window(now: datetime) -> Tuple[Optional[str], Optional[time]]:
    """Return (window_name, cross_time) or (None, None) if outside any window.

    window_name in {"open", "close"}.  Weekends → (None, None).
    """
    if now.weekday() >= 5:  # Sat/Sun
        return None, None
    t = now.time()
    if _OPEN_WINDOW_START <= t < _OPEN_WINDOW_END:
        return "open", _OPEN_CROSS_TIME
    if _CLOSE_WINDOW_START <= t < _CLOSE_WINDOW_END:
        return "close", _CLOSE_CROSS_TIME
    return None, None


def _minutes_to_cross(now: datetime, cross: time) -> float:
    """Whole minutes (float) from ``now`` to ``cross`` on the same date.

    Always non-negative; returns 0.0 if past the cross (defensive — we
    only call this when `now` is inside the window so this branch is
    informational).
    """
    cross_dt = datetime.combine(now.date(), cross, tzinfo=_ET)
    delta_s = (cross_dt - now).total_seconds()
    return max(0.0, delta_s / 60.0)


def _adv_from_daily_candles(candles_daily) -> float:
    """20-day average volume from daily candles, or 0.0 if unavailable."""
    if candles_daily is None:
        return 0.0
    try:
        # Candles is a dataclass with .volume list; len() returns bar count.
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


def compute_auction_score(
    *,
    auction_imbalance: Optional[float],
    auction_volume: Optional[int],
    auction_price: Optional[float],
    regulatory_imbalance: Optional[float],
    adv: float,
    now: datetime,
) -> SignalResult:
    """Pure function — score auction imbalance given fully resolved inputs.

    Exposed at module scope so tests can drive it directly without
    reconstructing a full data dict.
    """
    window, cross = _classify_window(now)
    if window is None:
        return SignalResult(0.0, 0.0, {"abstain": "outside_auction_window"})

    if auction_imbalance is None:
        return SignalResult(0.0, 0.0, {
            "window": window,
            "abstain": "no_imbalance_data",
        })

    if adv <= 0:
        return SignalResult(0.0, 0.0, {
            "window": window,
            "abstain": "no_adv",
        })

    ratio = auction_imbalance / adv
    score = math.tanh(ratio * _SCORE_SCALE)

    # Confidence: 50% from proximity to the cross + 50% from magnitude.
    window_minutes = (
        (datetime.combine(now.date(), _OPEN_WINDOW_END, tzinfo=_ET) -
         datetime.combine(now.date(), _OPEN_WINDOW_START, tzinfo=_ET)).total_seconds() / 60.0
        if window == "open"
        else (datetime.combine(now.date(), _CLOSE_WINDOW_END, tzinfo=_ET) -
              datetime.combine(now.date(), _CLOSE_WINDOW_START, tzinfo=_ET)).total_seconds() / 60.0
    )
    mins_to_cross = _minutes_to_cross(now, cross)
    proximity = 1.0 - min(mins_to_cross / window_minutes, 1.0) if window_minutes > 0 else 1.0
    magnitude = min(abs(ratio) * _SCORE_SCALE, 1.0)
    confidence = 0.5 * proximity + 0.5 * magnitude

    components = {
        "window": window,
        "imbalance_shares": int(auction_imbalance),
        "paired_shares": int(auction_volume) if auction_volume is not None else None,
        "imbalance_pct_adv": float(round(ratio * 100.0, 4)),
        "auction_price": float(auction_price) if auction_price is not None else None,
        "regulatory_imbalance": (
            int(regulatory_imbalance) if regulatory_imbalance is not None else None
        ),
        "minutes_to_cross": float(round(mins_to_cross, 2)),
    }
    return SignalResult(score, confidence, components)


class AuctionImbalanceSignal(Signal):
    name = "auction_imbalance"
    category = "microstructure"
    data_source = "mda_quotes"
    refresh_rate = "every_round"
    tier = 1
    # Auction print resolves within minutes of the cross; override the
    # microstructure default (5min × 6 bars = 30 min) to fine-grained
    # 1-min × 30 (still ~30 minutes, but lets IC catch the print bar).
    return_resolution = "1min"
    return_horizon = 30
    return_lookback_days = 5

    def compute(self, symbol: str, data: dict) -> SignalResult:
        quote = data.get("quote")
        if quote is None:
            return SignalResult(0.0, 0.0, {"abstain": "no_quote"})

        adv = _adv_from_daily_candles(data.get("candles_daily"))
        now = _now_et(data.get("now"))  # tests inject "now"; production uses real clock

        return compute_auction_score(
            auction_imbalance=getattr(quote, "auction_imbalance", None),
            auction_volume=getattr(quote, "auction_volume", None),
            auction_price=getattr(quote, "auction_price", None),
            regulatory_imbalance=getattr(quote, "regulatory_imbalance", None),
            adv=adv,
            now=now,
        )
