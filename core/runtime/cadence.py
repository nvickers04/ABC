"""
Session-aware cadence for the research daemon.

Returns the seconds-between-rounds the daemon should sleep based on
the current ET wall-clock.  Pure-function helpers — no IBKR, no DB —
so it's trivially testable and can run in any process.

Tiers (Mon–Fri only; weekends always overnight):

    * Regular hours      09:30 – 16:00 ET   →  10s
    * Active extended    07:00 – 09:30 ET   →  30s
                         16:00 – 18:00 ET   →  30s
    * Quiet extended     04:00 – 07:00 ET   →  300s
                         18:00 – 20:00 ET   →  300s
    * Overnight          20:00 – 04:00 ET   →  1800s  (30 min)
    * Weekend            anytime Sat/Sun    →  1800s

Why these numbers — credit-budget reasoning
-------------------------------------------
A scoring round burns ~120 MarketData credits and takes ~65s wall-clock
end-to-end (Tier 1 + Tier 2 + Tier 3).  Effective cycle ≈ cadence + 65s.

The MDA Trader plan budget is 100,000 credits/day.  With trader uptime
~07:00–20:00 ET (13h), this allocation keeps us under budget while
maximising freshness when the trader is most active:

    REGULAR          6.5h  ×  ~46 rounds/h  ×  120 cr  ≈  35,900 cr
    ACTIVE_EXTENDED  4.5h  ×  ~38 rounds/h  ×  120 cr  ≈  20,500 cr
    QUIET_EXTENDED   5.0h  ×  ~10 rounds/h  ×  120 cr  ≈   6,000 cr
    OVERNIGHT        8.0h  ×  ~ 2 rounds/h  ×  120 cr  ≈   1,900 cr
    -------------------------------------------------------------
    Total                                              ≈  64,300 cr

That leaves ~35k credits of headroom for credit spikes (universe growth,
larger option chains, retry storms).  The MDA client's circuit breaker
short-circuits all calls once credits hit zero, so going over still
fails safe — but planning under headroom lets us avoid that path.

Why these tiers map to real signal value
----------------------------------------
* REGULAR 10s — all the intraday signals (momentum, mean_reversion,
  opening_range, breakout, gap, vwap, volume_clock, quote_stability,
  spread_dynamics, option_flow, gamma_exposure, put_call_ratio,
  iv_change, news_sentiment, market_breadth, sector_momentum) decay
  on a sub-minute scale.  10s sleep ≈ saturation given the 65s round
  floor.

* ACTIVE_EXTENDED 30s — 07:00–09:30 is gap formation and news
  digestion; 16:00–18:00 is earnings reactions.  Quote / IV / sentiment
  signals matter and the trader is awake.

* QUIET_EXTENDED 300s — 04:00–07:00 has thin liquidity (mostly noise);
  18:00–20:00 is post-earnings drift.  5 min is plenty.

* OVERNIGHT 1800s — only fundamentals (debt_health, valuation, quality,
  revenue_growth, cash_flow_yield), insider/institutional ownership,
  short interest, earnings calendar, seasonality, and beta_adjusted
  signals are still meaningful while the market is closed, and none of
  those move overnight.  30 min keeps them fresh enough for the next
  open.

These thresholds are the source of truth for daemon cadence.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo


_ET = ZoneInfo("America/New_York")

# Boundaries (ET local time-of-day).
_QUIET_PREMARKET_OPEN = time(4, 0)
_ACTIVE_PREMARKET_OPEN = time(7, 0)
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)
_ACTIVE_POSTMARKET_CLOSE = time(18, 0)
_QUIET_POSTMARKET_CLOSE = time(20, 0)

# Cadence values (seconds).  See module docstring for budget reasoning.
CADENCE_REGULAR_S: int = 10
CADENCE_ACTIVE_EXTENDED_S: int = 30
CADENCE_QUIET_EXTENDED_S: int = 5 * 60
CADENCE_OVERNIGHT_S: int = 30 * 60

# Backward-compatibility alias — older code (and the heartbeat staleness
# threshold) imported CADENCE_EXTENDED_S; map it to the active tier so
# behaviour is unchanged or stricter, never looser.
CADENCE_EXTENDED_S: int = CADENCE_ACTIVE_EXTENDED_S


def _now_et(dt: Optional[datetime] = None) -> datetime:
    if dt is None:
        return datetime.now(_ET)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)


def session_label(dt: Optional[datetime] = None) -> str:
    """Return one of:
        'regular'           — 09:30–16:00 ET, weekday
        'active_extended'   — 07:00–09:30 or 16:00–18:00 ET, weekday
        'quiet_extended'    — 04:00–07:00 or 18:00–20:00 ET, weekday
        'overnight'         — anything else (incl. weekends)

    Weekends are always 'overnight'.
    """
    now = _now_et(dt)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return "overnight"
    t = now.time()
    if _REGULAR_OPEN <= t < _REGULAR_CLOSE:
        return "regular"
    if _ACTIVE_PREMARKET_OPEN <= t < _REGULAR_OPEN:
        return "active_extended"
    if _REGULAR_CLOSE <= t < _ACTIVE_POSTMARKET_CLOSE:
        return "active_extended"
    if _QUIET_PREMARKET_OPEN <= t < _ACTIVE_PREMARKET_OPEN:
        return "quiet_extended"
    if _ACTIVE_POSTMARKET_CLOSE <= t < _QUIET_POSTMARKET_CLOSE:
        return "quiet_extended"
    return "overnight"


def cadence_seconds(dt: Optional[datetime] = None) -> int:
    """Seconds the research daemon should sleep before the next round."""
    label = session_label(dt)
    if label == "regular":
        return CADENCE_REGULAR_S
    if label == "active_extended":
        return CADENCE_ACTIVE_EXTENDED_S
    if label == "quiet_extended":
        return CADENCE_QUIET_EXTENDED_S
    return CADENCE_OVERNIGHT_S


# Tiered universe coverage — base = RESEARCH_UNIVERSE, focus = symbols
# the trader is actively engaged with (positions, watching, attention).
# Focus is scored EVERY round.  Base is scored once every N rounds.
#
# Effective base freshness  = N × (round_wallclock + cadence_seconds)
#
# During regular hours: round ~65s + 10s sleep ≈ 75s, N=3 ⇒ base every
# ~225s (~3.75 min).  Focus stays at ~75s freshness.  This is the right
# trade: a held position deserves 75s freshness; AVGO sitting in the
# base universe with no trader interest can wait 4 minutes.
#
# Outside regular hours we run base every round — the cadence sleep is
# already long enough that there's no credit win from skipping it, and
# overnight signals (fundamentals/insider/seasonality) don't change
# fast enough for tiering to matter.

BASE_EVERY_N_REGULAR: int = 3
BASE_EVERY_N_ACTIVE_EXTENDED: int = 1
BASE_EVERY_N_QUIET_EXTENDED: int = 1
BASE_EVERY_N_OVERNIGHT: int = 1


def base_universe_every_n_rounds(dt: Optional[datetime] = None) -> int:
    """How often to include the static base universe in a scoring round.

    Returns 1 = every round, 3 = every third round, etc.  The daemon
    always scores the focus set (positions/attention) every round; this
    governs only the base.
    """
    label = session_label(dt)
    if label == "regular":
        return BASE_EVERY_N_REGULAR
    if label == "active_extended":
        return BASE_EVERY_N_ACTIVE_EXTENDED
    if label == "quiet_extended":
        return BASE_EVERY_N_QUIET_EXTENDED
    return BASE_EVERY_N_OVERNIGHT
