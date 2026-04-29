"""Tests for signals.auction_imbalance — opening/closing cross signal."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from signals.auction_imbalance import (
    AuctionImbalanceSignal,
    compute_auction_score,
    _classify_window,
    _adv_from_daily_candles,
)


_ET = ZoneInfo("America/New_York")


def _et(year, month, day, hour, minute, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=_ET)


def _make_daily_candles(n=20, vol=1_000_000):
    """Minimal candles-like object with .volume of length n and len()."""
    closes = [100.0] * n

    class _C:
        pass

    c = _C()
    c.close = closes
    c.high = [101.0] * n
    c.low = [99.0] * n
    c.open = [100.0] * n
    c.volume = [vol] * n
    _C.__len__ = lambda self: n
    return c


def _quote(*, imb=None, vol=None, price=None, reg=None):
    """Build a Quote-like object exposing the auction fields."""
    return SimpleNamespace(
        symbol="AAPL",
        last=100.0,
        bid=99.95,
        ask=100.05,
        volume=1_000_000,
        auction_imbalance=imb,
        auction_volume=vol,
        auction_price=price,
        regulatory_imbalance=reg,
    )


# ── _classify_window ─────────────────────────────────────────────────

@pytest.mark.parametrize("dt,expected", [
    (_et(2026, 4, 28, 9, 25),  "open"),    # NASDAQ open-window start
    (_et(2026, 4, 28, 9, 28),  "open"),    # NYSE open-window start
    (_et(2026, 4, 28, 9, 29),  "open"),
    (_et(2026, 4, 28, 9, 30),  None),      # cross — outside (END is exclusive)
    (_et(2026, 4, 28, 15, 50), "close"),
    (_et(2026, 4, 28, 15, 59), "close"),
    (_et(2026, 4, 28, 16, 0),  None),      # cross — outside
    (_et(2026, 4, 28, 11, 0),  None),      # mid-day
    (_et(2026, 4, 28, 9, 24),  None),      # 1 min before window
])
def test_classify_window_weekday(dt, expected):
    window, _ = _classify_window(dt)
    assert window == expected


@pytest.mark.parametrize("dt", [
    _et(2026, 5, 2, 9, 28),   # Saturday open-window time
    _et(2026, 5, 3, 15, 55),  # Sunday close-window time
])
def test_classify_window_weekend_inactive(dt):
    assert _classify_window(dt) == (None, None)


# ── compute_auction_score core ───────────────────────────────────────

def test_outside_window_scores_zero_zero():
    r = compute_auction_score(
        auction_imbalance=500_000.0,
        auction_volume=1_000_000,
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 11, 0),
    )
    assert r.score == 0.0
    assert r.confidence == 0.0
    assert r.components["abstain"] == "outside_auction_window"


def test_open_window_positive_imbalance_score_positive():
    r = compute_auction_score(
        auction_imbalance=200_000.0,
        auction_volume=500_000,
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    assert r.score > 0.0
    assert r.confidence > 0.0
    assert r.components["window"] == "open"


def test_close_window_negative_imbalance_score_negative():
    r = compute_auction_score(
        auction_imbalance=-200_000.0,
        auction_volume=500_000,
        auction_price=100.0,
        regulatory_imbalance=-150_000.0,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 15, 55),
    )
    assert r.score < 0.0
    assert r.confidence > 0.0
    assert r.components["window"] == "close"
    assert r.components["regulatory_imbalance"] == -150_000


@pytest.mark.parametrize("imb,sign", [
    (50_000.0, 1),
    (-50_000.0, -1),
    (1_000_000.0, 1),
    (-1_000_000.0, -1),
])
def test_score_sign_matches_imbalance_sign(imb, sign):
    r = compute_auction_score(
        auction_imbalance=imb,
        auction_volume=100_000,
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    assert (r.score > 0) == (sign > 0)
    assert (r.score < 0) == (sign < 0)


def test_score_monotone_in_magnitude():
    base = dict(auction_volume=100_000, auction_price=100.0,
                regulatory_imbalance=None, adv=10_000_000.0,
                now=_et(2026, 4, 28, 9, 29))
    s_small = compute_auction_score(auction_imbalance=50_000.0, **base).score
    s_med = compute_auction_score(auction_imbalance=200_000.0, **base).score
    s_big = compute_auction_score(auction_imbalance=500_000.0, **base).score
    assert 0 < s_small < s_med < s_big


def test_confidence_increases_toward_cross():
    base = dict(auction_imbalance=200_000.0, auction_volume=100_000,
                auction_price=100.0, regulatory_imbalance=None,
                adv=10_000_000.0)
    early = compute_auction_score(now=_et(2026, 4, 28, 9, 25, 30), **base).confidence
    mid = compute_auction_score(now=_et(2026, 4, 28, 9, 27, 30), **base).confidence
    late = compute_auction_score(now=_et(2026, 4, 28, 9, 29, 50), **base).confidence
    assert early < mid < late


def test_abstains_when_imbalance_missing():
    r = compute_auction_score(
        auction_imbalance=None,
        auction_volume=100_000,
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    assert r.score == 0.0 and r.confidence == 0.0
    assert r.components["abstain"] == "no_imbalance_data"
    assert r.components["window"] == "open"


def test_abstains_when_paired_and_adv_both_zero():
    """With no paired volume and no ADV we have no denominator at all."""
    r = compute_auction_score(
        auction_imbalance=200_000.0,
        auction_volume=0,        # no paired
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=0.0,                 # no ADV
        now=_et(2026, 4, 28, 9, 29),
    )
    assert r.score == 0.0 and r.confidence == 0.0
    assert r.components["abstain"] == "no_paired_or_adv"


def test_falls_back_to_adv_when_paired_missing():
    """When paired_volume is None, denominator should be ADV (with a flag)."""
    r = compute_auction_score(
        auction_imbalance=200_000.0,
        auction_volume=None,
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    assert r.components["denominator"] == "adv"
    assert r.score > 0


def test_uses_paired_when_present():
    r = compute_auction_score(
        auction_imbalance=200_000.0,
        auction_volume=500_000,
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    assert r.components["denominator"] == "paired"


def test_components_dict_keys():
    r = compute_auction_score(
        auction_imbalance=200_000.0,
        auction_volume=500_000,
        auction_price=100.0,
        regulatory_imbalance=180_000.0,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
        mid=100.0,
    )
    assert set(r.components.keys()) == {
        "window", "imbalance_shares", "paired_shares",
        "imbalance_pct_paired", "denominator",
        "auction_price", "mid", "dislocation_bps",
        "regulatory_imbalance", "regulatory_disagrees",
        "thin_auction", "minutes_to_cross",
        "imb_component", "dislocation_component",
    }


# ── _adv_from_daily_candles ──────────────────────────────────────────

def test_adv_zero_when_no_candles():
    assert _adv_from_daily_candles(None) == 0.0


def test_adv_zero_when_too_few_bars():
    c = _make_daily_candles(n=10)
    assert _adv_from_daily_candles(c) == 0.0


def test_adv_averages_last_20_bars():
    c = _make_daily_candles(n=25, vol=2_000_000)
    assert _adv_from_daily_candles(c) == 2_000_000.0


# ── End-to-end: AuctionImbalanceSignal.compute ───────────────────────

def test_signal_compute_full_path():
    sig = AuctionImbalanceSignal()
    data = {
        "quote": _quote(imb=300_000.0, vol=500_000, price=100.0, reg=290_000.0),
        "candles_daily": _make_daily_candles(n=20, vol=10_000_000),
        "now": _et(2026, 4, 28, 9, 29),
    }
    out = sig.compute("AAPL", data)
    assert out.score > 0
    assert out.confidence > 0
    assert out.components["window"] == "open"
    assert out.components["imbalance_shares"] == 300_000


def test_signal_compute_abstains_when_no_quote():
    sig = AuctionImbalanceSignal()
    out = sig.compute("AAPL", {
        "candles_daily": _make_daily_candles(),
        "now": _et(2026, 4, 28, 9, 29),
    })
    assert out.score == 0.0 and out.confidence == 0.0
    assert out.components["abstain"] == "no_quote"


def test_signal_registered():
    """Ensure the auto-registry picked up AuctionImbalanceSignal."""
    from signals.base import SIGNAL_REGISTRY
    assert "auction_imbalance" in SIGNAL_REGISTRY


def test_signal_score_wrapper_clamps_and_catches():
    """The base Signal.score() wrapper should never raise."""
    sig = AuctionImbalanceSignal()
    out = sig.score("AAPL", {})  # missing everything
    assert out["score"] == 0.0
    assert out["confidence"] == 0.0


# ── Rewrite-specific behaviors ────────────────────────────────────────

def test_dislocation_reinforces_score_when_aligned():
    """auction_price > mid + positive imbalance → score larger than imbalance alone."""
    base = dict(
        auction_imbalance=200_000.0,
        auction_volume=500_000,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    no_disloc = compute_auction_score(auction_price=100.0, mid=100.0, **base).score
    with_disloc = compute_auction_score(auction_price=100.5, mid=100.0, **base).score
    assert with_disloc > no_disloc > 0


def test_dislocation_dampens_score_when_opposed():
    """auction_price < mid + positive imbalance → score should shrink."""
    base = dict(
        auction_imbalance=200_000.0,
        auction_volume=500_000,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    aligned = compute_auction_score(auction_price=100.0, mid=100.0, **base).score
    opposed = compute_auction_score(auction_price=99.5, mid=100.0, **base).score
    assert opposed < aligned


def test_regulatory_disagree_halves_confidence():
    """If regulatory_imbalance disagrees in sign, confidence is halved."""
    base = dict(
        auction_imbalance=200_000.0,
        auction_volume=500_000,
        auction_price=100.0,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    agree = compute_auction_score(regulatory_imbalance=180_000.0, **base)
    disagree = compute_auction_score(regulatory_imbalance=-180_000.0, **base)
    assert disagree.confidence == pytest.approx(agree.confidence * 0.5, rel=1e-6)
    assert disagree.components["regulatory_disagrees"] is True
    assert agree.components["regulatory_disagrees"] is False


def test_thin_auction_halves_confidence():
    """When paired_volume < 1% of ADV, confidence is halved."""
    base = dict(
        auction_imbalance=200_000.0,
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,                  # 1% of ADV = 100k
        now=_et(2026, 4, 28, 9, 29),
    )
    thin = compute_auction_score(auction_volume=50_000, **base)        # < 100k → thin
    healthy = compute_auction_score(auction_volume=500_000, **base)    # ≥ 100k
    assert thin.components["thin_auction"] is True
    assert healthy.components["thin_auction"] is False
    assert thin.confidence < healthy.confidence


def test_paired_normalization_more_sensitive_than_adv():
    """Paired-volume denominator should yield a much larger ratio than ADV."""
    paired = compute_auction_score(
        auction_imbalance=200_000.0,
        auction_volume=500_000,            # ratio = 200k / 700k ≈ 28.6%
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    adv_fallback = compute_auction_score(
        auction_imbalance=200_000.0,
        auction_volume=None,               # ratio = 200k / 10M = 2%
        auction_price=100.0,
        regulatory_imbalance=None,
        adv=10_000_000.0,
        now=_et(2026, 4, 28, 9, 29),
    )
    assert paired.components["imbalance_pct_paired"] > adv_fallback.components["imbalance_pct_paired"] * 5


def test_score_in_bounds():
    """Score should always be in [-1, 1] even with extreme inputs."""
    r = compute_auction_score(
        auction_imbalance=10_000_000.0,
        auction_volume=1,                  # ratio ≈ 1.0
        auction_price=200.0,               # 100% dislocation
        regulatory_imbalance=10_000_000.0,
        adv=100_000.0,
        now=_et(2026, 4, 28, 9, 29),
        mid=100.0,
    )
    assert -1.0 <= r.score <= 1.0
    assert 0.0 <= r.confidence <= 1.0
