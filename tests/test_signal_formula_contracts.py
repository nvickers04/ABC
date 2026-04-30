"""Signal formula contract tests.

These tests pin cross-family output invariants introduced by the formula
standardization work:
- score is finite and bounded [-1, 1]
- confidence is finite and bounded [0, 1]
- score/confidence keys are always present
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _make_candles(n: int = 120, start: float = 100.0):
    rng = np.random.default_rng(7)
    close = []
    price = start
    for _ in range(n):
        price = max(1.0, price * (1.0 + rng.normal(0.0008, 0.01)))
        close.append(float(price))
    high = [c * (1.0 + abs(rng.normal(0.0, 0.004))) for c in close]
    low = [c * (1.0 - abs(rng.normal(0.0, 0.004))) for c in close]
    open_ = [close[0]] + close[:-1]
    volume = [int(1_000_000 * max(0.2, 1.0 + rng.normal(0.0, 0.3))) for _ in close]

    class Candles:
        pass

    c = Candles()
    c.close = close
    c.high = high
    c.low = low
    c.open = open_
    c.volume = volume
    Candles.__len__ = lambda self: len(close)
    return c


def _build_common_data():
    candles = _make_candles(140)
    daily = _make_candles(140, start=90.0)
    quote = SimpleNamespace(
        bid=99.8, ask=100.2, last=100.0, mid=100.0, volume=1_800_000,
        auction_imbalance=50_000, auction_volume=600_000, auction_price=100.1,
        regulatory_imbalance=20_000,
    )
    fundamentals = SimpleNamespace(beta=1.15, institutional_pct=67.0)
    return {
        "candles": candles,
        "candles_daily": daily,
        "spy_candles": _make_candles(140, start=400.0),
        "qqq_candles": _make_candles(140, start=350.0),
        "quote": quote,
        "fundamentals": fundamentals,
    }


def _build_vol_data():
    data = _build_common_data()
    contracts = []
    for i, strike in enumerate([95, 100, 105]):
        for side in ("call", "put"):
            contracts.append(
                SimpleNamespace(
                    side=side,
                    strike=float(strike),
                    mid=2.0 + i * 0.4,
                    iv=28.0 + i * 2.0 + (2.0 if side == "put" else 0.0),
                    delta=0.45 if side == "call" else -0.45,
                    dte=14 + i * 14,
                    expiration=f"2026-0{i+6}-20",
                    open_interest=1200 + i * 300,
                    volume=400 + i * 120,
                    gamma=0.02 + i * 0.004,
                    underlying_price=100.0,
                )
            )

    class OptionChain:
        def __init__(self, contracts):
            self.contracts = contracts

        def calls(self):
            return [c for c in self.contracts if c.side == "call"]

        def puts(self):
            return [c for c in self.contracts if c.side == "put"]

    data["option_chain"] = OptionChain(contracts)
    data["iv_info"] = SimpleNamespace(
        iv_current=35.0,
        iv_rank=72.0,
        iv_high=52.0,
        iv_low=18.0,
    )
    return data


def _build_micro_data():
    data = _build_vol_data()
    rng = np.random.default_rng(11)
    bars = []
    px = 100.0
    for _ in range(40):
        px = max(1.0, px * (1.0 + rng.normal(0.0, 0.003)))
        hi = px * (1.0 + abs(rng.normal(0.0, 0.0015)))
        lo = px * (1.0 - abs(rng.normal(0.0, 0.0015)))
        bars.append(SimpleNamespace(high=hi, low=lo, close=px, volume=int(600_000 * (1 + rng.normal(0, 0.2)))))
    data["candles"] = bars
    return data


@pytest.mark.parametrize(
    "module_name,signal_name",
    [
        ("signals.momentum", "momentum"),
        ("signals.mean_reversion", "mean_reversion"),
        ("signals.breakout", "breakout"),
        ("signals.vwap", "vwap_deviation"),
        ("signals.volume", "volume_profile"),
        ("signals.beta_adjusted_momentum", "beta_adjusted_momentum"),
        ("signals.gap", "overnight_gap"),
        ("signals.price_acceleration", "price_acceleration"),
        ("signals.volume_weighted_momentum", "volume_weighted_momentum"),
        ("signals.multi_timeframe", "multi_timeframe"),
        ("signals.opening_range", "opening_range"),
        ("signals.trend_strength", "trend_strength"),
        ("signals.support_resistance", "support_resistance"),
    ],
)
def test_price_family_contract_bounds(module_name, signal_name):
    __import__(module_name)
    from signals.base import SIGNAL_REGISTRY

    sig = SIGNAL_REGISTRY[signal_name]
    out = sig.score("AAPL", _build_common_data())

    assert "score" in out and "confidence" in out and "components" in out
    assert np.isfinite(out["score"])
    assert np.isfinite(out["confidence"])
    assert -1.0 <= out["score"] <= 1.0
    assert 0.0 <= out["confidence"] <= 1.0


@pytest.mark.parametrize(
    "module_name,signal_name",
    [
        ("signals.iv_rank", "iv_rank"),
        ("signals.iv_rv_spread", "iv_rv_spread"),
        ("signals.term_structure", "term_structure"),
        ("signals.skew", "volatility_skew"),
        ("signals.iv_change", "iv_rank_momentum"),
        ("signals.gamma_exposure", "gamma_exposure"),
        ("signals.put_call_ratio", "put_call_ratio"),
        ("signals.option_volume_ratio", "option_volume_surge"),
        ("signals.straddle_cost", "straddle_cost"),
        ("signals.realized_vol_cone", "realized_vol_cone"),
    ],
)
def test_volatility_family_contract_bounds(module_name, signal_name):
    __import__(module_name)
    from signals.base import SIGNAL_REGISTRY

    sig = SIGNAL_REGISTRY[signal_name]
    out = sig.score("AAPL", _build_vol_data())

    assert "score" in out and "confidence" in out and "components" in out
    assert np.isfinite(out["score"])
    assert np.isfinite(out["confidence"])
    assert -1.0 <= out["score"] <= 1.0
    assert 0.0 <= out["confidence"] <= 1.0


@pytest.mark.parametrize(
    "module_name,signal_name",
    [
        ("signals.spread_dynamics", "spread_dynamics"),
        ("signals.quote_stability", "quote_stability"),
        ("signals.volume_clock", "volume_clock"),
        ("signals.option_flow", "option_flow"),
        ("signals.institutional_flow", "institutional_flow"),
        ("signals.auction_imbalance", "auction_imbalance"),
    ],
)
def test_microstructure_family_contract_bounds(module_name, signal_name):
    __import__(module_name)
    from signals.base import SIGNAL_REGISTRY

    sig = SIGNAL_REGISTRY[signal_name]
    out = sig.score("AAPL", _build_micro_data())

    assert "score" in out and "confidence" in out and "components" in out
    assert np.isfinite(out["score"])
    assert np.isfinite(out["confidence"])
    assert -1.0 <= out["score"] <= 1.0
    assert 0.0 <= out["confidence"] <= 1.0
