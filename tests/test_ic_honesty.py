"""Tests for cadence-aware IC trust thresholds.

A 1-min signal with 50 rows must NOT pass the trust gate (its bars are
tightly autocorrelated, so 50 rows is not statistically meaningful).
A daily signal with 50 rows MUST pass.  This is the core fix that
prevents sub-daily signals from posting fake high-confidence IC after
just a handful of intraday rounds.
"""
from __future__ import annotations

import pytest


class TestCadenceAwareMinObs:
    def test_min_obs_for_resolutions(self):
        from signals.combiner import _min_obs_for

        class _S:
            def __init__(self, res):
                self.return_resolution = res

        assert _min_obs_for(_S("1min")) == 200
        assert _min_obs_for(_S("5min")) == 100
        assert _min_obs_for(_S("15min")) == 75
        assert _min_obs_for(_S("1h")) == 50
        assert _min_obs_for(_S("D")) == 30

    def test_unknown_resolution_falls_back_to_default(self):
        from signals.combiner import _min_obs_for, _IC_MIN_OBS

        class _S:
            return_resolution = "weird"

        assert _min_obs_for(_S()) == _IC_MIN_OBS

    def test_missing_signal_falls_back_to_default(self):
        from signals.combiner import _min_obs_for, _IC_MIN_OBS

        # SIGNAL_REGISTRY.get(name) can return None when a signal was
        # retired or renamed; the threshold lookup must still work.
        assert _min_obs_for(None) == _IC_MIN_OBS

    @pytest.mark.parametrize(
        "name,expected_min",
        [
            ("market_momentum", 30),    # macro / D
            ("momentum", 50),           # price / 1h
            ("iv_rv_spread", 50),       # volatility / 1h
            ("option_flow", 100),       # microstructure / 5min
            ("spread_dynamics", 200),   # microstructure / 1min (override)
            ("seasonality", 30),        # macro / D (override)
        ],
    )
    def test_real_signals_use_their_resolution(self, name, expected_min):
        # Force registry population.
        import signals  # noqa: F401
        import importlib, pkgutil
        for _, mod_name, _ in pkgutil.iter_modules(signals.__path__):
            try:
                importlib.import_module(f"signals.{mod_name}")
            except Exception:
                pass

        from signals.base import SIGNAL_REGISTRY
        from signals.combiner import _min_obs_for

        sig = SIGNAL_REGISTRY.get(name)
        assert sig is not None, f"signal {name} should be registered"
        assert _min_obs_for(sig) == expected_min
