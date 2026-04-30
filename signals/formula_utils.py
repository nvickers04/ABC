"""Shared math helpers for signal formula consistency.

These helpers keep signal outputs comparable across families:
- bounded score maps centered around 0
- confidence driven by signal strength and data-quality factor
- common volatility/range normalizers
"""

from __future__ import annotations

import numpy as np


def bounded_tanh(value: float, scale: float = 1.0) -> float:
    """Map an arbitrary value to [-1, 1] using tanh."""
    return float(np.tanh(float(value) * float(scale)))


def bounded_clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Hard clip to score bounds."""
    return float(np.clip(float(value), float(lo), float(hi)))


def confidence_from_strength(
    score_abs: float,
    *,
    data_quality: float = 1.0,
    floor: float = 0.05,
) -> float:
    """Compute confidence from |score| and a [0,1] data quality multiplier."""
    dq = float(np.clip(data_quality, 0.0, 1.0))
    base = float(np.clip(score_abs, 0.0, 1.0))
    conf = floor + (1.0 - floor) * base * dq
    return float(np.clip(conf, 0.0, 1.0))


def safe_pct_change(new: float, old: float, eps: float = 1e-12) -> float:
    """Numerically stable percentage change."""
    denom = old if abs(old) > eps else np.sign(old) * eps if old != 0 else eps
    return float((new - old) / denom)


def rolling_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Simple rolling ATR estimate from OHLC arrays."""
    if len(close) < 2 or len(high) < 2 or len(low) < 2:
        return 0.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    if len(tr) == 0:
        return 0.0
    p = min(max(int(period), 1), len(tr))
    return float(np.mean(tr[-p:]))
