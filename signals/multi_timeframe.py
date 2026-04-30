"""Signal 9: Multi-timeframe alignment — trend direction on daily, hourly, and 5-min."""

import numpy as np
from signals.formula_utils import bounded_tanh, confidence_from_strength
from signals.base import Signal, SignalResult


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _trend_direction(close: np.ndarray) -> float:
    """Returns +1/-1/0 for bullish/bearish/neutral based on EMA cross."""
    if len(close) < 20:
        return 0.0
    ema_fast = _ema(close, 8)
    ema_slow = _ema(close, 20)
    diff = (ema_fast[-1] - ema_slow[-1]) / close[-1]
    if diff > 0.001:
        return 1.0
    elif diff < -0.001:
        return -1.0
    return 0.0


class MultiTimeframeSignal(Signal):
    name = "multi_timeframe"
    category = "price"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        candles = data.get("candles")  # Intraday (1-min or 5-min)
        candles_daily = data.get("candles_daily")

        if candles is None or len(candles) < 20:
            return SignalResult(0.0, 0.0, {"error": "insufficient data"})

        close = np.array(candles.close, dtype=float)

        # 5-min timeframe (from intraday candles)
        tf_5min = _trend_direction(close)

        # Hourly approximation (every 12th bar if 5-min, or every 60th if 1-min)
        step = max(1, len(close) // 12)
        hourly_close = close[::step] if step > 1 else close
        tf_hourly = _trend_direction(hourly_close) if len(hourly_close) >= 20 else tf_5min

        # Daily timeframe
        if candles_daily is not None and len(candles_daily) >= 20:
            daily_close = np.array(candles_daily.close, dtype=float)
            tf_daily = _trend_direction(daily_close)
        else:
            tf_daily = tf_5min

        alignment = tf_5min + tf_hourly + tf_daily
        score = bounded_tanh(alignment / 3.0, scale=1.3)

        # Confidence proportional to alignment
        aligned_count = sum(1 for t in [tf_5min, tf_hourly, tf_daily] if t != 0)
        agreement = abs(alignment) / max(aligned_count, 1)
        data_quality = min(1.0, len(close) / 90.0) * (1.0 if candles_daily is not None else 0.7)
        confidence = confidence_from_strength(
            abs(score),
            data_quality=data_quality * agreement * (aligned_count / 3.0),
            floor=0.02,
        )

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "tf_5min": float(tf_5min),
                "tf_hourly": float(tf_hourly),
                "tf_daily": float(tf_daily),
                "alignment": float(alignment),
            },
        )
