"""Signal 45: Market-level momentum — SPY/QQQ trend direction and strength."""

import numpy as np
from signals.base import Signal, SignalResult


class MarketMomentumSignal(Signal):
    name = "market_momentum"
    category = "macro"
    data_source = "mda_candles"
    refresh_rate = "every_round"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        spy_candles = data.get("spy_candles")

        if spy_candles is None or len(spy_candles) < 21:
            return SignalResult(0.0, 0.0, {"error": "insufficient SPY candle data"})

        components = {}
        scores = []

        try:
            closes = np.array(
                [c.close for c in spy_candles if c.close], dtype=float
            )
        except (AttributeError, TypeError):
            return SignalResult(0.0, 0.0, {"error": "invalid candle data"})

        if len(closes) < 21:
            return SignalResult(0.0, 0.0, {"error": "insufficient closes"})

        # EMA 10 / EMA 50 cross on SPY
        ema10 = self._ema(closes, 10)
        ema50 = self._ema(closes, min(50, len(closes)))

        if ema50 > 0:
            ema_ratio = (ema10 - ema50) / ema50
            ema_score = np.clip(ema_ratio * 50, -1, 1)
            scores.append(ema_score * 0.4)
            components["spy_ema_ratio"] = float(round(ema_ratio * 100, 3))

        # 5-day ROC
        roc5 = (closes[-1] / closes[-6]) - 1.0 if len(closes) > 5 else 0.0
        roc_score = np.clip(roc5 / 0.03, -1, 1)
        scores.append(roc_score * 0.3)
        components["spy_roc5"] = float(round(roc5 * 100, 2))

        # 20-day ROC
        roc20 = (closes[-1] / closes[-21]) - 1.0
        roc20_score = np.clip(roc20 / 0.06, -1, 1)
        scores.append(roc20_score * 0.3)
        components["spy_roc20"] = float(round(roc20 * 100, 2))

        score = np.clip(sum(scores), -1, 1)
        confidence = min(1.0, abs(score) * 1.5 + 0.3)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        if len(data) == 0:
            return 0.0
        alpha = 2.0 / (period + 1)
        ema = data[0]
        for val in data[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return float(ema)
