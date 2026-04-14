"""Signal 28: Insider flow — net insider buy/sell activity."""

import numpy as np
from signals.base import Signal, SignalResult


class InsiderSignal(Signal):
    name = "insider_flow"
    category = "fundamental"
    data_source = "yfinance"
    refresh_rate = "10min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        insider = data.get("insider")

        if insider is None:
            return SignalResult(0.0, 0.0, {"error": "no insider data"})

        buys = getattr(insider, "recent_buys", 0)
        sells = getattr(insider, "recent_sells", 0)
        buy_val = getattr(insider, "total_buy_value", 0)
        sell_val = getattr(insider, "total_sell_value", 0)
        sentiment = getattr(insider, "net_sentiment", "neutral")

        # Net transaction count
        net_count = buys - sells
        total_count = buys + sells

        if total_count == 0:
            return SignalResult(0.0, 0.0, {"error": "no recent insider transactions"})

        # Score from count ratio
        count_score = np.clip(net_count / max(total_count, 1), -1, 1)

        # Score from value ratio
        net_value = buy_val - sell_val
        total_value = buy_val + sell_val
        value_score = np.clip(net_value / max(total_value, 1), -1, 1) if total_value > 0 else 0

        # Sentiment override
        if sentiment == "bullish":
            sent_adj = 0.2
        elif sentiment == "bearish":
            sent_adj = -0.2
        else:
            sent_adj = 0.0

        score = count_score * 0.3 + value_score * 0.5 + sent_adj
        score = np.clip(score, -1, 1)
        confidence = min(1.0, total_count / 5.0 * 0.5 + abs(score) * 0.5)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "buys": buys,
                "sells": sells,
                "buy_value": float(buy_val),
                "sell_value": float(sell_val),
                "sentiment": sentiment,
            },
        )
