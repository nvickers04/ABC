"""Signal 40: News sentiment — aggregate news sentiment from MDA news API."""

import numpy as np
from signals.base import Signal, SignalResult


class NewsSentimentSignal(Signal):
    name = "news_sentiment"
    category = "macro"
    data_source = "mda_quotes"
    refresh_rate = "5min"
    tier = 1

    def compute(self, symbol: str, data: dict) -> SignalResult:
        news = data.get("news")

        if not news:
            return SignalResult(0.0, 0.0, {"error": "no news data"})

        # news is a NewsData dataclass with .items (list[NewsItem]) and .sentiment
        articles = getattr(news, "items", None) or []
        overall_sentiment = getattr(news, "sentiment", "neutral")
        article_count = len(articles)

        if article_count == 0:
            return SignalResult(0.0, 0.0, {"article_count": 0})

        # Map overall sentiment string to a numeric score
        sentiment_map = {"positive": 0.5, "negative": -0.5, "neutral": 0.0}
        score = sentiment_map.get(overall_sentiment, 0.0)

        # Scale by article count — more articles = stronger signal
        if article_count >= 5:
            score *= 1.5
        score = float(np.clip(score, -1, 1))

        confidence = min(1.0, article_count / 10.0 * 0.8)

        components = {
            "overall_sentiment": overall_sentiment,
            "article_count": article_count,
        }

        return SignalResult(
            score=score,
            confidence=float(confidence),
            components=components,
        )
