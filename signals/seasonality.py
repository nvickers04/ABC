"""Signal 43: Calendar seasonality — day-of-week, OPEX proximity, monthly patterns."""

import datetime
import numpy as np
from signals.base import Signal, SignalResult


class SeasonalitySignal(Signal):
    name = "seasonality"
    category = "macro"
    data_source = "environment"
    refresh_rate = "every_round"
    tier = 1

    # Historical day-of-week biases (slight, from academic research)
    # Monday: slight negative, Tuesday-Thursday: neutral-positive, Friday: mixed
    _DOW_BIAS = {
        0: -0.15,  # Monday
        1: 0.05,   # Tuesday
        2: 0.10,   # Wednesday
        3: 0.05,   # Thursday
        4: -0.05,  # Friday
    }

    def compute(self, symbol: str, data: dict) -> SignalResult:
        env = data.get("environment")
        now = datetime.datetime.now()
        components = {}
        scores = []

        # Day-of-week effect
        dow = now.weekday()
        dow_score = self._DOW_BIAS.get(dow, 0.0)
        scores.append(dow_score)
        components["day_of_week"] = now.strftime("%A")
        components["dow_bias"] = dow_score

        # OPEX proximity — monthly options expiry is 3rd Friday
        # Gamma effects intensify near OPEX
        day = now.day
        # Find 3rd Friday of current month
        first_day = now.replace(day=1)
        first_friday = first_day + datetime.timedelta(
            days=(4 - first_day.weekday()) % 7
        )
        third_friday = first_friday + datetime.timedelta(weeks=2)
        days_to_opex = (third_friday - now).days

        if days_to_opex < 0:
            # Already past this month's OPEX, look at next month
            next_month = (now.month % 12) + 1
            next_year = now.year + (1 if next_month == 1 else 0)
            first_day_next = now.replace(year=next_year, month=next_month, day=1)
            first_friday_next = first_day_next + datetime.timedelta(
                days=(4 - first_day_next.weekday()) % 7
            )
            third_friday_next = first_friday_next + datetime.timedelta(weeks=2)
            days_to_opex = (third_friday_next - now).days

        components["days_to_opex"] = days_to_opex

        # Near OPEX (0-2 days): gamma pinning increases, slight negative for momentum
        if days_to_opex <= 1:
            opex_score = -0.2  # Pinning risk
        elif days_to_opex <= 3:
            opex_score = -0.1
        else:
            opex_score = 0.0
        scores.append(opex_score)

        # Month-of-year pattern (very slight)
        # Jan effect, sell-in-May, Santa rally
        month_bias = {
            1: 0.10,   # January effect
            2: 0.02,
            3: 0.02,
            4: 0.05,
            5: -0.05,  # Sell in May
            6: -0.03,
            7: 0.02,
            8: -0.03,
            9: -0.05,  # September weakness
            10: 0.0,
            11: 0.05,  # Pre-holiday
            12: 0.08,  # Santa rally
        }
        m_bias = month_bias.get(now.month, 0.0)
        scores.append(m_bias)
        components["month_bias"] = m_bias

        # Environment regime can modulate
        if env:
            regime = env.get("volatility_regime", "normal")
            components["vol_regime"] = regime

        score = np.clip(sum(scores), -1, 1)
        # Low confidence — seasonality is a weak signal
        confidence = 0.3

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components=components,
        )
