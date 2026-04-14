"""Signal 20: ATM straddle cost — expected move vs actual recent moves."""

import numpy as np
from signals.base import Signal, SignalResult


class StraddleCostSignal(Signal):
    name = "straddle_cost"
    category = "volatility"
    data_source = "mda_options"
    refresh_rate = "every_round"
    tier = 2

    def compute(self, symbol: str, data: dict) -> SignalResult:
        option_chain = data.get("option_chain")
        candles_daily = data.get("candles_daily")
        quote = data.get("quote")

        if option_chain is None or not option_chain.contracts:
            return SignalResult(0.0, 0.0, {"error": "no option chain"})

        # Find ATM call and put
        underlying_price = None
        if quote and quote.last:
            underlying_price = quote.last
        elif candles_daily and candles_daily.close:
            underlying_price = candles_daily.close[-1]

        if underlying_price is None or underlying_price <= 0:
            return SignalResult(0.0, 0.0, {"error": "no underlying price"})

        # Find nearest ATM strike
        atm_calls = sorted(
            [c for c in option_chain.calls() if c.mid and c.mid > 0],
            key=lambda c: abs(c.strike - underlying_price),
        )
        atm_puts = sorted(
            [c for c in option_chain.puts() if c.mid and c.mid > 0],
            key=lambda c: abs(c.strike - underlying_price),
        )

        if not atm_calls or not atm_puts:
            return SignalResult(0.0, 0.0, {"error": "no ATM options"})

        call_mid = atm_calls[0].mid
        put_mid = atm_puts[0].mid

        # Straddle cost as % of underlying
        straddle_pct = (call_mid + put_mid) / underlying_price * 100

        # Compare to actual recent moves
        if candles_daily is not None and len(candles_daily) >= 21:
            daily_close = np.array(candles_daily.close, dtype=float)
            daily_returns = np.abs(np.diff(daily_close[-21:]) / daily_close[-21:-1]) * 100
            avg_daily_move = np.mean(daily_returns)

            # Estimate same-DTE expected move from realized moves
            dte = atm_calls[0].dte or 30
            realized_expected_move = avg_daily_move * np.sqrt(dte)
        else:
            realized_expected_move = straddle_pct

        # Straddle cheap vs realized = buy vol (+1)
        # Straddle expensive vs realized = sell vol (-1)
        ratio = straddle_pct / realized_expected_move if realized_expected_move > 0 else 1.0
        score = np.clip(-(ratio - 1.0) * 3, -1, 1)

        confidence = min(1.0, abs(ratio - 1.0) * 2)

        return SignalResult(
            score=float(score),
            confidence=float(confidence),
            components={
                "straddle_pct": float(straddle_pct),
                "realized_expected_move": float(realized_expected_move),
                "ratio": float(ratio),
            },
        )
