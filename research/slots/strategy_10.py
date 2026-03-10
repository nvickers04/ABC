def scan(candles, symbol):
    if len(candles) < 10:
        return None
    c = candles[-1]
    if c['close'] > c['open'] and (c['high'] - c['low']) / c['close'] > 0.04:
        return {
            'signal': 'buy_call',
            'strike': round(c['close'] * 1.02, 2),
            'reason': 'strong_up_move_in_extreme_vol'
        }
    return None